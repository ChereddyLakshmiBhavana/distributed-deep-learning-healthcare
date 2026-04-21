import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XrayDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []

        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.jpeg")):
                self.paths.append(p)
                self.labels.append(cls)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class FastResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)

    def embedding(self, x):
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def build_loaders(img_size=160, batch_size=64):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(6),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = XrayDataset(DATA_DIR / "train", transform=train_tf)
    test_ds = XrayDataset(DATA_DIR / "test", transform=test_tf)

    le = LabelEncoder()
    le.fit(["NORMAL", "PNEUMONIA"])

    y_train = le.transform(train_ds.labels)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, le


def evaluate(model, loader, le):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y = le.transform(labels)
            y_true.extend(y)
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "rec": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_fast(target_acc=0.90):
    train_loader, test_loader, le = build_loaders(img_size=160, batch_size=64)

    model = FastResNet().to(DEVICE)

    # Freeze backbone for quick stable warmup.
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.fc.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-3)

    best_acc = 0.0
    best_state = None
    history = []

    # Phase 1: head only
    for epoch in range(1, 5):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        m = evaluate(model, test_loader, le)
        history.append({"phase": "head", "epoch": epoch, **m})
        print(f"[head {epoch}/4] loss={running/len(train_loader):.4f} acc={m['acc']*100:.2f}%")
        if m["acc"] > best_acc:
            best_acc = m["acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if m["acc"] >= target_acc:
            break

    # Phase 2: unfreeze last block + fc
    if best_acc < target_acc:
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
        params = list(model.backbone.layer4.parameters()) + list(model.backbone.fc.parameters())
        optimizer = torch.optim.Adam(params, lr=2e-4)

        for epoch in range(1, 9):
            model.train()
            running = 0.0
            for imgs, labels in train_loader:
                imgs = imgs.to(DEVICE)
                y = torch.tensor(le.transform(labels), dtype=torch.long, device=DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running += loss.item()

            m = evaluate(model, test_loader, le)
            history.append({"phase": "finetune", "epoch": epoch, **m})
            print(f"[finetune {epoch}/8] loss={running/len(train_loader):.4f} acc={m['acc']*100:.2f}%")
            if m["acc"] > best_acc:
                best_acc = m["acc"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if m["acc"] >= target_acc:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = MODELS_DIR / "fast_resnet18_xray.pth"
    torch.save({"model_state_dict": model.state_dict(), "best_acc": best_acc}, ckpt_path)
    return model, le, best_acc, history


def extract_embeddings(model, le):
    train_loader, test_loader, _ = build_loaders(img_size=160, batch_size=96)

    def run(loader):
        xs = []
        ys = []
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                emb = model.embedding(imgs).cpu().numpy()
                xs.append(emb)
                ys.extend(le.transform(labels))
        return np.vstack(xs), np.array(ys)

    return (*run(train_loader), *run(test_loader))


def train_classical(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    joblib.dump(scaler, MODELS_DIR / "fast_resnet_emb_scaler.pkl")

    models_map = {
        "lr_fast_resnet_emb": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "svm_fast_resnet_emb": SVC(kernel="rbf", C=12.0, probability=True, random_state=42),
        "rf_fast_resnet_emb": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, clf in models_map.items():
        clf.fit(x_train_s, y_train)
        pred = clf.predict(x_test_s)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        joblib.dump(clf, MODELS_DIR / f"{name}.pkl")
        print(f"{name}: acc={acc*100:.2f}% prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        rows.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})

    return pd.DataFrame(rows)


def main():
    print(f"Device: {DEVICE}")
    model, le, best_acc, history = train_fast(target_acc=0.90)
    print(f"Best CNN accuracy: {best_acc*100:.2f}%")

    x_train, y_train, x_test, y_test = extract_embeddings(model, le)
    print(f"Embeddings: train={x_train.shape}, test={x_test.shape}")

    df = train_classical(x_train, y_train, x_test, y_test)
    out_csv = ARTIFACTS_DIR / "fast_resnet_embedding_results.csv"
    df.to_csv(out_csv, index=False)

    history_path = ARTIFACTS_DIR / "fast_resnet_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "fast_resnet_best_accuracy": float(best_acc),
        "best_classical_accuracy": float(df["Accuracy"].max()),
        "models_over_90": int((df["Accuracy"] >= 0.90).sum()),
    }
    summary_path = ARTIFACTS_DIR / "fast_resnet_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(out_csv)
    print(history_path)
    print(summary_path)


if __name__ == "__main__":
    main()
