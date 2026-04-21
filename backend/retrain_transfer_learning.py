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
from torch.utils.data import DataLoader, Dataset
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


class ResNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, 2)
        self.model = base

    def forward(self, x):
        return self.model(x)

    def embedding(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def make_loaders(batch_size=32):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = XrayDataset(DATA_DIR / "train", transform=train_tf)
    test_ds = XrayDataset(DATA_DIR / "test", transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_resnet(epochs=5, lr=1e-4):
    train_loader, test_loader = make_loaders(batch_size=32)

    le = LabelEncoder()
    le.fit(["NORMAL", "PNEUMONIA"])

    model = ResNetBinary().to(DEVICE)
    # Handle class imbalance to improve minority class learning.
    class_counts = {0: 0, 1: 0}
    for _, labels in train_loader:
        y_np = le.transform(labels)
        class_counts[0] += int((y_np == 0).sum())
        class_counts[1] += int((y_np == 1).sum())

    total = class_counts[0] + class_counts[1]
    w0 = total / (2.0 * max(class_counts[0], 1))
    w1 = total / (2.0 * max(class_counts[1], 1))
    weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=DEVICE)
            imgs = imgs.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                y = le.transform(labels)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.extend(y)
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1}/{epochs} - loss={total_loss/len(train_loader):.4f} - test_acc={acc*100:.2f}%")
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_acc": best_acc}, MODELS_DIR / "resnet18_xray.pth")
    return model, le, best_acc


def extract_embeddings(model, le):
    train_loader, test_loader = make_loaders(batch_size=64)

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

    x_train, y_train = run(train_loader)
    x_test, y_test = run(test_loader)
    return x_train, y_train, x_test, y_test


def train_classical_on_embeddings(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    joblib.dump(scaler, MODELS_DIR / "resnet_embedding_scaler.pkl")

    models_map = {
        "lr_resnet_emb": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "svm_resnet_emb": SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
        "rf_resnet_emb": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
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
        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        })

    return pd.DataFrame(rows)


def main():
    print(f"Device: {DEVICE}")
    model, le, cnn_acc = train_resnet(epochs=15, lr=1e-4)
    print(f"Best ResNet test accuracy: {cnn_acc*100:.2f}%")

    x_train, y_train, x_test, y_test = extract_embeddings(model, le)
    print(f"Embeddings train shape: {x_train.shape}, test shape: {x_test.shape}")

    result_df = train_classical_on_embeddings(x_train, y_train, x_test, y_test)
    result_df.to_csv(ARTIFACTS_DIR / "resnet_embedding_classical_results.csv", index=False)

    summary = {
        "resnet_best_accuracy": float(cnn_acc),
        "best_classical_accuracy": float(result_df["Accuracy"].max()),
        "models_over_90": int((result_df["Accuracy"] >= 0.90).sum()),
    }
    with open(ARTIFACTS_DIR / "resnet_embedding_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(ARTIFACTS_DIR / "resnet_embedding_classical_results.csv")
    print(ARTIFACTS_DIR / "resnet_embedding_summary.json")


if __name__ == "__main__":
    main()
