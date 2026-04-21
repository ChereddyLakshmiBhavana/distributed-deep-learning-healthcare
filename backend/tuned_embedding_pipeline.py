import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class XrayDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []

        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for path in sorted(cls_dir.glob(ext)):
                    self.paths.append(path)
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


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(img_size=160, batch_size=96):
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = XrayDataset(DATA_DIR / "train", transform=eval_tf)
    test_ds = XrayDataset(DATA_DIR / "test", transform=eval_tf)

    le = LabelEncoder()
    le.fit(["NORMAL", "PNEUMONIA"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, le


def load_or_train_backbone(le):
    ckpt_path = MODELS_DIR / "fast_resnet18_xray.pth"
    model = FastResNet().to(DEVICE)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded backbone from {ckpt_path}")
        return model

    print("fast_resnet18_xray.pth not found, training a quick fallback backbone...")
    train_loader, test_loader, _ = build_loaders(img_size=160, batch_size=64)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(4):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        print(f"  fallback epoch {epoch + 1}/4 complete")

    model.eval()
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved fallback backbone to {ckpt_path}")
    return model


def extract_embeddings(model, loader, le):
    x_out = []
    y_out = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            emb = model.embedding(imgs).cpu().numpy()
            x_out.append(emb)
            y_out.extend(le.transform(labels))
    return np.vstack(x_out), np.array(y_out)


def best_threshold_for_accuracy(y_true, y_scores):
    thresholds = np.linspace(0.2, 0.8, 61)
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def fit_and_evaluate(name, estimator, param_grid, x_train, y_train, x_val, y_val, x_test, y_test):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_

    threshold = 0.5
    val_best_acc = None
    if hasattr(best_model, "predict_proba"):
        val_scores = best_model.predict_proba(x_val)[:, 1]
        threshold, val_best_acc = best_threshold_for_accuracy(y_val, val_scores)

    final_model = search.best_estimator_
    final_model.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]))

    if hasattr(final_model, "predict_proba"):
        test_scores = final_model.predict_proba(x_test)[:, 1]
        y_pred = (test_scores >= threshold).astype(int)
    else:
        y_pred = final_model.predict(x_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "CV_Best_Accuracy": float(search.best_score_),
        "Threshold": float(threshold),
        "Val_Best_Threshold_Accuracy": float(val_best_acc) if val_best_acc is not None else None,
        "Best_Params": json.dumps(search.best_params_),
    }

    model_file = MODELS_DIR / f"tuned_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(final_model, model_file)
    return metrics


def main():
    seed_everything()
    print(f"Device: {DEVICE}")

    train_loader, test_loader, le = build_loaders()
    model = load_or_train_backbone(le)

    x_train_all, y_train_all = extract_embeddings(model, train_loader, le)
    x_test, y_test = extract_embeddings(model, test_loader, le)

    scaler = StandardScaler()
    x_train_all = scaler.fit_transform(x_train_all)
    x_test = scaler.transform(x_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_all,
    )

    joblib.dump(scaler, MODELS_DIR / "tuned_embedding_scaler.pkl")

    model_specs = [
        (
            "Random Forest",
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators": [300, 600],
                "max_depth": [None, 20, 40],
                "min_samples_leaf": [1, 2],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        ),
        (
            "SVM RBF",
            SVC(kernel="rbf", probability=True, random_state=SEED),
            {
                "C": [4.0, 8.0, 12.0],
                "gamma": ["scale", 0.01, 0.005],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=SEED),
            {
                "n_estimators": [200, 400],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=4000, solver="lbfgs", random_state=SEED),
            {
                "C": [0.5, 1.0, 2.0, 4.0],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "KNN",
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "manhattan"],
            },
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=SEED),
            {
                "max_depth": [8, 12, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        ),
    ]

    rows = []
    for name, estimator, param_grid in model_specs:
        print(f"\nTuning {name}...")
        metrics = fit_and_evaluate(
            name,
            estimator,
            param_grid,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
        )
        rows.append(metrics)
        print(f"  test accuracy: {metrics['Accuracy'] * 100:.2f}%")

    results_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

    results_path = ARTIFACTS_DIR / "tuned_embedding_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    over_90 = int((results_df["Accuracy"] >= 0.90).sum())
    summary = {
        "num_models": int(len(results_df)),
        "num_models_over_90": over_90,
        "best_model": str(results_df.iloc[0]["Model"]),
        "best_accuracy": float(results_df.iloc[0]["Accuracy"]),
    }
    summary_path = ARTIFACTS_DIR / "tuned_embedding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 84)
    print("TUNED EMBEDDING RESULTS")
    print("=" * 84)
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "CV_Best_Accuracy"]].to_string(index=False))
    print("=" * 84)
    print(f"Models >= 90% accuracy: {over_90}")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
