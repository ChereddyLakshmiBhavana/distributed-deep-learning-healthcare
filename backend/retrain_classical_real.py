import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FEATURE_NAMES = [
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "median_intensity",
    "hist_bin_0",
    "hist_bin_1",
    "hist_bin_2",
    "hist_bin_3",
    "hist_bin_4",
    "hist_bin_5",
    "hist_bin_6",
    "hist_bin_7",
    "hist_bin_8",
    "hist_bin_9",
    "edge_density",
    "contrast",
]


def extract_features(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((64, 64))
    img_array = np.asarray(img, dtype=np.float32) / 255.0

    values = []
    values.append(float(np.mean(img_array)))
    values.append(float(np.std(img_array)))
    values.append(float(np.min(img_array)))
    values.append(float(np.max(img_array)))
    values.append(float(np.median(img_array)))

    hist, _ = np.histogram(img_array, bins=10, range=(0, 1))
    hist = hist / max(1, hist.sum()) * 100.0
    values.extend([float(v) for v in hist])

    edges = np.abs(np.diff(img_array))
    values.append(float(np.sum(edges) / img_array.size))
    values.append(float(img_array.max() - img_array.min()))

    return np.array(values, dtype=np.float32)


def load_split(split_dir: Path):
    X = []
    y = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        class_dir = split_dir / cls
        image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        for image_path in image_files:
            try:
                X.append(extract_features(image_path))
                y.append(cls)
            except Exception:
                continue
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_ROOT}")

    print("Loading training split...")
    X_train_raw, y_train_text = load_split(TRAIN_DIR)
    print("Loading test split...")
    X_test_raw, y_test_text = load_split(TEST_DIR)

    print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    models = {
        "logistic_regression_model": LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=42,
        ),
        "decision_tree_model": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest_model": RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "k-nearest_neighbors_model": KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="minkowski",
            p=2,
        ),
        "naive_bayes_model": GaussianNB(var_smoothing=1e-11),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary")
        rec = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")

        rows.append({
            "Model": model_name.replace("_model", "").replace("_", " ").title(),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        })

        joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")
        print(f"  Accuracy={acc:.4f}")

    # Save preprocessing artifacts expected by backend
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    pd.DataFrame(FEATURE_NAMES).to_csv(PROJECT_ROOT / "data" / "feature_names.csv", index=False, header=False)

    results_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison_results.csv", index=False)

    # Keep existing file names expected elsewhere
    np.save(PROJECT_ROOT / "data" / "X_train_scaled.npy", X_train)
    np.save(PROJECT_ROOT / "data" / "X_test_scaled.npy", X_test)
    np.save(PROJECT_ROOT / "data" / "y_train_encoded.npy", y_train)
    np.save(PROJECT_ROOT / "data" / "y_test_encoded.npy", y_test)

    print("\nFinal results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
