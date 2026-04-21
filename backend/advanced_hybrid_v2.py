"""
Streamlined advanced training - focus on best-performing ensemble models.
Writes results directly to CSV for easy parsing.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from PIL import Image
import cv2
import json

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)


def extract_features(image_path):
    """Extract 38 advanced features."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(38)
        
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        
        features = []
        
        # Basic stats (5)
        features.extend([np.mean(img), np.std(img), np.min(img), np.max(img), np.median(img)])
        
        # Histogram (10)
        hist, _ = np.histogram(img, bins=10, range=(0, 1))
        features.extend(hist / (hist.sum() + 1e-10))
        
        # Contrast and edge features (4)
        features.append(np.max(img) - np.min(img))
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features.extend([np.mean(sobel_mag), np.std(sobel_mag), np.mean(np.abs(cv2.Laplacian(img, cv2.CV_64F)))])
        
        # Entropy (1)
        hist_e, _ = np.histogram(img, bins=256)
        hist_e = hist_e / (hist_e.sum() + 1e-10)
        features.append(-np.sum(hist_e * np.log2(hist_e + 1e-10)))
        
        # Texture and frequency (8)
        h, w = img.shape
        lbp = np.mean([(img[1:-1, 1:-1] > img[:-2, :-2]).astype(float)])
        features.append(lbp)
        
        fft = np.abs(np.fft.fft2(img))
        fft_log = np.log1p(fft)
        features.extend([np.mean(fft_log), np.std(fft_log)])
        
        # Quadrants (4)
        q1 = np.mean(img[:h//2, :w//2])
        q2 = np.mean(img[:h//2, w//2:])
        q3 = np.mean(img[h//2:, :w//2])
        q4 = np.mean(img[h//2:, w//2:])
        features.extend([q1, q2, q3, q4])
        
        return np.array(features)
    except:
        return np.zeros(38)


def load_data(split_name):
    """Load features and labels."""
    split_dir = DATA_DIR / split_name
    images, labels = [], []
    
    for label_dir in [split_dir / "NORMAL", split_dir / "PNEUMONIA"]:
        if label_dir.exists():
            label = label_dir.name
            for img_path in sorted(label_dir.glob("*.jpeg")):
                images.append(str(img_path))
                labels.append(label)
    
    X = np.array([extract_features(img) for img in images])
    le = LabelEncoder()
    y = le.fit_transform(labels) if split_name == "train" else le.fit(['NORMAL', 'PNEUMONIA']).transform(labels)
    
    return X, y


def main():
    print("ADVANCED HYBRID TRAINING - STREAMLINED")
    print("="*60)
    
    # Load data
    print("Loading train data...", end=" ", flush=True)
    X_train, y_train = load_data("train")
    print(f"Done: {X_train.shape}")
    
    print("Loading test data...", end=" ", flush=True)
    X_test, y_test = load_data("test")
    print(f"Done: {X_test.shape}")
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "advanced_scaler.pkl")
    
    results = []
    
    # Voting Hard
    print("\nTraining Voting (Hard)...", end=" ", flush=True)
    v_hard = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=800, max_depth=40, n_jobs=-1, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=400, max_depth=7, random_state=42)),
        ('svm', SVC(kernel='rbf', C=100, probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=42))
    ], voting='hard')
    v_hard.fit(X_train_s, y_train)
    y_p = v_hard.predict(X_test_s)
    acc = accuracy_score(y_test, y_p)
    print(f"Accuracy: {acc*100:.2f}%")
    results.append({'Model': 'voting_hard', 'Accuracy': acc, 'Precision': precision_score(y_test, y_p), 'Recall': recall_score(y_test, y_p), 'F1': f1_score(y_test, y_p)})
    
    # Voting Soft
    print("Training Voting (Soft)...", end=" ", flush=True)
    v_soft = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=800, max_depth=40, n_jobs=-1, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=400, max_depth=7, random_state=42)),
        ('svm', SVC(kernel='rbf', C=100, probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=42))
    ], voting='soft')
    v_soft.fit(X_train_s, y_train)
    y_p = v_soft.predict(X_test_s)
    acc = accuracy_score(y_test, y_p)
    print(f"Accuracy: {acc*100:.2f}%")
    results.append({'Model': 'voting_soft', 'Accuracy': acc, 'Precision': precision_score(y_test, y_p), 'Recall': recall_score(y_test, y_p), 'F1': f1_score(y_test, y_p)})
    
    # Stacking
    print("Training Stacking...", end=" ", flush=True)
    stack = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=600, max_depth=35, n_jobs=-1, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=300, max_depth=7, random_state=42)),
            ('svm', SVC(kernel='rbf', C=100, probability=True, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )
    stack.fit(X_train_s, y_train)
    y_p = stack.predict(X_test_s)
    acc = accuracy_score(y_test, y_p)
    print(f"Accuracy: {acc*100:.2f}%")
    results.append({'Model': 'stacking', 'Accuracy': acc, 'Precision': precision_score(y_test, y_p), 'Recall': recall_score(y_test, y_p), 'F1': f1_score(y_test, y_p)})
    
    # Results
    df = pd.DataFrame(results)
    df_path = ARTIFACTS_DIR / "advanced_hybrid_results.csv"
    df.to_csv(df_path, index=False)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(df.to_string(index=False))
    
    best_idx = df['Accuracy'].idxmax()
    best_acc = df.loc[best_idx, 'Accuracy']
    best_model = df.loc[best_idx, 'Model']
    
    print(f"\nBest: {best_model} - {best_acc*100:.2f}%")
    print(f"Target: 90%, Gap: {(0.90-best_acc)*100:.2f}%")
    
    # Save summary
    summary = {
        'best_model': best_model,
        'best_accuracy': float(best_acc),
        'target': 0.90,
        'achieved_90_plus': best_acc >= 0.90
    }
    
    with open(ARTIFACTS_DIR / "advanced_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {df_path}")


if __name__ == "__main__":
    main()
