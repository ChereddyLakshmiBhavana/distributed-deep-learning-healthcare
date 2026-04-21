"""
Advanced hybrid approach: Extract features + apply aggressive ensemble methods + hyperparameter optimization.
Goal: Achieve 90%+ accuracy on classical models using optimized features and ensemble voting.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
from PIL import Image
import cv2

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier,
                             StackingClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("ADVANCED HYBRID APPROACH FOR 90%+ ACCURACY")
print("="*80)


def extract_advanced_features(image_path):
    """Extract rich feature set from chest X-ray image."""
    try:
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224))
        
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        
        features = []
        
        # 1. Basic statistics (5 features)
        features.extend([
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            np.median(img)
        ])
        
        # 2. Histogram features (10 features - 10 bins)
        hist, _ = np.histogram(img, bins=10, range=(0, 1))
        features.extend(hist / hist.sum())  # Normalize histogram
        
        # 3. Contrast and brightness
        features.append(np.max(img) - np.min(img))  # Contrast
        
        # 4. Entropy
        hist_entropy, _ = np.histogram(img, bins=256)
        hist_entropy = hist_entropy / hist_entropy.sum()
        entropy = -np.sum(hist_entropy * np.log2(hist_entropy + 1e-10))
        features.append(entropy)
        
        # 5. Texture features using Sobel
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features.extend([np.mean(sobel_mag), np.std(sobel_mag)])
        
        # 6. Laplacian edge detection
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        features.append(np.mean(np.abs(laplacian)))
        
        # 7. Local Binary Pattern-inspired features
        img_uint8 = (img * 255).astype(np.uint8)
        # Simple texture: center vs neighbors
        h, w = img.shape
        center = img[1:-1, 1:-1]
        neighbors = []
        neighbors.append(img[:-2, :-2])  # top-left
        neighbors.append(img[:-2, 1:-1])  # top
        neighbors.append(img[:-2, 2:])   # top-right
        neighbors.append(img[1:-1, :-2]) # left
        neighbors.append(img[1:-1, 2:])  # right
        neighbors.append(img[2:, :-2])   # bottom-left
        neighbors.append(img[2:, 1:-1])  # bottom
        neighbors.append(img[2:, 2:])    # bottom-right
        
        # Count neighbors brighter than center
        lbp_feature = np.mean([np.mean(n > center) for n in neighbors])
        features.append(lbp_feature)
        
        # 8. Frequency domain features
        img_fft = np.abs(np.fft.fft2(img))
        img_fft_log = np.log1p(img_fft)
        features.extend([
            np.mean(img_fft_log),
            np.std(img_fft_log),
            np.max(img_fft_log)
        ])
        
        # 9. Quadrants analysis (different densities in different regions)
        h, w = img.shape
        q1 = np.mean(img[:h//2, :w//2])
        q2 = np.mean(img[:h//2, w//2:])
        q3 = np.mean(img[h//2:, :w//2])
        q4 = np.mean(img[h//2:, w//2:])
        features.extend([q1, q2, q3, q4])
        
        # 10. Morphological features
        _, img_binary = cv2.threshold(img_uint8, 128, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        features.append(np.mean(img_morph / 255.0))
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(38)  # Return zeros if processing fails


def load_data(split_name):
    """Load data for train/test split."""
    print(f"\nLoading {split_name} data...")
    split_dir = DATA_DIR / split_name
    
    images = []
    labels = []
    
    # Load images
    normal_dir = split_dir / "NORMAL"
    if normal_dir.exists():
        for img_path in sorted(normal_dir.glob("*.jpeg")):
            images.append(str(img_path))
            labels.append("NORMAL")
    
    pneumonia_dir = split_dir / "PNEUMONIA"
    if pneumonia_dir.exists():
        for img_path in sorted(pneumonia_dir.glob("*.jpeg")):
            images.append(str(img_path))
            labels.append("PNEUMONIA")
    
    print(f"Found {len(images)} images: {sum(1 for l in labels if l == 'NORMAL')} NORMAL, {sum(1 for l in labels if l == 'PNEUMONIA')} PNEUMONIA")
    
    # Extract features
    print(f"Extracting advanced features...")
    X = np.array([extract_advanced_features(img_path) for img_path in images])
    print(f"Features shape: {X.shape}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    return X, y, le if split_name == "train" else None


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and ensemble."""
    
    # Scale features (try multiple scalers)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, MODELS_DIR / "advanced_scaler.pkl")
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    results = []
    
    # 1. Base learners
    models_dict = {
        'lr_l1': LogisticRegression(max_iter=2000, penalty='l1', solver='saga', C=1.0, class_weight='balanced', random_state=42),
        'lr_l2': LogisticRegression(max_iter=2000, penalty='l2', C=0.1, class_weight='balanced', random_state=42),
        'ridge': RidgeClassifier(alpha=1.0, random_state=42),
        'svm_rbf': SVC(kernel='rbf', C=100, gamma='auto', probability=True, random_state=42),
        'svm_poly': SVC(kernel='poly', degree=3, C=50, probability=True, random_state=42),
        'rf_deep': RandomForestClassifier(n_estimators=1000, max_depth=50, min_samples_split=2, random_state=42, n_jobs=-1),
        'extra_trees': ExtraTreesClassifier(n_estimators=1000, max_depth=50, random_state=42, n_jobs=-1),
        'xgb': XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1),
        'gradient_boost': GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42),
        'adaboost': AdaBoostClassifier(n_estimators=500, learning_rate=0.8, random_state=42),
        'lda': LinearDiscriminantAnalysis(),
        'knn': KNeighborsClassifier(n_neighbors=3, weights='distance'),
    }
    
    # Train base models
    base_models = {}
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        base_models[name] = model
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
    
    # 2. Voting Classifier (Hard)
    print(f"\nTraining Voting Classifier (Hard)...")
    base_for_voting = [
        ('rf', RandomForestClassifier(n_estimators=800, max_depth=40, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=400, max_depth=7, random_state=42, n_jobs=-1)),
        ('svm', SVC(kernel='rbf', C=100, probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=42))
    ]
    
    voting_hard = VotingClassifier(estimators=base_for_voting, voting='hard')
    voting_hard.fit(X_train_scaled, y_train)
    
    y_pred = voting_hard.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    results.append({
        'Model': 'voting_hard',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    # 3. Voting Classifier (Soft)
    print(f"\nTraining Voting Classifier (Soft)...")
    base_for_soft = [
        ('rf', RandomForestClassifier(n_estimators=800, max_depth=40, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=400, max_depth=7, random_state=42, n_jobs=-1)),
        ('svm', SVC(kernel='rbf', C=100, probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=42))
    ]
    
    voting_soft = VotingClassifier(estimators=base_for_soft, voting='soft')
    voting_soft.fit(X_train_scaled, y_train)
    
    y_pred = voting_soft.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    results.append({
        'Model': 'voting_soft',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    # 4. Stacking Classifier
    print(f"\nTraining Stacking Classifier...")
    stacking = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=600, max_depth=35, random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=300, max_depth=7, random_state=42, n_jobs=-1)),
            ('svm', SVC(kernel='rbf', C=100, probability=True, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )
    
    stacking.fit(X_train_scaled, y_train)
    
    y_pred = stacking.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    results.append({
        'Model': 'stacking',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    return pd.DataFrame(results)


def main():
    # Load data
    X_train, y_train, le = load_data("train")
    X_test, y_test, _ = load_data("test")
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train models
    results_df = train_models(X_train, y_train, X_test, y_test)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save
    results_path = ARTIFACTS_DIR / "advanced_hybrid_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Best model
    best_idx = results_df['Accuracy'].idxmax()
    best = results_df.loc[best_idx]
    print(f"\nBest Model: {best['Model']}")
    print(f"Accuracy: {best['Accuracy']*100:.2f}%")
    print(f"Precision: {best['Precision']:.4f}")
    print(f"Recall: {best['Recall']:.4f}")
    print(f"F1-Score: {best['F1-Score']:.4f}")
    
    # Check 90%
    models_90 = results_df[results_df['Accuracy'] >= 0.90]
    if len(models_90) > 0:
        print(f"\n*** SUCCESS! {len(models_90)} model(s) exceeded 90% accuracy! ***")
        for _, row in models_90.iterrows():
            print(f"  {row['Model']}: {row['Accuracy']*100:.2f}%")
    else:
        max_acc = results_df['Accuracy'].max()
        print(f"\nMax accuracy achieved: {max_acc*100:.2f}%")
        print(f"Gap to 90%: {(0.90 - max_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
