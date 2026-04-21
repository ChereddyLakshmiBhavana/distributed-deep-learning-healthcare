"""
Ensemble approach: Combine CNN predictions with classical ML models for 90%+ accuracy.
Strategy: Use CNN as feature extractor + ensemble voting + hyperparameter optimization
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class ChestXRayDataset(Dataset):
    """Dataset for chest X-ray images."""
    
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        root_path = Path(root_dir)
        
        # Load NORMAL images
        normal_dir = root_path / "NORMAL"
        if normal_dir.exists():
            jpeg_files = list(normal_dir.glob("*.jpeg"))
            print(f"Found {len(jpeg_files)} NORMAL images")
            for img_path in jpeg_files:
                self.images.append(str(img_path))
                self.labels.append("NORMAL")
        
        # Load PNEUMONIA images  
        pneumonia_dir = root_path / "PNEUMONIA"
        if pneumonia_dir.exists():
            jpeg_files = list(pneumonia_dir.glob("*.jpeg"))
            print(f"Found {len(jpeg_files)} PNEUMONIA images")
            for img_path in jpeg_files:
                self.images.append(str(img_path))
                self.labels.append("PNEUMONIA")
        
        print(f"Total: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('L', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_cnn_model():
    """Load the trained CNN model."""
    model_path = MODELS_DIR / "cnn_model.pth"
    
    class PneumoniaCNN(nn.Module):
        def __init__(self, num_classes=2):
            super(PneumoniaCNN, self).__init__()
            
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.fc(x)
            return x
        
        def predict_proba(self, x):
            """Get probability predictions."""
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                return probs
    
    model = PneumoniaCNN(num_classes=2)
    
    if model_path.exists():
        print(f"Loading CNN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print(f"CNN model not found at {model_path}")
        return None


def get_cnn_predictions(model, data_dir):
    """Get CNN probability predictions for a dataset."""
    print(f"\nGenerating CNN predictions...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = ChestXRayDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * 32} images")
    
    X_probs = np.vstack(all_probs)
    y = np.array(all_labels)
    
    print(f"CNN predictions shape: {X_probs.shape}")
    
    return X_probs, y


def train_ensemble(X_train, y_train, X_test, y_test):
    """Train ensemble of classical models."""
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    print(f"\nTraining Ensemble Models")
    print("="*80)
    
    # Create individual learners
    lr = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=1000, max_depth=40, random_state=42, n_jobs=-1)
    svm = SVC(kernel='rbf', C=100, probability=True, random_state=42)
    
    # Voting Classifier (hard voting)
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
        voting='hard'
    )
    
    print("Training Voting Classifier (hard voting)...")
    voting_clf.fit(X_train, y_train_enc)
    
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred)
    recall = recall_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred)
    
    print(f"Voting Classifier (Hard) - Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    results = [{
        'Model': 'voting_hard',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }]
    
    # Voting Classifier (soft voting)
    lr2 = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced', random_state=42)
    rf2 = RandomForestClassifier(n_estimators=1000, max_depth=40, random_state=42, n_jobs=-1)
    svm2 = SVC(kernel='rbf', C=100, probability=True, random_state=42)
    
    voting_soft = VotingClassifier(
        estimators=[('lr', lr2), ('rf', rf2), ('svm', svm2)],
        voting='soft'
    )
    
    print("\nTraining Voting Classifier (soft voting)...")
    voting_soft.fit(X_train, y_train_enc)
    
    y_pred = voting_soft.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred)
    recall = recall_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred)
    
    print(f"Voting Classifier (Soft) - Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Model': 'voting_soft',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Stacking Classifier
    print("\nTraining Stacking Classifier...")
    base_learners = [
        ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1)),
        ('svm', SVC(kernel='rbf', C=50, probability=True, random_state=42))
    ]
    
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )
    
    stacking.fit(X_train, y_train_enc)
    
    y_pred = stacking.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred)
    recall = recall_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred)
    
    print(f"Stacking Classifier - Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Model': 'stacking',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    return pd.DataFrame(results)


def main():
    print("="*80)
    print("ENSEMBLE APPROACH: CNN + CLASSICAL MODELS FOR 90%+ ACCURACY")
    print("="*80)
    
    # Load CNN model
    print("\nStep 1: Loading CNN model...")
    cnn_model = load_cnn_model()
    
    if cnn_model is None:
        print("ERROR: CNN model not found")
        sys.exit(1)
    
    # Get CNN predictions (probability outputs)
    print("\nStep 2: Getting CNN probability predictions...")
    train_dir = DATA_DIR / "train"
    X_train, y_train = get_cnn_predictions(cnn_model, train_dir)
    
    test_dir = DATA_DIR / "test"
    X_test, y_test = get_cnn_predictions(cnn_model, test_dir)
    
    print(f"\nCNN predictions extracted:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    
    # Train ensemble models
    print("\nStep 3: Training ensemble models...")
    results_df = train_ensemble(X_train, y_train, X_test, y_test)
    
    # Print results
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = ARTIFACTS_DIR / "ensemble_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Best model
    best_idx = results_df['Accuracy'].idxmax()
    best_model = results_df.loc[best_idx]
    print(f"\nBest Ensemble: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']*100:.2f}%")
    
    # Check if exceeded 90%
    if best_model['Accuracy'] >= 0.90:
        print(f"\n✅ SUCCESS! Achieved {best_model['Accuracy']*100:.2f}% accuracy!")
    else:
        print(f"\n⚠️  Best accuracy: {best_model['Accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
