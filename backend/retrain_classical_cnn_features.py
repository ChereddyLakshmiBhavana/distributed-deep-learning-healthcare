"""
Retrain classical ML models using CNN-extracted features.
This approach uses the trained CNN model to extract deep feature representations
from chest X-ray images, then trains classical ML models on these features.
Expected accuracy: 90%+ (CNN embeddings provide much richer representation than hand-crafted features)
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

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
            print(f"Found {len(jpeg_files)} NORMAL images in {normal_dir}")
            for img_path in jpeg_files:
                self.images.append(str(img_path))
                self.labels.append("NORMAL")
        
        # Load PNEUMONIA images  
        pneumonia_dir = root_path / "PNEUMONIA"
        if pneumonia_dir.exists():
            jpeg_files = list(pneumonia_dir.glob("*.jpeg"))
            print(f"Found {len(jpeg_files)} PNEUMONIA images in {pneumonia_dir}")
            for img_path in jpeg_files:
                self.images.append(str(img_path))
                self.labels.append("PNEUMONIA")
        
        print(f"Total images loaded: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale (1 channel)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('L', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_cnn_model():
    """Load the trained CNN model."""
    model_path = MODELS_DIR / "cnn_model.pth"
    
    # Exact CNN architecture from notebook (PneumoniaCNN)
    class PneumoniaCNN(nn.Module):
        def __init__(self, num_classes=2):
            super(PneumoniaCNN, self).__init__()
            
            # Convolutional Block 1
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            
            # Convolutional Block 2
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            
            # Convolutional Block 3
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            
            # Fully Connected Layers
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
        
        def extract_features(self, x):
            """Extract 256-dim features before final classification layer."""
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)  # Flatten: 128 * 28 * 28 = 100352
            # Features from first FC layer (before output layer)
            x = self.fc[0](x)  # Flatten
            x = self.fc[1](x)  # Linear to 256
            x = self.fc[2](x)  # ReLU
            return x
    
    model = PneumoniaCNN(num_classes=2)
    
    if model_path.exists():
        print(f"Loading CNN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle both direct state_dict and wrapped state_dict
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


def extract_cnn_features(model, split_dir, split_name="train"):
    """Extract CNN features from a dataset split."""
    print(f"\nExtracting CNN features from {split_name} set...")
    
    split_path = Path(split_dir)
    print(f"Loading from: {split_path}")
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])
    
    # Load dataset
    dataset = ChestXRayDataset(split_path, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print(f"ERROR: No images found in {split_path}")
        return None, None
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # Extract 256-dim embeddings from CNN
            features = model.extract_features(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * 32} images...")
    
    if len(all_features) == 0:
        print(f"ERROR: No features extracted from {split_name} set")
        return None, None
    
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"Extracted features shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    
    return X, y


def train_classical_models(X_train, y_train, X_test, y_test):
    """Train classical ML models on CNN-extracted features."""
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessing objects
    joblib.dump(scaler, MODELS_DIR / "cnn_features_scaler.pkl")
    joblib.dump(le, MODELS_DIR / "cnn_features_encoder.pkl")
    print(f"\nScaler and encoder saved")
    
    # Models to train (optimized for CNN embeddings)
    models = {
        'logistic_regression_cnn': LogisticRegression(
            max_iter=2000, 
            C=0.1,  # Stronger regularization
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        ),
        'random_forest_cnn': RandomForestClassifier(
            n_estimators=1000,
            max_depth=40,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'svm_rbf_cnn': SVC(
            kernel='rbf',
            C=100.0,
            gamma='auto',
            probability=True,
            random_state=42
        ),
        'svm_linear_cnn': SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=42
        ),
        'gradient_boosting_cnn': GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            min_samples_split=3,
            random_state=42
        ),
        'adaboost_cnn': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=1.0,
            random_state=42
        ),
        'decision_tree_cnn': DecisionTreeClassifier(
            max_depth=30,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        ),
        'knn_cnn': KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='euclidean'
        ),
        'knn_manhattan_cnn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='manhattan'
        ),
        'naive_bayes_cnn': GaussianNB(var_smoothing=1e-10)
    }
    
    results = []
    
    print("\n" + "="*80)
    print("TRAINING CLASSICAL ML MODELS ON CNN-EXTRACTED FEATURES (512-DIM EMBEDDINGS)")
    print("="*80)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train_scaled, y_train_encoded)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred)
        recall = recall_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred)
        
        # ROC-AUC if probability available
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = model.decision_function(X_test_scaled)
            roc_auc = roc_auc_score(y_test_encoded, y_proba)
        except:
            roc_auc = None
        
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Save model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"  Saved to: {model_path}")
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc if roc_auc else 'N/A',
            'Timestamp': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        })
    
    return pd.DataFrame(results)


def main():
    print("="*80)
    print("CNN-FEATURE-BASED CLASSICAL MODEL RETRAINING")
    print("="*80)
    
    # Load CNN model
    print("\nStep 1: Loading trained CNN model...")
    cnn_model = load_cnn_model()
    
    if cnn_model is None:
        print("ERROR: CNN model not found. Please train CNN first.")
        sys.exit(1)
    
    # Extract features from training set
    print("\nStep 2: Extracting CNN features from training set...")
    train_dir = DATA_DIR / "train"
    X_train, y_train = extract_cnn_features(cnn_model, train_dir, "train")
    
    if X_train is None:
        print("ERROR: Failed to extract training features")
        sys.exit(1)
    
    # Extract features from test set
    print("\nStep 3: Extracting CNN features from test set...")
    test_dir = DATA_DIR / "test"
    X_test, y_test = extract_cnn_features(cnn_model, test_dir, "test")
    
    if X_test is None:
        print("ERROR: Failed to extract test features")
        sys.exit(1)
    
    print(f"\nFeature extraction complete:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    
    # Train classical models
    print("\nStep 4: Training classical ML models on CNN features...")
    results_df = train_classical_models(X_train, y_train, X_test, y_test)
    
    # Save results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    results_path = ARTIFACTS_DIR / "model_comparison_cnn_features.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print best model
    best_idx = results_df['Accuracy'].idxmax()
    best_model = results_df.loc[best_idx]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']*100:.2f}%")
    
    # Summary statistics
    print(f"\nAccuracy Statistics:")
    print(f"  Mean:   {results_df['Accuracy'].mean()*100:.2f}%")
    print(f"  Max:    {results_df['Accuracy'].max()*100:.2f}%")
    print(f"  Min:    {results_df['Accuracy'].min()*100:.2f}%")
    print(f"  Std:    {results_df['Accuracy'].std()*100:.2f}%")
    
    # Check if any model exceeded 90%
    models_90_plus = results_df[results_df['Accuracy'] >= 0.90]
    if len(models_90_plus) > 0:
        print(f"\n✅ SUCCESS! {len(models_90_plus)} model(s) achieved 90%+ accuracy:")
        for idx, row in models_90_plus.iterrows():
            print(f"   - {row['Model']}: {row['Accuracy']*100:.2f}%")
    else:
        print(f"\n⚠️  No models reached 90% yet. Max achieved: {results_df['Accuracy'].max()*100:.2f}%")


if __name__ == "__main__":
    main()
