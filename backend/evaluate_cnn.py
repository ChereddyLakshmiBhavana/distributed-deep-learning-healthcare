"""
Evaluate CNN model performance and use it for direct predictions.
Then ensemble CNN predictions with classical ML models.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        root_path = Path(root_dir)
        
        normal_dir = root_path / "NORMAL"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.jpeg"):
                self.images.append(str(img_path))
                self.labels.append("NORMAL")
        
        pneumonia_dir = root_path / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob("*.jpeg"):
                self.images.append(str(img_path))
                self.labels.append("PNEUMONIA")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('L')
        except:
            image = Image.new('L', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_cnn_model():
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
    
    model = PneumoniaCNN(num_classes=2)
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(DEVICE)
        model.eval()
        return model
    
    return None


def evaluate_cnn():
    print("\n" + "="*80)
    print("EVALUATING CNN MODEL")
    print("="*80)
    
    cnn = load_cnn_model()
    if cnn is None:
        print("CNN model not found")
        return None
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load test data
    test_dataset = ChestXRayDataset(DATA_DIR / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    le = LabelEncoder()
    le.fit(['NORMAL', 'PNEUMONIA'])
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = cnn(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(le.transform(labels))
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\nCNN Test Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return acc, prec, rec, f1


if __name__ == "__main__":
    evaluate_cnn()
