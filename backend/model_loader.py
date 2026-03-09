"""
Model Loader Module
Handles loading of saved ML and DL models
"""

import joblib
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

class PneumoniaCNN(nn.Module):
    """CNN Model Architecture - Must match training definition"""
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


class ModelLoader:
    """Centralized model loading and prediction"""
    
    def __init__(self, models_dir='../models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_classical_models(self):
        """Load all classical ML models"""
        model_names = [
            'logistic_regression_model',
            'decision_tree_model',
            'random_forest_model',
            'k-nearest_neighbors_model',
            'naive_bayes_model'
        ]
        
        for model_name in model_names:
            model_path = self.models_dir / f'{model_name}.pkl'
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded: {model_name}")
            else:
                print(f"⚠ Not found: {model_name}")
        
        # Load scaler and label encoder
        scaler_path = self.models_dir / 'feature_scaler.pkl'
        encoder_path = self.models_dir / 'label_encoder.pkl'
        
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print("✓ Loaded: feature_scaler")
        
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            print("✓ Loaded: label_encoder")
    
    def load_cnn_model(self):
        """Load PyTorch CNN model"""
        model_path = self.models_dir / 'cnn_model.pth'
        
        if model_path.exists():
            # Initialize model
            cnn_model = PneumoniaCNN(num_classes=2).to(self.device)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
            cnn_model.eval()
            
            self.models['cnn_model'] = cnn_model
            print(f"✓ Loaded: CNN model on {self.device}")
        else:
            print("⚠ CNN model not found")
    
    def predict_classical(self, features, model_name):
        """Make prediction using classical ML model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Scale features
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 1.0
        
        # Decode label
        if self.label_encoder:
            label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            label = str(prediction)
        
        return {
            'prediction': label,
            'confidence': confidence,
            'model': model_name
        }
    
    def predict_cnn(self, image_tensor):
        """Make prediction using CNN model"""
        if 'cnn_model' not in self.models:
            raise ValueError("CNN model not loaded")
        
        model = self.models['cnn_model']
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Decode label
            if self.label_encoder:
                label = self.label_encoder.inverse_transform([predicted.item()])[0]
            else:
                label = str(predicted.item())
            
            return {
                'prediction': label,
                'confidence': float(confidence.item()),
                'model': 'cnn_model'
            }
    
    def list_available_models(self):
        """Return list of loaded models"""
        return list(self.models.keys())
