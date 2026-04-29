"""
Model Loader Module
Handles loading of saved ML and DL models
"""

import joblib
import torch
import torch.nn as nn
import numpy as np
import sys
import types
from pathlib import Path
import json
from torchvision import models, transforms


def _install_numpy_core_compat():
    """Provide a compatibility alias for pickles that reference numpy._core."""
    if 'numpy._core' in sys.modules:
        return

    core_alias = types.ModuleType('numpy._core')
    core_alias.__dict__.update(np.core.__dict__)
    sys.modules['numpy._core'] = core_alias
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = np.core.numeric
    try:
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    except AttributeError:
        pass
    setattr(np, '_core', core_alias)


_install_numpy_core_compat()


def _apply_sklearn_pickle_compat(model):
    """Patch known missing attrs on legacy sklearn pickles loaded in newer runtimes."""
    class_name = model.__class__.__name__

    if class_name == 'LogisticRegression':
        # Older serialized artifacts may miss attrs required by newer sklearn internals.
        if not hasattr(model, 'multi_class'):
            model.multi_class = 'auto'
        if not hasattr(model, 'n_jobs'):
            model.n_jobs = None
        if not hasattr(model, 'l1_ratio'):
            model.l1_ratio = None

    return model

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


class FastResNet18(nn.Module):
    """ResNet18 backbone used for the 90%+ model."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ModelLoader:
    """Centralized model loading and prediction"""
    
    def __init__(self, models_dir='../models'):
        base_dir = Path(__file__).resolve().parent
        candidate_dir = Path(models_dir)
        # Resolve relative model path from this file location, not process CWD.
        self.models_dir = (base_dir / candidate_dir).resolve() if not candidate_dir.is_absolute() else candidate_dir
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_aliases = {
            'random_forest': 'random_forest_model',
            'random_forest_model': 'random_forest_model',
            'logistic_regression': 'logistic_regression_model',
            'logistic_regression_model': 'logistic_regression_model',
            'decision_tree': 'decision_tree_model',
            'decision_tree_model': 'decision_tree_model',
            'knn': 'k-nearest_neighbors_model',
            'k-nearest_neighbors': 'k-nearest_neighbors_model',
            'k-nearest_neighbors_model': 'k-nearest_neighbors_model',
            'naive_bayes': 'naive_bayes_model',
            'naive_bayes_model': 'naive_bayes_model',
            'cnn': 'cnn_model',
            'cnn_model': 'cnn_model',
            'fast_resnet': 'fast_resnet_model',
            'fast_resnet_model': 'fast_resnet_model',
            'best_model': 'fast_resnet_model'
        }
        self.fast_resnet_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _resolve_model_name(self, model_name):
        """Normalize model names so API callers can use stable aliases."""
        if not model_name:
            return model_name
        return self.model_aliases.get(model_name.strip().lower(), model_name)
        
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
                loaded_model = joblib.load(model_path)
                self.models[model_name] = _apply_sklearn_pickle_compat(loaded_model)
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

    def load_fast_resnet_model(self):
        """Load the best-performing ResNet checkpoint used for the 90%+ model."""
        model_path = self.models_dir / 'fast_resnet18_xray.pth'

        if model_path.exists():
            fast_model = FastResNet18(num_classes=2).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            fast_model.load_state_dict(state_dict)
            fast_model.eval()

            self.models['fast_resnet_model'] = fast_model
            print(f"✓ Loaded: fast_resnet_model on {self.device}")
        else:
            print("⚠ fast_resnet18_xray checkpoint not found")
    
    def predict_classical(self, features, model_name):
        """Make prediction using classical ML model"""
        canonical_name = self._resolve_model_name(model_name)

        if canonical_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[canonical_name]
        
        # Scale features
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Predict (retry once after compatibility patch for legacy sklearn artifacts)
        try:
            prediction = model.predict(features_scaled)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 1.0
        except AttributeError:
            model = _apply_sklearn_pickle_compat(model)
            self.models[canonical_name] = model
            prediction = model.predict(features_scaled)[0]
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
            'model': canonical_name
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

    def predict_fast_resnet(self, image):
        """Make prediction using the ResNet checkpoint trained to 90%+ accuracy."""
        if 'fast_resnet_model' not in self.models:
            raise ValueError("Fast ResNet model not loaded")

        model = self.models['fast_resnet_model']
        image_tensor = self.fast_resnet_transform(image.convert('RGB')).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            if self.label_encoder:
                label = self.label_encoder.inverse_transform([predicted.item()])[0]
            else:
                label = str(predicted.item())

            return {
                'prediction': label,
                'confidence': float(confidence.item()),
                'model': 'fast_resnet_model'
            }
    
    def list_available_models(self):
        """Return list of loaded models"""
        return list(self.models.keys())
