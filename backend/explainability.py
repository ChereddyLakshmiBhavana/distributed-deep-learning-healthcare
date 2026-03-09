"""
SHAP Explainability Module
Provides interpretable explanations for pneumonia predictions
"""

import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import os


class PneumoniaExplainer:
    """Generate SHAP explanations for pneumonia detection models"""
    
    def __init__(self, output_dir='explanations'):
        """
        Initialize explainer
        
        Args:
            output_dir: Directory to save explanation visualizations
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.feature_names = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'median_intensity', 'hist_bin_0', 'hist_bin_1', 'hist_bin_2', 
            'hist_bin_3', 'hist_bin_4', 'hist_bin_5', 'hist_bin_6', 
            'hist_bin_7', 'hist_bin_8', 'hist_bin_9', 'edge_density', 'contrast'
        ]
        
        self.feature_descriptions = {
            'mean_intensity': 'Average brightness of X-ray',
            'std_intensity': 'Variation in brightness (contrast)',
            'min_intensity': 'Darkest pixel value',
            'max_intensity': 'Brightest pixel value',
            'median_intensity': 'Middle brightness value',
            'hist_bin_0': 'Very dark region distribution',
            'hist_bin_1': 'Dark region distribution',
            'hist_bin_2': 'Dark-medium region distribution',
            'hist_bin_3': 'Medium-dark region distribution',
            'hist_bin_4': 'Medium region distribution',
            'hist_bin_5': 'Medium region distribution',
            'hist_bin_6': 'Medium-bright region distribution',
            'hist_bin_7': 'Bright-medium region distribution',
            'hist_bin_8': 'Bright region distribution',
            'hist_bin_9': 'Very bright region distribution',
            'edge_density': 'Sharpness of boundaries/edges',
            'contrast': 'Overall contrast range'
        }
    
    def explain_prediction(self, model, features, scaler, model_name='random_forest'):
        """
        Generate SHAP explanation for a prediction
        
        Args:
            model: Trained sklearn model
            features: Feature array (1D or 2D)
            scaler: StandardScaler used during training
            model_name: Name of the model for labeling
            
        Returns:
            dict: Explanation data including visualizations and feature importance
        """
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Get prediction for context
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Create SHAP explainer
        try:
            # For tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled_features)
            
            # For binary classification, get PNEUMONIA class (index 1)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_pneumonia = shap_values[1][0]  # Shape: (n_features,)
                base_value = explainer.expected_value[1]
            else:
                # Handle single output case
                if len(shap_values.shape) == 2:
                    shap_values_pneumonia = shap_values[0]  # Shape: (n_features,)
                else:
                    shap_values_pneumonia = shap_values
                base_value = explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value[0]
                
        except Exception as e:
            print(f"TreeExplainer failed, using KernelExplainer: {e}")
            # Fallback to KernelExplainer for non-tree models
            # Use a small background dataset (just the current sample)
            explainer = shap.KernelExplainer(model.predict_proba, scaled_features)
            shap_values = explainer.shap_values(scaled_features)
            
            if isinstance(shap_values, list):
                shap_values_pneumonia = shap_values[1][0]
            else:
                shap_values_pneumonia = shap_values[0]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        
        # CRITICAL: Ensure shap_values_pneumonia is always 1D array with correct length
        shap_values_pneumonia = np.array(shap_values_pneumonia).flatten()
        
        # Validate we have exactly 17 features
        if len(shap_values_pneumonia) != 17:
            raise ValueError(f"Expected 17 SHAP values, got {len(shap_values_pneumonia)}")
        
        # Generate visualizations
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Waterfall plot (shows contribution of each feature)
        waterfall_path = self._create_waterfall_plot(
            shap_values_pneumonia, 
            base_value,
            scaled_features[0],
            model_name,
            timestamp
        )
        
        # 2. Bar plot (feature importance)
        bar_path = self._create_bar_plot(
            shap_values_pneumonia,
            model_name,
            timestamp
        )
        
        # 3. Force plot (alternative visualization)
        force_path = self._create_force_plot(
            shap_values_pneumonia,
            base_value,
            scaled_features[0],
            model_name,
            timestamp
        )
        
        # Get feature importance ranking
        feature_importance = self._get_feature_importance(shap_values_pneumonia)
        
        # Generate human-readable explanation
        explanation_text = self._generate_explanation_text(
            feature_importance,
            prediction,
            prediction_proba
        )
        
        return {
            'prediction': prediction,
            'confidence': float(prediction_proba[1]),  # PNEUMONIA class probability
            'visualizations': {
                'waterfall': waterfall_path,
                'bar': bar_path,
                'force': force_path
            },
            'feature_importance': feature_importance,
            'explanation_text': explanation_text,
            'shap_values': shap_values_pneumonia.tolist(),
            'base_value': float(base_value)
        }
    
    def _create_waterfall_plot(self, shap_values, base_value, features, model_name, timestamp):
        """Create waterfall plot showing feature contributions"""
        
        plt.figure(figsize=(10, 8))
        
        # Ensure shap_values is 1D array
        if len(shap_values.shape) > 1:
            shap_values = shap_values.flatten()
        
        # Ensure features is 1D array
        if len(features.shape) > 1:
            features = features.flatten()
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=features,
            feature_names=self.feature_names
        )
        
        # Create waterfall plot
        shap.plots.waterfall(explanation, show=False)
        
        plt.title(f'SHAP Waterfall Plot - {model_name}\nHow each feature contributes to prediction', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'waterfall_{model_name}_{timestamp}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_bar_plot(self, shap_values, model_name, timestamp):
        """Create bar plot of feature importance"""
        
        # Ensure shap_values is 1D
        shap_values = np.array(shap_values).flatten()
        
        plt.figure(figsize=(10, 7))
        
        # Get absolute SHAP values for importance
        importance = np.abs(shap_values)
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        top_k = min(10, len(indices))  # Show top 10 features
        
        top_indices = indices[:top_k]
        top_importance = importance[top_indices]
        top_names = [self.feature_names[i] for i in top_indices]
        
        # Create bar plot
        colors_list = ['#dc3545' if shap_values[i] > 0 else '#28a745' for i in top_indices]
        
        plt.barh(range(top_k), top_importance, color=colors_list)
        plt.yticks(range(top_k), top_names)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_k} Most Important Features - {model_name}\n' + 
                  'Red = pushes toward PNEUMONIA, Green = pushes toward NORMAL',
                  fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_importance):
            plt.text(v, i, f'  {v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'bar_{model_name}_{timestamp}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_force_plot(self, shap_values, base_value, features, model_name, timestamp):
        """Create force plot showing feature contributions"""
        
        try:
            fig, ax = plt.subplots(figsize=(12, 3))
            
            # Create explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=features,
                feature_names=self.feature_names
            )
            
            # Force plot (rendered as matplotlib)
            shap.plots.force(explanation, matplotlib=True, show=False)
            
            plt.title(f'SHAP Force Plot - {model_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'force_{model_name}_{timestamp}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return filepath
        except Exception as e:
            print(f"Could not create force plot: {e}")
            return None
    
    def _get_feature_importance(self, shap_values):
        """Extract and rank feature importance"""
        
        # Ensure shap_values is 1D
        shap_values = np.array(shap_values).flatten()
        
        importance_list = []
        
        for i, feature_name in enumerate(self.feature_names):
            importance_list.append({
                'feature': feature_name,
                'description': self.feature_descriptions.get(feature_name, feature_name),
                'shap_value': float(shap_values[i]),
                'abs_importance': float(abs(shap_values[i])),
                'direction': 'PNEUMONIA' if shap_values[i] > 0 else 'NORMAL'
            })
        
        # Sort by absolute importance
        importance_list.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        return importance_list
    
    def _generate_explanation_text(self, feature_importance, prediction, prediction_proba):
        """Generate human-readable explanation"""
        
        prediction_label = 'PNEUMONIA' if prediction == 1 else 'NORMAL'
        confidence = prediction_proba[1] * 100  # PNEUMONIA probability
        
        explanation = f"PREDICTION EXPLANATION\n\n"
        explanation += f"The model predicts: {prediction_label}\n"
        explanation += f"Confidence: {confidence:.1f}% for PNEUMONIA\n\n"
        
        explanation += "KEY FACTORS INFLUENCING THIS PREDICTION:\n\n"
        
        # Top 5 most important features
        for i, feat in enumerate(feature_importance[:5], 1):
            direction_text = "increases" if feat['direction'] == 'PNEUMONIA' else "decreases"
            
            explanation += f"{i}. {feat['feature']} (Impact: {feat['abs_importance']:.4f})\n"
            explanation += f"   → {feat['description']}\n"
            explanation += f"   → This feature {direction_text} likelihood of {feat['direction']}\n"
            
            # Add interpretation
            if 'intensity' in feat['feature'].lower():
                if feat['direction'] == 'PNEUMONIA':
                    explanation += f"   → Abnormal brightness pattern detected\n"
                else:
                    explanation += f"   → Normal brightness pattern\n"
            elif 'hist_bin' in feat['feature']:
                if feat['direction'] == 'PNEUMONIA':
                    explanation += f"   → Unusual density distribution (characteristic of fluid/infection)\n"
                else:
                    explanation += f"   → Normal tissue density distribution\n"
            elif 'edge' in feat['feature'].lower():
                if feat['direction'] == 'PNEUMONIA':
                    explanation += f"   → Sharp boundaries detected (possible infiltrates)\n"
                else:
                    explanation += f"   → Smooth boundaries (normal)\n"
            elif 'contrast' in feat['feature'].lower():
                if feat['direction'] == 'PNEUMONIA':
                    explanation += f"   → High contrast suggests dense regions\n"
                else:
                    explanation += f"   → Normal contrast levels\n"
            
            explanation += "\n"
        
        explanation += "\nINTERPRETATION:\n"
        
        if prediction_label == 'PNEUMONIA':
            if confidence >= 90:
                explanation += "Strong indicators of pneumonia detected. Multiple features show patterns "
                explanation += "consistent with pulmonary infiltrates or infection.\n"
            elif confidence >= 75:
                explanation += "Notable indicators of pneumonia present. Features suggest abnormal lung patterns.\n"
            else:
                explanation += "Some indicators suggest pneumonia, but confidence is moderate. "
                explanation += "Manual review recommended.\n"
        else:
            if confidence < 50:  # NORMAL prediction
                normal_conf = (1 - prediction_proba[1]) * 100
                explanation += f"Features predominantly indicate normal lung tissue (NORMAL: {normal_conf:.1f}%).\n"
                explanation += "No significant abnormalities detected.\n"
            else:
                explanation += "Some uncertainty in classification. Features show mixed patterns.\n"
        
        return explanation


# Convenience function for easy use
def explain_model_prediction(model, features, scaler, model_name='random_forest', output_dir='explanations'):
    """
    Generate SHAP explanation for a prediction
    
    Args:
        model: Trained sklearn model
        features: Feature array
        scaler: StandardScaler
        model_name: Name of model
        output_dir: Output directory for visualizations
        
    Returns:
        dict: Explanation data
    """
    
    explainer = PneumoniaExplainer(output_dir=output_dir)
    return explainer.explain_prediction(model, features, scaler, model_name)
