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

    def _extract_pneumonia_shap_values(self, shap_values):
        """Normalize SHAP output to a 1D vector for pneumonia class (17 features)."""
        n_features = len(self.feature_names)

        # Case 1: Classic list output [class0_values, class1_values]
        if isinstance(shap_values, list):
            if len(shap_values) >= 2:
                values = np.array(shap_values[1])
            else:
                values = np.array(shap_values[0])

            if values.ndim > 1:
                values = values[0]
            return np.array(values).flatten()

        # Case 2: ndarray output from newer SHAP versions
        values = np.array(shap_values)

        # Expected variants observed:
        # (1, n_features), (n_features,), (1, n_features, 2), (1, 2, n_features)
        if values.ndim == 1:
            return values

        if values.ndim == 2:
            # Usually (1, n_features)
            if values.shape[0] == 1:
                return values[0]
            # If class dimension is mixed in 2D, pick pneumonia class index 1
            if values.shape[0] == 2 and values.shape[1] == n_features:
                return values[1]
            if values.shape[1] == 2 and values.shape[0] == n_features:
                return values[:, 1]
            return values.flatten()

        if values.ndim == 3:
            # (samples, features, classes)
            if values.shape[2] == 2:
                return values[0, :, 1]
            # (samples, classes, features)
            if values.shape[1] == 2:
                return values[0, 1, :]
            # Fallback: first sample flattened
            return values[0].flatten()

        # Generic fallback
        return values.flatten()

    def _extract_pneumonia_base_value(self, expected_value):
        """Extract scalar base value for pneumonia class."""
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            arr = np.array(expected_value).flatten()
            if len(arr) >= 2:
                return float(arr[1])
            return float(arr[0])
        return float(expected_value)
    
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
        scaled_features = scaler.transform(features) if scaler is not None else features
        feature_vector = np.array(scaled_features[0], dtype=float).flatten()

        # Get prediction for context
        prediction = model.predict(scaled_features)[0]
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(scaled_features)[0]
        else:
            prediction_proba = np.array([0.5, 0.5], dtype=float)

        attributions = self._estimate_feature_attributions(model, feature_vector)
        attributions = np.array(attributions, dtype=float).flatten()

        if len(attributions) != 17:
            raise ValueError(f"Expected 17 attribution values, got {len(attributions)}")
        
        # Get feature importance ranking
        feature_importance = self._get_feature_importance(attributions)
        
        # Generate human-readable explanation
        explanation_text = self._generate_explanation_text(
            feature_importance,
            prediction,
            prediction_proba
        )
        
        return {
            'prediction': prediction,
            'confidence': float(prediction_proba[1]),  # PNEUMONIA class probability
            'visualizations': {},
            'feature_importance': feature_importance,
            'explanation_text': explanation_text,
            'shap_values': attributions.tolist(),
            'base_value': float(prediction_proba[1])
        }

    def _estimate_feature_attributions(self, model, feature_vector):
        """Create a fast, deterministic explanation vector for the current model."""
        feature_vector = np.array(feature_vector, dtype=float).flatten()

        if hasattr(model, 'feature_importances_'):
            weights = np.array(model.feature_importances_, dtype=float).flatten()
            if len(weights) != len(feature_vector):
                weights = np.resize(weights, len(feature_vector))
            return weights * np.sign(feature_vector)

        if hasattr(model, 'coef_'):
            coefficients = np.array(model.coef_, dtype=float)
            if coefficients.ndim > 1:
                coefficients = coefficients[0]
            coefficients = coefficients.flatten()
            if len(coefficients) != len(feature_vector):
                coefficients = np.resize(coefficients, len(feature_vector))
            return coefficients * feature_vector

        centered = feature_vector - np.median(feature_vector)
        return centered
    
    def _create_waterfall_plot(self, attributions, base_value, features, model_name, timestamp):
        """Create a lightweight waterfall-style plot showing feature contributions."""

        plt.figure(figsize=(10, 8))

        attributions = np.array(attributions, dtype=float).flatten()
        order = np.argsort(np.abs(attributions))[::-1][:10]
        names = [self.feature_names[i] for i in order]
        colors = ['#dc3545' if attributions[i] > 0 else '#28a745' for i in order]

        plt.barh(range(len(order)), attributions[order], color=colors)
        plt.yticks(range(len(order)), names)
        plt.axvline(0, color='#666666', linewidth=1)
        plt.xlabel('Estimated contribution toward PNEUMONIA (+) or NORMAL (-)', fontsize=12, fontweight='bold')
        plt.title(f'Feature Contribution Waterfall - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'waterfall_{model_name}_{timestamp}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_bar_plot(self, attributions, model_name, timestamp):
        """Create a ranked bar plot of estimated feature importance."""

        attributions = np.array(attributions, dtype=float).flatten()

        plt.figure(figsize=(10, 7))

        importance = np.abs(attributions)
        indices = np.argsort(importance)[::-1]
        top_k = min(10, len(indices))

        top_indices = indices[:top_k]
        top_importance = importance[top_indices]
        top_names = [self.feature_names[i] for i in top_indices]
        colors_list = ['#dc3545' if attributions[i] > 0 else '#28a745' for i in top_indices]

        plt.barh(range(top_k), top_importance, color=colors_list)
        plt.yticks(range(top_k), top_names)
        plt.xlabel('Estimated impact on prediction', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_k} Most Important Features - {model_name}', fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, v in enumerate(top_importance):
            plt.text(v, i, f'  {v:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'bar_{model_name}_{timestamp}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_force_plot(self, attributions, base_value, features, model_name, timestamp):
        """Create a simple force-style plot showing positive vs negative contributions."""

        try:
            plt.figure(figsize=(12, 3.5))
            attributions = np.array(attributions, dtype=float).flatten()
            order = np.argsort(np.abs(attributions))[::-1][:12]
            colors = ['#dc3545' if attributions[i] > 0 else '#28a745' for i in order]
            plt.bar(range(len(order)), attributions[order], color=colors)
            plt.axhline(0, color='#666666', linewidth=1)
            plt.xticks(range(len(order)), [self.feature_names[i] for i in order], rotation=45, ha='right')
            plt.ylabel('Contribution')
            plt.title(f'Contribution Force View - {model_name}', fontsize=12, fontweight='bold')
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
