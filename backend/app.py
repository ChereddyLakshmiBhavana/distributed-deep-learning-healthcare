"""
Flask REST API for Pneumonia Detection
Backend API with model integration - Required for Hackathon Syllabus
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_loader import ModelLoader
import traceback
import os
from datetime import datetime
from report_generator import MedicalReportGenerator
from explainability import PneumoniaExplainer

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize model loader
print("Loading models...")
model_loader = ModelLoader()

try:
    model_loader.load_classical_models()
    model_loader.load_cnn_model()
    print("\n✓ All models loaded successfully!\n")
except Exception as e:
    print(f"⚠ Error loading models: {e}")

# Initialize report generator and explainer
report_generator = MedicalReportGenerator(output_dir='reports')
explainer = PneumoniaExplainer(output_dir='explanations')
print("✓ Report generator and explainability modules initialized\n")

# Image preprocessing for CNN
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def validate_chest_xray_image(image):
    """
    Lightweight input validation to reject obvious non-X-ray images.

    This is a heuristic guardrail (not a diagnostic classifier).
    """
    # Work on a fixed-size thumbnail for stable stats
    rgb_img = image.convert('RGB').resize((224, 224))
    rgb = np.array(rgb_img, dtype=np.float32)

    # X-rays are grayscale-like; color photos have larger channel differences.
    rg_diff = np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1])) / 255.0
    gb_diff = np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2])) / 255.0
    rb_diff = np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 2])) / 255.0
    color_difference = float((rg_diff + gb_diff + rb_diff) / 3.0)

    # Additional sanity checks on grayscale intensity distribution.
    gray = np.array(rgb_img.convert('L'), dtype=np.float32) / 255.0
    gray_std = float(np.std(gray))
    dynamic_range = float(np.max(gray) - np.min(gray))

    # Thresholds tuned to block obvious natural/color images while keeping X-rays.
    is_valid = (
        color_difference <= 0.03 and
        gray_std >= 0.05 and
        dynamic_range >= 0.20
    )

    return {
        'is_valid': is_valid,
        'color_difference': round(color_difference, 4),
        'gray_std': round(gray_std, 4),
        'dynamic_range': round(dynamic_range, 4)
    }

def extract_features_from_image(image):
    """
    Extract statistical features from image for classical ML models
    MUST MATCH exactly with notebook preprocessing for correct predictions
    """
    # Convert to grayscale and resize to 64x64 (MUST match notebook)
    img = image.convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize pixel values to 0-1 range (important!)
    img_array = img_array / 255.0
    
    features = {
        'mean_intensity': float(np.mean(img_array)),
        'std_intensity': float(np.std(img_array)),
        'min_intensity': float(np.min(img_array)),
        'max_intensity': float(np.max(img_array)),
        'median_intensity': float(np.median(img_array)),
    }
    
    # Histogram features (10 bins, matching notebook)
    hist, _ = np.histogram(img_array, bins=10, range=(0, 1))
    hist = hist / hist.sum() * 100  # Normalize to percentage
    for i, val in enumerate(hist):
        features[f'hist_bin_{i}'] = float(val)
    
    # Edge detection for spatial features
    edges = np.abs(np.diff(img_array))
    features['edge_density'] = float(np.sum(edges) / img_array.size)
    
    # Contrast
    features['contrast'] = float(img_array.max() - img_array.min())
    
    # Return as numpy array in correct format
    feature_values = [
        features['mean_intensity'],
        features['std_intensity'],
        features['min_intensity'],
        features['max_intensity'],
        features['median_intensity'],
        features['hist_bin_0'],
        features['hist_bin_1'],
        features['hist_bin_2'],
        features['hist_bin_3'],
        features['hist_bin_4'],
        features['hist_bin_5'],
        features['hist_bin_6'],
        features['hist_bin_7'],
        features['hist_bin_8'],
        features['hist_bin_9'],
        features['edge_density'],
        features['contrast']
    ]
    
    return np.array([feature_values])


@app.route('/')
def home():
    """API Home endpoint"""
    return jsonify({
        'message': 'Pneumonia Detection API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/models': 'List available models',
            '/predict': 'Make prediction with base64 JSON (POST)',
            '/predict/upload': 'Make prediction with file upload (POST)',
            '/predict/batch': 'Batch predictions (POST)',
            '/test': 'Test endpoint helper (GET)'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_loader.list_available_models()
    })


@app.route('/models', methods=['GET'])
def get_models():
    """List all available models"""
    models = model_loader.list_available_models()
    return jsonify({
        'available_models': models,
        'total': len(models)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts JSON with base64 encoded image and model selection
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        # Get image data
        if 'image' not in data:
            return jsonify({
                'error': 'No image data provided. Expected "image" field with base64 encoded image.'
            }), 400
        
        image_data = data['image']
        model_name = data.get('model', 'random_forest_model')  # Default model
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            return jsonify({
                'error': f'Invalid image data: {str(e)}'
            }), 400

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Make prediction based on model type
        if model_name == 'cnn_model':
            # CNN prediction
            image_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
            result = model_loader.predict_cnn(image_tensor)
        else:
            # Classical ML prediction
            features = extract_features_from_image(image)
            result = model_loader.predict_classical(features, model_name)
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'model_used': result['model'],
            'message': f'Prediction: {result["prediction"]} (Confidence: {result["confidence"]:.2%})'
        })
    
    except ValueError as e:
        return jsonify({
            'error': str(e)
        }), 400
    
    except Exception as e:
        print(f"Error in prediction: {traceback.format_exc()}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Accepts multiple images
    """
    try:
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({
                'error': 'No images provided. Expected "images" array.'
            }), 400
        
        images_data = data['images']
        model_name = data.get('model', 'random_forest_model')
        
        results = []
        for idx, image_data in enumerate(images_data):
            try:
                # Decode image
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))

                # Validate X-ray suitability per item
                validation = validate_chest_xray_image(image)
                if not validation['is_valid']:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': 'Image does not appear to be a chest X-ray',
                        'validation': validation
                    })
                    continue
                
                # Predict
                if model_name == 'cnn_model':
                    image_tensor = image_transform(image).unsqueeze(0)
                    result = model_loader.predict_cnn(image_tensor)
                else:
                    features = extract_features_from_image(image)
                    result = model_loader.predict_classical(features, model_name)
                
                results.append({
                    'index': idx,
                    'success': True,
                    'prediction': result['prediction'],
                    'confidence': round(result['confidence'], 4)
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(images_data),
            'results': results,
            'model_used': model_name
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


@app.route('/test', methods=['GET'])
def test_endpoint():
    """
    Test endpoint for Postman testing
    Returns sample request format
    """
    return jsonify({
        'message': 'Test endpoint for API validation',
        'sample_request': {
            'method': 'POST',
            'url': '/predict',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': {
                'image': 'base64_encoded_image_string',
                'model': 'random_forest_model'  # or any other available model
            }
        },
        'available_models': model_loader.list_available_models(),
        'example_models': [
            'logistic_regression_model',
            'decision_tree_model',
            'random_forest_model',
            'k-nearest_neighbors_model',
            'naive_bayes_model',
            'cnn_model'
        ]
    })


@app.route('/predict/upload', methods=['POST'])
def predict_file_upload():
    """
    Alternative prediction endpoint with direct file upload
    Simpler for Postman testing - no base64 encoding needed
    
    Usage in Postman:
    - Method: POST
    - Body: form-data
    - Key: 'file' (type: File)
    - Key: 'model' (type: Text, value: model name)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided. Use key "file" in form-data'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get model selection (default to random forest)
        model_name = request.form.get('model', 'random_forest_model')
        
        # Read and process image
        image = Image.open(file.stream)

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Make prediction based on model type
        if model_name == 'cnn_model':
            # CNN prediction
            image_tensor = image_transform(image).unsqueeze(0)
            result = model_loader.predict_cnn(image_tensor)
        else:
            # Classical ML prediction
            features = extract_features_from_image(image)
            result = model_loader.predict_classical(features, model_name)
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'model_used': result['model'],
            'message': f'Prediction: {result["prediction"]} (Confidence: {result["confidence"]:.2%})'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict/report', methods=['POST'])
def predict_with_report():
    """
    Generate prediction with professional PDF medical report
    
    Usage:
    - Upload image file
    - Get prediction + PDF report download link
    - Optionally get predictions from all models for comparison
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided. Use key "file" in form-data'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get model selection (default to random forest - best performer)
        model_name = request.form.get('model', 'random_forest_model')
        include_all_models = request.form.get('include_all_models', 'false').lower() == 'true'
        
        # Read and process image
        image = Image.open(file.stream)

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Save temporary image for report
        temp_image_path = os.path.join('reports', f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        os.makedirs('reports', exist_ok=True)
        image.save(temp_image_path)
        
        # Extract features for classical ML
        features = extract_features_from_image(image)
        
        # Make prediction with selected model
        result = model_loader.predict_classical(features, model_name)
        
        # Get predictions from all models if requested
        all_models_results = None
        if include_all_models:
            all_models_results = []
            for model_key in model_loader.list_available_models():
                if model_key != 'cnn_model':  # Skip CNN for now
                    try:
                        model_result = model_loader.predict_classical(features, model_key)
                        all_models_results.append({
                            'model': model_key.replace('_', ' ').title(),
                            'prediction': model_result['prediction'],
                            'confidence': model_result['confidence']
                        })
                    except Exception as e:
                        print(f"Error with model {model_key}: {e}")
        
        # Generate PDF report
        prediction_data = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'model_name': model_name.replace('_', ' ').title(),
            'image_path': temp_image_path,
            'all_models_results': all_models_results,
            'exam_id': f"XR-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        }
        
        report_path = report_generator.generate_report(prediction_data)
        
        # Get just the filename for download URL
        report_filename = os.path.basename(report_path)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'model_used': result['model'],
            'report_pdf': report_filename,
            'download_url': f'/download/reports/{report_filename}',
            'all_models': all_models_results,
            'message': f'Report generated successfully. Prediction: {result["prediction"]}'
        })
    
    except Exception as e:
        print(f"Error in predict_with_report: {traceback.format_exc()}")
        return jsonify({
            'error': f'Report generation failed: {str(e)}'
        }), 500


@app.route('/predict/explain', methods=['POST'])
def predict_with_explanation():
    """
    Generate prediction with SHAP explainability visualizations
    
    Usage:
    - Upload image file
    - Get prediction + SHAP explanation showing which features contributed
    - Returns visualization images and human-readable explanation
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided. Use key "file" in form-data'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get model selection (default to random forest)
        model_name = request.form.get('model', 'random_forest_model')
        
        # Read and process image
        image = Image.open(file.stream)

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Extract features
        features = extract_features_from_image(image)
        
        # Get the model and scaler
        model = model_loader.models.get(model_name)
        scaler = model_loader.scaler
        
        if model is None:
            return jsonify({
                'error': f'Model {model_name} not found or not supported for explanation'
            }), 400
        
        # Generate SHAP explanation
        explanation_data = explainer.explain_prediction(
            model=model,
            features=features,
            scaler=scaler,
            model_name=model_name
        )
        
        # Convert file paths to download URLs
        visualizations = {}
        for viz_type, viz_path in explanation_data['visualizations'].items():
            if viz_path:
                viz_filename = os.path.basename(viz_path)
                visualizations[viz_type] = {
                    'filename': viz_filename,
                    'download_url': f'/download/explanations/{viz_filename}'
                }
        
        # Get top 5 features for summary
        top_features = explanation_data['feature_importance'][:5]
        top_features_summary = [
            {
                'feature': f['feature'],
                'description': f['description'],
                'importance': round(f['abs_importance'], 4),
                'direction': f['direction']
            }
            for f in top_features
        ]
        
        return jsonify({
            'success': True,
            'prediction': 'PNEUMONIA' if explanation_data['prediction'] == 1 else 'NORMAL',
            'confidence': round(explanation_data['confidence'], 4),
            'model_used': model_name,
            'visualizations': visualizations,
            'top_features': top_features_summary,
            'explanation_text': explanation_data['explanation_text'],
            'message': 'Explanation generated successfully'
        })
    
    except Exception as e:
        print(f"Error in predict_with_explanation: {traceback.format_exc()}")
        return jsonify({
            'error': f'Explanation generation failed: {str(e)}'
        }), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    """
    Download generated files (reports, explanations)
    
    Usage:
    - GET /download/reports/report_name.pdf
    - GET /download/explanations/explanation_name.png
    """
    try:
        # Determine directory based on path
        if filename.startswith('reports/'):
            directory = 'reports'
            filename = filename.replace('reports/', '')
        elif filename.startswith('explanations/'):
            directory = 'explanations'
            filename = filename.replace('explanations/', '')
        else:
            # Default to reports directory
            directory = 'reports'
        
        return send_from_directory(directory, filename, as_attachment=True)
    
    except Exception as e:
        return jsonify({
            'error': f'File not found: {str(e)}'
        }), 404


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on this server.'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred.'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Pneumonia Detection API Server")
    print("="*60)
    print("API Documentation:")
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")
    print("  GET  /models     - List available models")
    print("  POST /predict    - Make prediction (JSON with base64 image)")
    print("  POST /predict/upload - Make prediction (File upload - easier!)")
    print("  POST /predict/batch - Make batch predictions")
    print("  POST /predict/report - Generate prediction + PDF report")
    print("  POST /predict/explain - Generate prediction + SHAP explanation")
    print("  GET  /download/<path> - Download reports/explanations")
    print("  GET  /test       - Test endpoint for Postman")
    print("="*60)
    print("\n🆕 NEW FEATURES:")
    print("  📄 PDF Medical Reports - Clinical-grade documentation")
    print("  🔍 SHAP Explainability - Visual feature importance")
    print("="*60)
    print("\nServer will start on http://localhost:5000")
    print("\n💡 Quick Postman Test:")
    print("   1. POST to /predict/upload")
    print("   2. Body: form-data")
    print("   3. Key: 'file' (select X-ray image)")
    print("   4. Key: 'model' = 'random_forest_model'")
    print("\n💡 Generate PDF Report:")
    print("   1. POST to /predict/report")
    print("   2. Body: form-data")
    print("   3. Key: 'file' (select X-ray image)")
    print("   4. Download PDF from returned URL")
    print("\n💡 Get AI Explanation:")
    print("   1. POST to /predict/explain")
    print("   2. Body: form-data")
    print("   3. Key: 'file' (select X-ray image)")
    print("   4. View SHAP visualizations")
    print("\nUse Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
