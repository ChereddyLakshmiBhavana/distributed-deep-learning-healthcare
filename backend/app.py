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
from pathlib import Path
from report_generator import MedicalReportGenerator
from explainability import PneumoniaExplainer
from prediction_logger import PredictionLogger
from prediction_database import PredictionDatabase
from distributed_engine import DistributedEngine, serialize_state_dict, deserialize_state_dict
from prediction_service import predict_from_base64, predict_from_image_bytes

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPORTS_DIR = PROJECT_ROOT / 'reports'
EXPLANATIONS_DIR = PROJECT_ROOT / 'explanations'
PREDICTION_LOGS_DIR = PROJECT_ROOT / 'prediction_logs'
LEGACY_BACKEND_REPORTS_DIR = BASE_DIR / 'reports'
LEGACY_BACKEND_EXPLANATIONS_DIR = BASE_DIR / 'explanations'

# Initialize model loader
print("Loading models...")
model_loader = ModelLoader()

try:
    model_loader.load_classical_models()
    model_loader.load_cnn_model()
    model_loader.load_fast_resnet_model()
    print("\n✓ All models loaded successfully!\n")
except Exception as e:
    print(f"⚠ Error loading models: {e}")

# Initialize report generator and explainer
report_generator = MedicalReportGenerator(output_dir=str(REPORTS_DIR))
explainer = PneumoniaExplainer(output_dir=str(EXPLANATIONS_DIR))
prediction_logger = PredictionLogger(output_dir=str(PREDICTION_LOGS_DIR))
prediction_db = PredictionDatabase(db_path=str(PREDICTION_LOGS_DIR / 'predictions.db'))
distributed_engine = DistributedEngine(state_dir='distributed_state', models_dir='../models')
print("✓ Report generator and explainability modules initialized\n")


def safe_log_prediction(endpoint, status, model=None, prediction=None, confidence=None, error=None):
    """Write prediction log entries without breaking API flow on logger errors."""
    try:
        prediction_logger.log(
            endpoint=endpoint,
            status=status,
            model=model,
            prediction=prediction,
            confidence=confidence,
            error=error
        )
    except Exception as logger_error:
        print(f"⚠ Logging failed for {endpoint}: {logger_error}")


def safe_store_prediction_record(
    endpoint,
    status,
    image_bytes,
    filename=None,
    content_type=None,
    model=None,
    prediction=None,
    confidence=None,
    error=None
):
    """Persist image + diagnosis in DB without affecting API flow."""
    try:
        prediction_db.save_prediction(
            endpoint=endpoint,
            status=status,
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
            model=model,
            prediction=prediction,
            confidence=confidence,
            error=error
        )
    except Exception as db_error:
        print(f"⚠ DB storage failed for {endpoint}: {db_error}")

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
    Thresholds are intentionally lenient to avoid false rejections of valid X-rays.
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

    # Lenient thresholds to allow valid X-rays through while blocking obvious color images.
    # Only reject if it has VERY high color variance (color photos) AND low contrast (not an X-ray)
    is_valid = (
        color_difference <= 0.10 and  # Loosened from 0.03: allow slight color shifts
        dynamic_range >= 0.10  # Loosened from 0.20: allow lower contrast X-rays
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
            '/test': 'Test endpoint helper (GET)',
            '/logs/info': 'Prediction log file metadata (GET)',
            '/database/info': 'Prediction image/result database metadata (GET)',
            '/distributed/info': 'Distributed runtime summary (GET)',
            '/distributed/jobs': 'Queue/list distributed jobs (GET/POST)',
            '/distributed/jobs/<id>': 'Distributed job details (GET)',
            '/distributed/jobs/process-next': 'Process next queued job (POST)',
            '/predict/async': 'Queue async prediction (POST)',
            '/distributed/federated/register': 'Register federated node (POST)',
            '/distributed/federated/update': 'Submit federated model update (POST)',
            '/distributed/federated/aggregate': 'Aggregate federated updates (POST)',
            '/distributed/federated/status': 'List federated nodes/updates (GET)',
            '/distributed/training/info': 'Distributed training readiness (GET)'
        },
        'prediction_logs': {
            'enabled': True,
            'file_path': prediction_logger.get_log_path()
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
            safe_log_prediction('/predict', 'failed', error='No JSON data provided')
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        # Get image data
        if 'image' not in data:
            safe_log_prediction('/predict', 'failed', model=model_name if 'model_name' in locals() else None,
                               error='No image data provided')
            return jsonify({
                'error': 'No image data provided. Expected "image" field with base64 encoded image.'
            }), 400
        
        image_data = data['image']
        requested_model = data.get('model', 'fast_resnet_model')  # Default model
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            safe_log_prediction('/predict', 'failed', model=requested_model, error=f'Invalid image data: {str(e)}')
            return jsonify({
                'error': f'Invalid image data: {str(e)}'
            }), 400

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            safe_log_prediction('/predict', 'failed', model=requested_model,
                               error='Uploaded image does not appear to be a chest X-ray')
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Make prediction with automatic fallback for unavailable models
        result = None
        used_fallback = False
        
        # Try requested model first
        if requested_model in ['cnn_model', 'fast_resnet_model']:
            try:
                if requested_model == 'cnn_model':
                    image_tensor = image_transform(image).unsqueeze(0)
                    result = model_loader.predict_cnn(image_tensor)
                else:
                    result = model_loader.predict_fast_resnet(image)
            except (ValueError, KeyError):
                # Fall back to fast_resnet if it fails
                result = model_loader.predict_fast_resnet(image)
                used_fallback = True
        else:
            # For classical models, try with fallback
            try:
                features = extract_features_from_image(image)
                result = model_loader.predict_classical(features, requested_model)
            except (ValueError, KeyError):
                # Classical model not loaded; use fast_resnet instead
                result = model_loader.predict_fast_resnet(image)
                used_fallback = True

        safe_log_prediction('/predict', 'success', model=result['model'],
                           prediction=result['prediction'], confidence=result['confidence'])
        safe_store_prediction_record(
            endpoint='/predict',
            status='success',
            image_bytes=image_bytes,
            filename='base64_upload',
            content_type='image/base64',
            model=result['model'],
            prediction=result['prediction'],
            confidence=result['confidence']
        )
        
        # Return prediction with fallback notice if applicable
        response = {
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'model_used': result['model'],
            'message': f'Prediction: {result["prediction"]} (Confidence: {result["confidence"]:.2%})'
        }
        if used_fallback:
            response['note'] = f'Requested model "{requested_model}" not available; using {result["model"]} instead.'
        
        return jsonify(response)
    
    except ValueError as e:
        safe_log_prediction('/predict', 'failed', model=requested_model,
                           error=str(e))
        return jsonify({
            'error': str(e)
        }), 400
    
    except Exception as e:
        safe_log_prediction('/predict', 'failed', model=requested_model,
                           error=str(e))
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
            safe_log_prediction('/predict/batch', 'failed', error='No images provided')
            return jsonify({
                'error': 'No images provided. Expected "images" array.'
            }), 400
        
        images_data = data['images']
        model_name = data.get('model', 'fast_resnet_model')
        
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
                safe_log_prediction('/predict/batch', 'success', model=model_name,
                                   prediction=result['prediction'], confidence=result['confidence'])
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
                safe_log_prediction('/predict/batch', 'failed', model=model_name, error=str(e))
        
        return jsonify({
            'success': True,
            'total': len(images_data),
            'results': results,
            'model_used': model_name
        })
    
    except Exception as e:
        safe_log_prediction('/predict/batch', 'failed', model=request.get_json(silent=True).get('model') if request.get_json(silent=True) else None,
                           error=str(e))
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
                'model': 'fast_resnet_model'  # or any other available model
            }
        },
        'available_models': model_loader.list_available_models(),
        'example_models': [
            'logistic_regression_model',
            'decision_tree_model',
            'random_forest_model',
            'k-nearest_neighbors_model',
            'naive_bayes_model',
            'fast_resnet_model',
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
            safe_log_prediction('/predict/upload', 'failed', error='No file provided')
            return jsonify({
                'error': 'No file provided. Use key "file" in form-data'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            safe_log_prediction('/predict/upload', 'failed', error='Empty filename')
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get model selection (default to the 90%+ ResNet checkpoint)
        requested_model = request.form.get('model', 'fast_resnet_model')
        
        # Read and process image
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes))

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            safe_log_prediction('/predict/upload', 'failed', model=requested_model,
                               error='Uploaded image does not appear to be a chest X-ray')
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Make prediction with automatic fallback for unavailable models
        result = None
        used_fallback = False
        
        # Try requested model first
        if requested_model in ['cnn_model', 'fast_resnet_model']:
            try:
                if requested_model == 'cnn_model':
                    image_tensor = image_transform(image).unsqueeze(0)
                    result = model_loader.predict_cnn(image_tensor)
                else:
                    result = model_loader.predict_fast_resnet(image)
            except (ValueError, KeyError):
                # Fall back to fast_resnet if it fails
                result = model_loader.predict_fast_resnet(image)
                used_fallback = True
        else:
            # For classical models, try with fallback
            try:
                features = extract_features_from_image(image)
                result = model_loader.predict_classical(features, requested_model)
            except (ValueError, KeyError):
                # Classical model not loaded; use fast_resnet instead
                result = model_loader.predict_fast_resnet(image)
                used_fallback = True

        safe_log_prediction('/predict/upload', 'success', model=result['model'],
                           prediction=result['prediction'], confidence=result['confidence'])
        safe_store_prediction_record(
            endpoint='/predict/upload',
            status='success',
            image_bytes=file_bytes,
            filename=file.filename,
            content_type=file.mimetype,
            model=result['model'],
            prediction=result['prediction'],
            confidence=result['confidence']
        )
        
        # Return prediction with fallback notice if applicable
        response = {
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'model_used': result['model'],
            'message': f'Prediction: {result["prediction"]} (Confidence: {result["confidence"]:.2%})'
        }
        if used_fallback:
            response['note'] = f'Requested model "{requested_model}" not available; using {result["model"]} instead.'
        
        return jsonify(response)
    
    except Exception as e:
        safe_log_prediction('/predict/upload', 'failed', model=request.form.get('model'), error=str(e))
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
            safe_log_prediction('/predict/report', 'failed', error='No file provided')
            return jsonify({
                'error': 'No file provided. Use key "file" in form-data'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            safe_log_prediction('/predict/report', 'failed', error='Empty filename')
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get model selection (default to the 90%+ ResNet checkpoint)
        model_name = request.form.get('model', 'fast_resnet_model')
        include_all_models = request.form.get('include_all_models', 'false').lower() == 'true'
        
        # Read and process image
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes))

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            safe_log_prediction('/predict/report', 'failed', model=model_name,
                               error='Uploaded image does not appear to be a chest X-ray')
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
        if model_name == 'fast_resnet_model':
            result = model_loader.predict_fast_resnet(image)
        else:
            result = model_loader.predict_classical(features, model_name)
        
        # Get predictions from all models if requested
        all_models_results = None
        if include_all_models:
            all_models_results = []
            for model_key in model_loader.list_available_models():
                if model_key not in {'cnn_model', 'fast_resnet_model'}:
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

        safe_log_prediction('/predict/report', 'success', model=result['model'],
                   prediction=result['prediction'], confidence=result['confidence'])
        safe_store_prediction_record(
            endpoint='/predict/report',
            status='success',
            image_bytes=file_bytes,
            filename=file.filename,
            content_type=file.mimetype,
            model=result['model'],
            prediction=result['prediction'],
            confidence=result['confidence']
        )
        
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
        safe_log_prediction('/predict/report', 'failed', model=request.form.get('model'), error=str(e))
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
            safe_log_prediction('/predict/explain', 'failed', error='No file provided')
            return jsonify({
                'error': 'No file provided. Use key "file" in form-data'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            safe_log_prediction('/predict/explain', 'failed', error='Empty filename')
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get model selection (default to the 90%+ ResNet checkpoint)
        model_name = request.form.get('model', 'fast_resnet_model')
        explanation_model_name = model_name
        
        # Read and process image
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes))

        # Validate that input resembles a chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation['is_valid']:
            safe_log_prediction('/predict/explain', 'failed', model=model_name,
                               error='Uploaded image does not appear to be a chest X-ray')
            return jsonify({
                'error': 'Uploaded image does not appear to be a chest X-ray. Please upload a valid chest X-ray image.',
                'validation': validation
            }), 400
        
        # Extract features
        features = extract_features_from_image(image)

        # The fast ResNet model is supported for prediction but not for SHAP explanation.
        # Fall back to a compatible classical model for explainability.
        if explanation_model_name in ('cnn_model', 'fast_resnet_model'):
            explanation_model_name = 'random_forest_model'

        if model_name == 'fast_resnet_model':
            prediction_result = model_loader.predict_fast_resnet(image)
        elif model_name == 'cnn_model':
            image_tensor = image_transform(image).unsqueeze(0)
            prediction_result = model_loader.predict_cnn(image_tensor)
        else:
            prediction_result = model_loader.predict_classical(features, model_name)

        if explanation_model_name not in model_loader.models:
            explanation_model_name = 'random_forest_model'
        
        # Get the model and scaler
        model = model_loader.models.get(explanation_model_name)
        scaler = model_loader.scaler
        
        if model is None:
            safe_log_prediction('/predict/explain', 'failed', model=explanation_model_name,
                               error=f'Model {explanation_model_name} not found or not supported for explanation')
            return jsonify({
                'error': f'Model {model_name} is not supported for explanation right now. Try a classical model such as Random Forest.'
            }), 400
        
        # Generate SHAP explanation
        explanation_data = explainer.explain_prediction(
            model=model,
            features=features,
            scaler=scaler,
            model_name=explanation_model_name
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

        safe_log_prediction('/predict/explain', 'success', model=prediction_result['model'],
                           prediction=prediction_result['prediction'],
                           confidence=prediction_result['confidence'])
        safe_store_prediction_record(
            endpoint='/predict/explain',
            status='success',
            image_bytes=file_bytes,
            filename=file.filename,
            content_type=file.mimetype,
            model=prediction_result['model'],
            prediction=prediction_result['prediction'],
            confidence=prediction_result['confidence']
        )
        
        return jsonify({
            'success': True,
            'prediction': prediction_result['prediction'],
            'confidence': round(prediction_result['confidence'], 4),
            'model_used': prediction_result['model'],
            'explanation_model_used': explanation_model_name,
            'visualizations': visualizations,
            'top_features': top_features_summary,
            'explanation_text': explanation_data['explanation_text'],
            'note': 'ResNet uses a classical-model fallback for explanation.' if model_name == 'fast_resnet_model' else ('CNN uses a classical-model fallback for explanation.' if model_name == 'cnn_model' else None),
            'message': 'Explanation generated successfully'
        })
    
    except Exception as e:
        safe_log_prediction('/predict/explain', 'failed', model=request.form.get('model'), error=str(e))
        print(f"Error in predict_with_explanation: {traceback.format_exc()}")
        return jsonify({
            'error': f'Explanation generation failed: {str(e)}'
        }), 500


@app.route('/logs/info', methods=['GET'])
def logs_info():
    """Return prediction history logging metadata."""
    return jsonify({
        'logging_enabled': True,
        'log_file': prediction_logger.get_log_path(),
        'columns': ['timestamp', 'endpoint', 'status', 'model', 'prediction', 'confidence', 'error']
    })


@app.route('/distributed/info', methods=['GET'])
def distributed_info():
    """Return a summary of the distributed engineering stack."""
    return jsonify({
        'summary': distributed_engine.get_summary(),
        'paths': {
            'state_db': str(distributed_engine.db_path),
            'models_dir': str(distributed_engine.models_dir),
        },
        'capabilities': {
            'job_queue': True,
            'federated_learning': True,
            'distributed_training_ready': True,
            'async_prediction': True,
            'audit_logging': True
        }
    })


@app.route('/distributed/jobs', methods=['GET', 'POST'])
def distributed_jobs():
    """Create jobs or list queued jobs for async processing."""
    if request.method == 'GET':
        status = request.args.get('status')
        limit = int(request.args.get('limit', 50))
        return jsonify({
            'jobs': distributed_engine.list_jobs(status=status, limit=limit)
        })

    data = request.get_json(silent=True) or {}
    task_type = data.get('task_type')
    payload = data.get('payload', {})
    priority = int(data.get('priority', 0))

    if not task_type:
        return jsonify({'error': 'task_type is required'}), 400

    job_id = distributed_engine.enqueue_job(task_type=task_type, payload=payload, priority=priority)
    return jsonify({
        'success': True,
        'job_id': job_id,
        'task_type': task_type
    }), 201


@app.route('/distributed/jobs/<int:job_id>', methods=['GET'])
def distributed_job_detail(job_id):
    """Fetch a single job record."""
    job = distributed_engine.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/distributed/jobs/process-next', methods=['POST'])
def distributed_job_process_next():
    """Process the next queued job inside the API process."""
    job = distributed_engine.claim_next_job()
    if not job:
        return jsonify({'message': 'No pending jobs available'}), 200

    try:
        task_type = job['task_type']
        payload = job['payload']

        if task_type == 'predict_base64':
            result = predict_from_base64(
                model_loader=model_loader,
                image_data=payload['image'],
                model_name=payload.get('model', 'fast_resnet_model'),
                validate=payload.get('validate', True)
            )
        elif task_type == 'predict_bytes':
            image_bytes = bytes.fromhex(payload['image_hex'])
            result = predict_from_image_bytes(
                model_loader=model_loader,
                image_bytes=image_bytes,
                model_name=payload.get('model', 'fast_resnet_model'),
                validate=payload.get('validate', True)
            )
        elif task_type == 'federated_aggregate':
            result = distributed_engine.aggregate_federated_updates(
                model_name=payload['model_name'],
                round_id=payload.get('round_id')
            )
        else:
            raise ValueError(f'Unsupported task type: {task_type}')

        distributed_engine.update_job(job['id'], 'completed', result=result)
        return jsonify({
            'success': True,
            'job_id': job['id'],
            'result': result
        })

    except Exception as exc:
        distributed_engine.update_job(job['id'], 'failed', error=str(exc))
        return jsonify({'error': str(exc), 'job_id': job['id']}), 500


@app.route('/predict/async', methods=['POST'])
def predict_async():
    """Queue a prediction request for asynchronous processing."""
    data = request.get_json(silent=True) or {}
    image_data = data.get('image')
    model_name = data.get('model', 'fast_resnet_model')
    validate = data.get('validate', True)

    if not image_data:
        return jsonify({'error': 'image is required'}), 400

    job_id = distributed_engine.enqueue_job(
        task_type='predict_base64',
        payload={
            'image': image_data,
            'model': model_name,
            'validate': bool(validate)
        },
        priority=int(data.get('priority', 0))
    )

    return jsonify({
        'success': True,
        'queued': True,
        'job_id': job_id,
        'status_url': f'/distributed/jobs/{job_id}'
    }), 202


@app.route('/distributed/federated/register', methods=['POST'])
def distributed_register_node():
    """Register a hospital/site node for federated learning."""
    data = request.get_json(silent=True) or {}
    node_name = data.get('node_name')
    if not node_name:
        return jsonify({'error': 'node_name is required'}), 400

    node_id = distributed_engine.register_node(
        node_name=node_name,
        site_name=data.get('site_name'),
        node_type=data.get('node_type', 'hospital'),
        metadata=data.get('metadata', {})
    )
    return jsonify({'success': True, 'node_id': node_id}), 201


@app.route('/distributed/federated/update', methods=['POST'])
def distributed_submit_update():
    """Submit a federated update as a pickled state_dict blob."""
    data = request.get_json(silent=True) or {}
    node_id = data.get('node_id')
    model_name = data.get('model_name')
    state_dict_b64 = data.get('state_dict')
    num_samples = int(data.get('num_samples', 1))

    if not node_id or not model_name or not state_dict_b64:
        return jsonify({'error': 'node_id, model_name, and state_dict are required'}), 400

    state_dict = deserialize_state_dict(state_dict_b64)
    update_id, round_id = distributed_engine.submit_federated_update(
        node_id=int(node_id),
        model_name=model_name,
        state_dict=state_dict,
        num_samples=num_samples,
        metrics=data.get('metrics', {}),
        round_id=data.get('round_id')
    )

    return jsonify({
        'success': True,
        'update_id': update_id,
        'round_id': round_id
    }), 201


@app.route('/distributed/federated/aggregate', methods=['POST'])
def distributed_aggregate_updates():
    """Aggregate all submitted federated updates for a model."""
    data = request.get_json(silent=True) or {}
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({'error': 'model_name is required'}), 400

    result = distributed_engine.aggregate_federated_updates(
        model_name=model_name,
        round_id=data.get('round_id')
    )
    return jsonify({'success': True, 'result': result})


@app.route('/distributed/federated/status', methods=['GET'])
def distributed_federated_status():
    """Return federated node and update status."""
    model_name = request.args.get('model_name')
    round_id = request.args.get('round_id')
    return jsonify({
        'nodes': distributed_engine.list_nodes(),
        'updates': distributed_engine.list_federated_updates(model_name=model_name, round_id=round_id),
    })


@app.route('/distributed/training/info', methods=['GET'])
def distributed_training_info():
    """Return distributed training readiness information."""
    return jsonify({
        'torch_cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count(),
        'ddp_recommended': torch.cuda.device_count() > 1,
        'multi_gpu_mode': 'DDP' if torch.cuda.device_count() > 1 else 'single_device',
        'distributed_launch_command': 'torchrun --standalone --nproc_per_node=<gpus> backend/distributed_ddp_train.py',
        'distributed_trainer': 'backend/distributed_ddp_train.py'
    })


@app.route('/database/info', methods=['GET'])
def database_info():
    """Return database metadata for stored image+prediction records."""
    stats = prediction_db.get_stats()
    return jsonify({
        'storage_enabled': True,
        **stats,
        'stored_fields': [
            'id', 'created_at', 'endpoint', 'status', 'filename',
            'content_type', 'image_bytes', 'model', 'prediction',
            'confidence', 'error'
        ]
    })


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
            directory = REPORTS_DIR
            filename = filename.replace('reports/', '')
        elif filename.startswith('explanations/'):
            directory = EXPLANATIONS_DIR
            filename = filename.replace('explanations/', '')
        else:
            # Default to reports directory
            directory = REPORTS_DIR

        file_path = Path(directory) / filename
        if not file_path.exists() and directory == REPORTS_DIR:
            legacy_path = LEGACY_BACKEND_REPORTS_DIR / filename
            if legacy_path.exists():
                file_path = legacy_path
        elif not file_path.exists() and directory == EXPLANATIONS_DIR:
            legacy_path = LEGACY_BACKEND_EXPLANATIONS_DIR / filename
            if legacy_path.exists():
                file_path = legacy_path

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path.name} not found in {directory}")

        return send_file(str(file_path), as_attachment=True)
    
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
    print("   4. Key: 'model' = 'fast_resnet_model'")
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
    
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
