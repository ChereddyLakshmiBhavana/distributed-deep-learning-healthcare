"""
Flask REST API for Pneumonia Detection
Backend API with model integration - Required for Hackathon Syllabus
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_loader import ModelLoader
import traceback

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

# Image preprocessing for CNN
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

def extract_features_from_image(image):
    """
    Extract statistical features from image for classical ML models
    """
    img_array = np.array(image.convert('L').resize((64, 64)))
    
    features = {
        'mean_intensity': np.mean(img_array),
        'std_intensity': np.std(img_array),
        'min_intensity': np.min(img_array),
        'max_intensity': np.max(img_array),
        'median_intensity': np.median(img_array),
    }
    
    # Histogram features
    hist, _ = np.histogram(img_array, bins=10, range=(0, 256))
    hist = hist / hist.sum() * 100
    for i, val in enumerate(hist):
        features[f'hist_bin_{i}'] = val
    
    # Additional features
    features['edge_density'] = np.sum(np.abs(np.diff(img_array))) / img_array.size
    features['contrast'] = img_array.max() - img_array.min()
    
    return np.array([list(features.values())])


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
    print("  GET  /test       - Test endpoint for Postman")
    print("="*60)
    print("\nServer will start on http://localhost:5000")
    print("\n💡 Quick Postman Test:")
    print("   1. POST to /predict/upload")
    print("   2. Body: form-data")
    print("   3. Key: 'file' (select X-ray image)")
    print("   4. Key: 'model' = 'random_forest_model'")
    print("\nUse Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
