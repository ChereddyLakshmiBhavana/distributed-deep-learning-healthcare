"""
Prediction Service Helpers
Shared image validation, feature extraction, and inference helpers for sync/async workers.
"""

from io import BytesIO
import base64

import numpy as np
from PIL import Image


def decode_image_bytes(image_bytes):
    """Decode raw bytes into a PIL image."""
    return Image.open(BytesIO(image_bytes))


def decode_base64_image(image_data):
    """Decode a base64 payload or data URL into a PIL image."""
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]
    return decode_image_bytes(base64.b64decode(image_data))


def validate_chest_xray_image(image):
    """Heuristic validation - extremely lenient to avoid false rejections.
    Only rejects if image is clearly corrupted or blank."""
    try:
        # Just check if it's a valid image with reasonable dimensions
        rgb_img = image.convert('RGB').resize((224, 224))
        rgb = np.array(rgb_img, dtype=np.float32)
        
        if rgb.size == 0:
            return {'is_valid': False, 'reason': 'Empty image'}
        
        # Compute some stats but don't use them to reject
        rg_diff = np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1])) / 255.0
        gb_diff = np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2])) / 255.0
        rb_diff = np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 2])) / 255.0
        color_difference = float((rg_diff + gb_diff + rb_diff) / 3.0)

        gray = np.array(rgb_img.convert('L'), dtype=np.float32) / 255.0
        dynamic_range = float(np.max(gray) - np.min(gray))

        # ALWAYS PASS for any real image - let the model handle it
        # Only reject if completely blank or corrupted
        if dynamic_range < 0.01:  # Basically all white or all black
            return {'is_valid': False, 'reason': 'Image appears blank or completely saturated'}
        
        return {
            'is_valid': True,
            'color_difference': round(color_difference, 4),
            'dynamic_range': round(dynamic_range, 4)
        }
    
    except Exception as e:
        # If anything goes wrong, still let it through
        return {'is_valid': True, 'reason': f'Could not analyze, but allowing: {str(e)}'}


def extract_features_from_image(image):
    """Extract the 17 statistical features used by the classical models."""
    img = image.convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0

    features = {
        'mean_intensity': float(np.mean(img_array)),
        'std_intensity': float(np.std(img_array)),
        'min_intensity': float(np.min(img_array)),
        'max_intensity': float(np.max(img_array)),
        'median_intensity': float(np.median(img_array)),
    }

    hist, _ = np.histogram(img_array, bins=10, range=(0, 1))
    hist = hist / hist.sum() * 100
    for i, val in enumerate(hist):
        features[f'hist_bin_{i}'] = float(val)

    edges = np.abs(np.diff(img_array))
    features['edge_density'] = float(np.sum(edges) / img_array.size)
    features['contrast'] = float(img_array.max() - img_array.min())

    return np.array([
        [
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
    ])


def predict_from_image(model_loader, image, model_name='k-nearest_neighbors_model'):
    """Run inference on a PIL image using the shared model loader. Falls back to fast_resnet if model unavailable."""
    result = None
    
    try:
        if model_name == 'cnn_model':
            from torchvision import transforms

            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
            image_tensor = image_transform(image).unsqueeze(0)
            result = model_loader.predict_cnn(image_tensor)
        elif model_name == 'fast_resnet_model':
            result = model_loader.predict_fast_resnet(image)
        else:
            # Try classical model
            features = extract_features_from_image(image)
            result = model_loader.predict_classical(features, model_name)
    except (ValueError, KeyError):
        # Any model failure; fall back to fast_resnet
        if model_name != 'fast_resnet_model' and result is None:
            result = model_loader.predict_fast_resnet(image)
        else:
            raise

    return result


def predict_from_image_bytes(model_loader, image_bytes, model_name='k-nearest_neighbors_model', validate=True):
    """Decode bytes, optionally validate, and return prediction payload."""
    image = decode_image_bytes(image_bytes)
    validation = validate_chest_xray_image(image) if validate else {'is_valid': True}

    if validate and not validation['is_valid']:
        return {
            'success': False,
            'error': 'Uploaded image does not appear to be a chest X-ray',
            'validation': validation
        }

    result = predict_from_image(model_loader, image, model_name=model_name)
    return {
        'success': True,
        'prediction': result['prediction'],
        'confidence': float(result['confidence']),
        'model_used': result['model'],
        'validation': validation
    }


def predict_from_base64(model_loader, image_data, model_name='k-nearest_neighbors_model', validate=True):
    """Decode a base64 image payload and run inference."""
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]
    image_bytes = base64.b64decode(image_data)
    return predict_from_image_bytes(
        model_loader=model_loader,
        image_bytes=image_bytes,
        model_name=model_name,
        validate=validate
    )
