"""
Generate Synthetic Test Images for Demo
Creates sample chest X-ray-like images for testing when real dataset is not available
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

def create_sample_xray(image_type='normal', size=(224, 224), output_path=None):
    """
    Create a synthetic X-ray-like image
    
    Args:
        image_type: 'normal' or 'pneumonia'
        size: Image dimensions (width, height)
        output_path: Where to save the image
    
    Returns:
        PIL Image object
    """
    # Create base grayscale image
    if image_type == 'normal':
        # Normal lungs - clearer, more uniform
        base_intensity = 120
        noise_factor = 20
    else:  # pneumonia
        # Pneumonia - cloudier, more variation
        base_intensity = 90
        noise_factor = 40
    
    # Generate base image with noise
    img_array = np.random.normal(base_intensity, noise_factor, size)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array, mode='L')
    
    # Add some structure to make it look more like X-ray
    draw = ImageDraw.Draw(img)
    
    # Simulate rib cage structure
    for i in range(5):
        y_pos = 50 + i * 30
        draw.ellipse([30, y_pos-5, size[0]-30, y_pos+5], fill=base_intensity+30)
    
    # Add lung areas (darker regions)
    left_lung = [40, 40, size[0]//2-10, size[1]-40]
    right_lung = [size[0]//2+10, 40, size[0]-40, size[1]-40]
    
    if image_type == 'pneumonia':
        # Add cloudy patches for pneumonia
        for _ in range(10):
            x = np.random.randint(left_lung[0], left_lung[2])
            y = np.random.randint(left_lung[1], left_lung[3])
            radius = np.random.randint(10, 30)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=base_intensity-40)
    
    # Apply some blur to make it look more realistic
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"✓ Created: {output_path}")
    
    return img


def generate_test_dataset(output_dir='../data/test_samples', num_normal=5, num_pneumonia=5):
    """
    Generate a complete test dataset with multiple samples
    
    Args:
        output_dir: Directory to save test images
        num_normal: Number of normal X-ray samples
        num_pneumonia: Number of pneumonia X-ray samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic test images...")
    print("="*60)
    
    # Generate normal samples
    for i in range(num_normal):
        filename = f"normal_{i+1:03d}.jpg"
        path = os.path.join(output_dir, filename)
        create_sample_xray('normal', output_path=path)
    
    # Generate pneumonia samples
    for i in range(num_pneumonia):
        filename = f"pneumonia_{i+1:03d}.jpg"
        path = os.path.join(output_dir, filename)
        create_sample_xray('pneumonia', output_path=path)
    
    print("="*60)
    print(f"✓ Generated {num_normal} normal samples")
    print(f"✓ Generated {num_pneumonia} pneumonia samples")
    print(f"✓ Saved to: {output_dir}")
    print("\nThese images can be used for API testing and demo!")


if __name__ == '__main__':
    # Generate test dataset
    generate_test_dataset(
        output_dir='../data/test_samples',
        num_normal=5,
        num_pneumonia=5
    )
    
    print("\n" + "="*60)
    print("USAGE:")
    print("="*60)
    print("1. These images are SYNTHETIC for demo purposes only")
    print("2. Use them to test your API and frontend")
    print("3. For Postman: Upload any .jpg file from data/test_samples/")
    print("4. For real results: Download actual dataset from Kaggle")
    print("")
    print("To use real dataset:")
    print("  Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("  Extract to: data/chest_xray/")
