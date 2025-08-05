"""
Create a sample uint8 test image for YOLO segmentation testing.
"""
import numpy as np
import cv2
from pathlib import Path

def create_test_image():
    """Create a sample uint8 image with circular objects."""
    # Create image with some noise
    image = np.random.randint(20, 50, (512, 512), dtype=np.uint8)
    
    # Add circular objects of different sizes
    centers_and_radii = [
        (128, 128, 30),   # Top-left
        (384, 128, 25),   # Top-right
        (128, 384, 35),   # Bottom-left
        (384, 384, 28),   # Bottom-right
        (256, 256, 40),   # Center
        (200, 150, 20),   # Additional small object
        (350, 300, 22),   # Additional small object
    ]
    
    for cx, cy, radius in centers_and_radii:
        y, x = np.ogrid[:512, :512]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        # Make objects brighter with some variation
        intensity = np.random.randint(180, 220)
        image[mask] = intensity
    
    return image

if __name__ == "__main__":
    # Create test image
    test_image = create_test_image()
    
    # Save as PNG
    output_path = Path(__file__).parent / "sample_nd2_images" / "test_sample.png"
    cv2.imwrite(str(output_path), test_image)
    
    print(f"Created test image: {output_path}")
    print(f"Image shape: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Intensity range: {test_image.min()} - {test_image.max()}")
