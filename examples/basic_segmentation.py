#!/usr/bin/env python3
"""
Basic YOLO segmentation example.

This example demonstrates how to use imageProcessingUtils for nuclei segmentation
on a simple synthetic image.
"""

import numpy as np
import matplotlib.pyplot as plt
from imageProcessingUtils.yolo.segmentation import YOLOSegmentation

def create_synthetic_image():
    """Create a synthetic microscopy-like image with circular objects."""
    # Create a 512x512 image with some circular objects
    image = np.zeros((512, 512), dtype=np.uint8)
    
    # Add some circular "nuclei"
    centers = [(100, 100), (200, 150), (350, 200), (150, 300), (400, 400)]
    
    for center in centers:
        y, x = np.ogrid[:512, :512]
        mask = (x - center[0])**2 + (y - center[1])**2 < 40**2
        image[mask] = np.random.randint(180, 255)
    
    # Add some noise
    noise = np.random.randint(0, 50, (512, 512)).astype(np.uint8)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image

def main():
    """Run the basic segmentation example."""
    print("Basic YOLO Segmentation Example")
    print("=" * 40)
    
    # Create synthetic test image
    print("Creating synthetic microscopy image...")
    image = create_synthetic_image()
    
    # Initialize YOLO segmentation
    print("Initializing YOLO segmentation...")
    segmenter = YOLOSegmentation()
    
    if not segmenter.is_available():
        print("ERROR: YOLO model not available!")
        print("Make sure the package is properly installed with model files.")
        return
    
    print("YOLO model loaded successfully")
    
    # Perform segmentation
    print("Running segmentation...")
    labels, mask = segmenter.segment(image, conf_thres=0.1)
    
    # Display results
    num_objects = labels.max()
    coverage = mask.sum() / mask.size * 100
    
    print(f"Results:")
    print(f"   • Objects detected: {num_objects}")
    print(f"   • Image coverage: {coverage:.1f}%")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='Reds', alpha=0.7)
    axes[1].imshow(image, cmap='gray', alpha=0.3)
    axes[1].set_title(f'Segmentation Mask\n({num_objects} objects)')
    axes[1].axis('off')
    
    axes[2].imshow(labels, cmap='tab10')
    axes[2].set_title('Instance Labels')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_example.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'segmentation_example.png'")
    plt.show()

if __name__ == "__main__":
    main()
