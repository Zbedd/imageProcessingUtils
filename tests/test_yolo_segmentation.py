"""
Test YOLO segmentation on synthetic data and uint8 images with visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo


def load_sample_images(sample_dir: Path):
    """Load all uint8 images from sample directory."""
    images = []
    
    # Support common image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
    
    for ext in image_extensions:
        for img_file in sample_dir.glob(ext):
            try:
                # Load image with OpenCV
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    continue
                
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                
                images.append((img_file.stem, image))
                print(f"Loaded {img_file.name}: {image.shape}")
                
            except Exception as e:
                print(f"Failed to load {img_file.name}: {e}")
    
    return images


def plot_segmentation_results(image_name: str, original: np.ndarray, 
                            instance_labels: np.ndarray, binary_mask: np.ndarray):
    """Create visualization plot for segmentation results."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'YOLO Segmentation: {image_name}')
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(binary_mask, cmap='Blues')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    # Instance labels
    if instance_labels.max() > 0:
        axes[1, 0].imshow(instance_labels, cmap='tab20')
        axes[1, 0].set_title(f'Instances ({instance_labels.max()} objects)')
    else:
        axes[1, 0].imshow(np.zeros_like(instance_labels), cmap='gray')
        axes[1, 0].set_title('No objects detected')
    axes[1, 0].axis('off')
    
    # Overlay
    axes[1, 1].imshow(original, cmap='gray')
    axes[1, 1].imshow(binary_mask, cmap='Reds', alpha=0.3)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def test_synthetic_image():
    """Test with a simple synthetic image."""
    print("Testing synthetic image...")
    
    # Create test image with circular objects
    image = np.zeros((256, 256), dtype=np.uint8)
    centers = [(64, 64), (192, 64), (128, 192)]
    
    for i, (cx, cy) in enumerate(centers):
        y, x = np.ogrid[:256, :256]
        mask = (x - cx)**2 + (y - cy)**2 <= 20**2
        image[mask] = 150 + i * 30
    
    # Run segmentation
    instance_labels, binary_mask = segmentation_pipeline_yolo(image, conf_thres=0.1)
    
    print(f"Synthetic test: {instance_labels.max()} objects detected")
    
    # Just display results, don't save
    return instance_labels, binary_mask


def test_sample_images():
    """Test segmentation on sample uint8 images."""
    sample_dir = Path(__file__).parent / "sample_uint8_images"
    
    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        return
    
    print(f"Loading samples from: {sample_dir}")
    images = load_sample_images(sample_dir)
    
    if not images:
        print("No images loaded")
        return
    
    print(f"Processing {len(images)} images...")
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_yolo" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    for image_name, image in images:
        try:
            # Run segmentation
            instance_labels, binary_mask = segmentation_pipeline_yolo(image, conf_thres=0.3)
            
            num_objects = instance_labels.max()
            coverage = (binary_mask.sum() / binary_mask.size) * 100
            
            print(f"{image_name}: {num_objects} objects, {coverage:.1f}% coverage")
            
            # Create and save plot
            fig = plot_segmentation_results(image_name, image, instance_labels, binary_mask)
            fig.savefig(output_dir / f"{image_name}_segmentation.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            successful += 1
            
        except Exception as e:
            print(f"Failed to process {image_name}: {e}")
    
    print(f"Completed: {successful}/{len(images)} successful")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    print("YOLO Segmentation Test")
    print("-" * 30)
    
    # Test synthetic image first
    try:
        test_synthetic_image()
    except Exception as e:
        print(f"Synthetic test failed: {e}")
    
    # Test sample images
    try:
        test_sample_images()
    except Exception as e:
        print(f"Sample test failed: {e}")
    
    print("Testing complete")
