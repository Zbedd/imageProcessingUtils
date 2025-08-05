"""
Diagnostic script for external repos to debug YOLO model loading issues.
Copy this to your other repository and run it there.
"""
import sys
from pathlib import Path

print("=== External Repository YOLO Diagnostic ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {Path.cwd()}")

# Check if imageProcessingUtils is installed
try:
    import imageProcessingUtils
    print(f"✅ imageProcessingUtils imported from: {imageProcessingUtils.__file__}")
    
    # Check package installation location
    package_path = Path(imageProcessingUtils.__file__).parent
    print(f"Package location: {package_path}")
    
    # Check if it's editable install vs regular install
    egg_info_dirs = list(package_path.glob("*.egg-info"))
    if egg_info_dirs:
        print(f"Package appears to be editable install: {egg_info_dirs}")
    else:
        print("Package appears to be regular pip install")
    
    # Check for yolo module
    yolo_path = package_path / "yolo"
    print(f"YOLO module path: {yolo_path}")
    print(f"YOLO module exists: {yolo_path.exists()}")
    
    if yolo_path.exists():
        # Check models directory
        models_dir = yolo_path / "models"
        print(f"Models directory: {models_dir}")
        print(f"Models directory exists: {models_dir.exists()}")
        
        if models_dir.exists():
            pt_files = list(models_dir.glob("*.pt"))
            print(f"Found {len(pt_files)} .pt files:")
            for pt_file in pt_files:
                try:
                    size_mb = pt_file.stat().st_size / (1024*1024)
                    print(f"  - {pt_file.name} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"  - {pt_file.name} (error reading size: {e})")
            
            # Check specifically for fine-tuned models
            fine_tuned = list(models_dir.glob("best_*.pt"))
            print(f"Fine-tuned models (best_*.pt): {len(fine_tuned)}")
        
        # Check runs directory
        runs_dir = yolo_path / "runs" / "segment"
        print(f"Runs directory: {runs_dir}")
        print(f"Runs directory exists: {runs_dir.exists()}")
        
        if runs_dir.exists():
            best_pts = list(runs_dir.glob("train*/weights/best.pt"))
            print(f"Found {len(best_pts)} training best.pt files:")
            for best_pt in best_pts:
                try:
                    size_mb = best_pt.stat().st_size / (1024*1024)
                    print(f"  - {best_pt.relative_to(runs_dir)} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"  - {best_pt.relative_to(runs_dir)} (error: {e})")

except ImportError as e:
    print(f"❌ Failed to import imageProcessingUtils: {e}")
    print("   Make sure the package is installed: pip install -e /path/to/imageProcessingUtils")
    sys.exit(1)

# Test model discovery and loading
print("\n=== Testing Model Discovery ===")
try:
    from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
    
    print("Creating YOLOSegmentation instance (1st time)...")
    yolo_seg = YOLOSegmentation()
    
    print(f"Model path discovered: {yolo_seg.model_path}")
    print(f"Model available: {yolo_seg.is_available()}")
    
    if yolo_seg.model_path and yolo_seg.model_path.exists():
        size_mb = yolo_seg.model_path.stat().st_size / (1024*1024)
        print(f"Model file size: {size_mb:.1f} MB")
    
    # Test creating a second instance (this might be where it fails)
    print("\nCreating YOLOSegmentation instance (2nd time)...")
    yolo_seg2 = YOLOSegmentation()
    
    print(f"Second instance model path: {yolo_seg2.model_path}")
    print(f"Second instance available: {yolo_seg2.is_available()}")
    
    # Test simple segmentation
    if yolo_seg.is_available():
        print("\nTesting segmentation...")
        import numpy as np
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        try:
            labels, mask = yolo_seg.segment(test_image)
            print(f"✅ First segmentation successful! Found {labels.max()} objects")
        except Exception as e:
            print(f"❌ First segmentation failed: {e}")
        
        try:
            labels2, mask2 = yolo_seg2.segment(test_image)
            print(f"✅ Second segmentation successful! Found {labels2.max()} objects")
        except Exception as e:
            print(f"❌ Second segmentation failed: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"❌ Error during model testing: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Environment Check ===")
print("Environment variables that might affect YOLO:")
env_vars = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'YOLO_VERBOSE']
for var in env_vars:
    import os
    value = os.environ.get(var, 'Not set')
    print(f"  {var}: {value}")

print("\n=== Recommendations ===")
print("1. If no models found: Reinstall package with: pip install -e /path/to/imageProcessingUtils")
print("2. If first works but second fails: Possible model state issue")
print("3. If paths exist but loading fails: Check file permissions")
print("4. If CPU-only errors: Set CUDA_VISIBLE_DEVICES='' before running")
