#!/usr/bin/env python3
"""
Debug script for external repositories.
Copy this file to your TUNEL repository and run it to see what's happening.
"""

import sys
from pathlib import Path
import traceback

print("=== External Repository Debug Info ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {Path.cwd()}")
print(f"Python path: {sys.path}")

# Check PyTorch and device availability
print()
print("=== PyTorch and Device Info ===")
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU device")
except ImportError:
    print("❌ PyTorch not available")
except Exception as e:
    print(f"❌ Error checking PyTorch: {e}")

print()

print("=== Attempting to import imageProcessingUtils ===")
try:
    import imageProcessingUtils
    print(f"✅ Successfully imported imageProcessingUtils from: {imageProcessingUtils.__file__}")
    print(f"Package location: {Path(imageProcessingUtils.__file__).parent}")
except ImportError as e:
    print(f"❌ Failed to import imageProcessingUtils: {e}")
    sys.exit(1)

print()
print("=== Attempting to import YOLOSegmentation ===")
try:
    from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
    print("✅ Successfully imported YOLOSegmentation")
except ImportError as e:
    print(f"❌ Failed to import YOLOSegmentation: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=== Attempting to create YOLOSegmentation instance ===")
try:
    print("Creating YOLOSegmentation instance...")
    segmenter = YOLOSegmentation()
    print("✅ Successfully created YOLOSegmentation instance")
    print(f"Model path: {segmenter.model_path}")
    print(f"Model loaded: {segmenter.model is not None}")
    print(f"Is available: {segmenter.is_available()}")
    
    if segmenter.model_path:
        print(f"Model file exists: {segmenter.model_path.exists()}")
        if segmenter.model_path.exists():
            print(f"Model file size: {segmenter.model_path.stat().st_size / (1024*1024):.1f} MB")
    
except Exception as e:
    print(f"❌ Failed to create YOLOSegmentation instance: {e}")
    traceback.print_exc()
    
print()
print("=== Checking package structure ===")
try:
    package_dir = Path(imageProcessingUtils.__file__).parent
    yolo_dir = package_dir / "yolo"
    models_dir = yolo_dir / "models"
    runs_dir = yolo_dir / "runs"
    
    print(f"Package directory: {package_dir}")
    print(f"  Exists: {package_dir.exists()}")
    print(f"YOLO directory: {yolo_dir}")
    print(f"  Exists: {yolo_dir.exists()}")
    print(f"Models directory: {models_dir}")
    print(f"  Exists: {models_dir.exists()}")
    
    if models_dir.exists():
        pt_files = list(models_dir.glob("*.pt"))
        print(f"  .pt files found: {len(pt_files)}")
        for pt_file in pt_files:
            print(f"    - {pt_file.name} ({pt_file.stat().st_size / (1024*1024):.1f} MB)")
    
    print(f"Runs directory: {runs_dir}")
    print(f"  Exists: {runs_dir.exists()}")
    
    if runs_dir.exists():
        train_dirs = list(runs_dir.glob("train*"))
        print(f"  Training directories found: {len(train_dirs)}")
        for train_dir in train_dirs:
            weights_dir = train_dir / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                if best_pt.exists():
                    print(f"    - {train_dir.name}/weights/best.pt ({best_pt.stat().st_size / (1024*1024):.1f} MB)")
                    
except Exception as e:
    print(f"❌ Error checking package structure: {e}")
    traceback.print_exc()

print()
print("=== Testing simple segmentation ===")
try:
    import numpy as np
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    print("Created test image")
    
    if 'segmenter' in locals() and segmenter.is_available():
        print("Running segmentation...")
        labels, mask = segmenter.segment(test_image, conf_thres=0.5)
        print(f"✅ Segmentation successful!")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Number of objects: {labels.max()}")
    else:
        print("❌ Segmenter not available, skipping segmentation test")
        
except Exception as e:
    print(f"❌ Segmentation test failed: {e}")
    traceback.print_exc()

print()
print("=== Debug Complete ===")
