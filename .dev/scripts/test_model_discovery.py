"""
Quick test of YOLO model discovery to debug the external repo issue.
"""
import sys
from pathlib import Path

# Add the src directory to Python path to import from development version
sys.path.insert(0, r'c:\VScode\imageProcessingUtils\src')

print("=== YOLO Model Discovery Test ===")

try:
    from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
    
    print("\n1. Testing model discovery...")
    yolo_seg = YOLOSegmentation()
    
    print(f"Model path discovered: {yolo_seg.model_path}")
    print(f"Model available: {yolo_seg.is_available()}")
    
    if yolo_seg.model_path:
        print(f"Model file exists: {yolo_seg.model_path.exists()}")
        if yolo_seg.model_path.exists():
            size_mb = yolo_seg.model_path.stat().st_size / (1024*1024)
            print(f"Model file size: {size_mb:.1f} MB")
    
    print("\n2. Manual check of models directory...")
    models_dir = Path(__file__).parent / "src" / "imageProcessingUtils" / "yolo" / "models"
    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {models_dir.exists()}")
    
    if models_dir.exists():
        pt_files = list(models_dir.glob("*.pt"))
        print(f"Found {len(pt_files)} .pt files:")
        for pt_file in pt_files:
            size_mb = pt_file.stat().st_size / (1024*1024)
            print(f"  - {pt_file.name} ({size_mb:.1f} MB)")
        
        # Check specifically for fine-tuned models
        fine_tuned = list(models_dir.glob("best_*.pt"))
        print(f"Fine-tuned models (best_*.pt): {len(fine_tuned)}")
        for model in fine_tuned:
            print(f"  - {model.name}")
    
    print("\n3. Testing segmentation on a simple image...")
    import numpy as np
    
    if yolo_seg.is_available():
        # Create a simple test image (synthetic microscopy-like data)
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        try:
            labels, mask = yolo_seg.segment(test_image, conf_thres=0.1)
            print(f"✅ Segmentation successful!")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Max label (nuclei count): {labels.max()}")
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Model not available for testing")
        
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()