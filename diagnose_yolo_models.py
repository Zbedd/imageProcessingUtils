"""
Diagnostic script to check YOLO model discovery in imageProcessingUtils package.
Run this in your other repo to diagnose the model loading issue.
"""
import sys
from pathlib import Path

print("=== YOLO Model Discovery Diagnostic ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import imageProcessingUtils
    print(f"✅ imageProcessingUtils imported from: {imageProcessingUtils.__file__}")
    
    # Check package location
    package_path = Path(imageProcessingUtils.__file__).parent
    print(f"Package location: {package_path}")
    
    # Check for yolo module
    yolo_path = package_path / "yolo"
    print(f"YOLO module exists: {yolo_path.exists()}")
    
    if yolo_path.exists():
        # Check models directory
        models_dir = yolo_path / "models"
        print(f"Models directory exists: {models_dir.exists()}")
        
        if models_dir.exists():
            pt_files = list(models_dir.glob("*.pt"))
            print(f"Found {len(pt_files)} .pt files in models directory:")
            for pt_file in pt_files:
                size_mb = pt_file.stat().st_size / (1024*1024)
                print(f"  - {pt_file.name} ({size_mb:.1f} MB)")
        
        # Check runs directory
        runs_dir = yolo_path / "runs" / "segment"
        print(f"Runs directory exists: {runs_dir.exists()}")
        
        if runs_dir.exists():
            best_pts = list(runs_dir.glob("train*/weights/best.pt"))
            print(f"Found {len(best_pts)} training best.pt files:")
            for best_pt in best_pts:
                size_mb = best_pt.stat().st_size / (1024*1024)
                print(f"  - {best_pt} ({size_mb:.1f} MB)")
    
    # Test model discovery
    print("\n=== Testing Model Discovery ===")
    from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
    
    # Create instance to trigger model discovery
    yolo_seg = YOLOSegmentation()
    print(f"Model path discovered: {yolo_seg.model_path}")
    print(f"Model available: {yolo_seg.is_available()}")
    
    if yolo_seg.model_path:
        print(f"Model file exists: {yolo_seg.model_path.exists()}")
        if yolo_seg.model_path.exists():
            size_mb = yolo_seg.model_path.stat().st_size / (1024*1024)
            print(f"Model file size: {size_mb:.1f} MB")

except ImportError as e:
    print(f"❌ Failed to import imageProcessingUtils: {e}")
except Exception as e:
    print(f"❌ Error during diagnostic: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Recommendations ===")
print("1. If no models found: The package may not include model files")
print("2. If models found but path is None: Model discovery logic issue")
print("3. If fine-tuned model missing: Need to install/copy fine-tuned model")
