#!/usr/bin/env python3
"""
Test script to verify the unified package structure works correctly
"""

print("🔍 Testing unified imageProcessingUtils package structure...")

# Test 1: Import the main package
try:
    import imageProcessingUtils
    print("✓ Main imageProcessingUtils package imports successfully")
    print(f"✓ Available modules: {imageProcessingUtils.__all__}")
    print(f"✓ Version: {imageProcessingUtils.__version__}")
except Exception as e:
    print(f"✗ Error importing main package: {e}")

# Test 2: Import submodules using new structure
try:
    from imageProcessingUtils import yolo, image_processing, file_io
    print("✓ All submodules import successfully with new structure")
except Exception as e:
    print(f"✗ Error importing submodules: {e}")

# Test 3: Import specific functions/classes
try:
    from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
    from imageProcessingUtils.image_processing.preprocessing import preprocess_dapi
    print("✓ Specific classes and functions import successfully")
except Exception as e:
    print(f"✗ Error importing specific functions: {e}")

# Test 4: Test backward compatibility (old style imports)
try:
    import src
    from src import yolo as old_yolo
    print("✓ Backward compatibility maintained - old imports still work")
except Exception as e:
    print(f"✗ Backward compatibility issue: {e}")

print("\n🎉 Unified package structure verification complete!")
