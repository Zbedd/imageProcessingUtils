#!/usr/bin/env python3
"""
Comprehensive test script for imageProcessingUtils package
Use this to verify the package is installed and working correctly
"""

import sys
import traceback

def test_basic_import():
    """Test basic package import"""
    print("🔍 Testing basic package import...")
    try:
        import imageProcessingUtils
        print(f"✅ SUCCESS: imageProcessingUtils imported correctly")
        print(f"   Version: {imageProcessingUtils.__version__}")
        print(f"   Author: {imageProcessingUtils.__author__}")
        print(f"   Available modules: {imageProcessingUtils.__all__}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not import imageProcessingUtils")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def test_submodule_imports():
    """Test submodule imports"""
    print("\n🔍 Testing submodule imports...")
    try:
        from imageProcessingUtils import yolo, image_processing, file_io
        print("✅ SUCCESS: All submodules imported correctly")
        print(f"   yolo: {yolo}")
        print(f"   image_processing: {image_processing}")
        print(f"   file_io: {file_io}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not import submodules")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def test_specific_imports():
    """Test specific class/function imports"""
    print("\n🔍 Testing specific imports...")
    try:
        from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
        from imageProcessingUtils.image_processing.preprocessing import preprocess_dapi
        from imageProcessingUtils.yolo.model_training import train_yolov8
        print("✅ SUCCESS: Specific classes and functions imported correctly")
        print(f"   YOLOSegmentation: {YOLOSegmentation}")
        print(f"   preprocess_dapi: {preprocess_dapi}")
        print(f"   train_yolov8: {train_yolov8}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not import specific items")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def test_package_location():
    """Test package installation location"""
    print("\n🔍 Testing package location...")
    try:
        import imageProcessingUtils
        import os
        location = os.path.dirname(imageProcessingUtils.__file__)
        print(f"✅ Package location: {location}")
        return True
    except Exception as e:
        print(f"❌ Could not determine package location: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 imageProcessingUtils Package Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_import,
        test_submodule_imports,
        test_specific_imports,
        test_package_location
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ The imageProcessingUtils package is installed and working correctly!")
        print("\n📝 You can now use it in your code like this:")
        print("   import imageProcessingUtils")
        print("   from imageProcessingUtils.yolo import segmentation")
        print("   from imageProcessingUtils.image_processing import preprocessing")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure the package is installed: pip install imageProcessingUtils")
        print("2. Check your Python environment")
        print("3. Try reinstalling: pip uninstall imageProcessingUtils && pip install imageProcessingUtils")

if __name__ == "__main__":
    main()
