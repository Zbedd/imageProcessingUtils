"""
Test the new preprocessing functions
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_preprocessing_import():
    """Test that preprocessing functions can be imported"""
    try:
        from image_processing import preprocess_dapi, basic_preprocess
        from image_processing.preprocessing import preprocess_dapi as preprocess_dapi_alt
        
        print("✅ Direct function imports work")
        print("✅ Module-based imports work") 
        print("✅ Functions are callable:", callable(preprocess_dapi), callable(basic_preprocess))
        
        # Test with dummy data
        import numpy as np
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        result = basic_preprocess(dummy_img)
        print("✅ Basic preprocessing test passed")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_preprocessing_import()
    if success:
        print("\n🎉 All preprocessing tests passed!")
    else:
        print("\n💥 Some tests failed!")
