"""
Test basic imports for the unified structure
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import_main_modules():
    """Test that main modules can be imported"""
    import imageProcessingUtils
    from imageProcessingUtils import yolo, image_processing, file_io
    
    assert imageProcessingUtils is not None
    assert yolo is not None
    assert image_processing is not None
    assert file_io is not None

def test_import_submodules():
    """Test that submodules can be imported"""
    from imageProcessingUtils.yolo import model_training, segmentation
    from imageProcessingUtils.image_processing import preprocessing, filters, preprocess_dapi, basic_preprocess
    from imageProcessingUtils.file_io import nd2_reader, data_loaders
    
    assert model_training is not None
    assert segmentation is not None
    assert preprocessing is not None
    assert filters is not None
    assert preprocess_dapi is not None
    assert basic_preprocess is not None
    assert nd2_reader is not None
    assert data_loaders is not None

if __name__ == "__main__":
    test_import_main_modules()
    test_import_submodules()
    print("All import tests passed!")
