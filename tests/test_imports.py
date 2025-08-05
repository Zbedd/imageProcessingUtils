"""
Test basic imports for the flattened structure
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import_main_modules():
    """Test that main modules can be imported"""
    import yolo
    import image_processing
    import file_io
    import visualization
    
    assert yolo is not None
    assert image_processing is not None
    assert file_io is not None
    assert visualization is not None

def test_import_submodules():
    """Test that submodules can be imported"""
    from yolo import model_training, segmentation
    from image_processing import preprocessing, filters
    from file_io import nd2_reader, data_loaders
    from visualization import plots, interactive
    
    assert model_training is not None
    assert segmentation is not None
    assert preprocessing is not None
    assert filters is not None
    assert nd2_reader is not None
    assert data_loaders is not None
    assert plots is not None
    assert interactive is not None

if __name__ == "__main__":
    test_import_main_modules()
    test_import_submodules()
    print("All import tests passed!")
