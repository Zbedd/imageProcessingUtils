"""
Test that the package is properly installed and can be imported
"""

def test_package_import():
    """Test that the main package can be imported"""
    try:
        # Test direct module imports
        import yolo
        import image_processing
        import file_io
        import visualization
        
        # Test that modules have expected attributes
        assert hasattr(yolo, '__all__')
        assert hasattr(image_processing, '__all__')
        assert hasattr(file_io, '__all__')
        assert hasattr(visualization, '__all__')
        
    except ImportError as e:
        assert False, f"Failed to import package modules: {e}"

def test_submodule_imports():
    """Test that submodules can be imported"""
    try:
        from yolo import model_training, segmentation
        from image_processing import preprocessing, filters
        from file_io import nd2_reader, data_loaders
        from visualization import plots, interactive
        
        # All should be modules
        import types
        assert isinstance(model_training, types.ModuleType)
        assert isinstance(segmentation, types.ModuleType)
        assert isinstance(preprocessing, types.ModuleType)
        assert isinstance(filters, types.ModuleType)
        assert isinstance(nd2_reader, types.ModuleType)
        assert isinstance(data_loaders, types.ModuleType)
        assert isinstance(plots, types.ModuleType)
        assert isinstance(interactive, types.ModuleType)
        
    except ImportError as e:
        assert False, f"Failed to import submodules: {e}"
