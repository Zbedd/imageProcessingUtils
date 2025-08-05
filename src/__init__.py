"""
imageProcessingUtils - Image Processing and YOLO Utilities Package

A comprehensive Python package for image processing, YOLO model training, and data analysis utilities.
"""

# Import the main package to maintain backward compatibility
from .imageProcessingUtils import *
from .imageProcessingUtils import __version__, __author__

__all__ = [
    "yolo",
    "image_processing", 
    "file_io"
]
