"""
Sample Data Acquisition Module

This module provides tools for acquiring and processing sample datasets 
for testing image processing and segmentation pipelines.

Main Components:
- BBBC022Fetcher: Download and process BBBC022 microscopy dataset
- Filtering and sampling utilities for reproducible data selection
- YOLO-compatible format conversion

Example:
    ```python
    from imageProcessingUtils.sample_data import BBBC022Fetcher, fetch_bbbc022_samples
    
    # Using the class directly
    fetcher = BBBC022Fetcher('./data')
    images, metadata = fetcher.fetch_samples(count=5, seed=42)
    
    # Using convenience function
    images, metadata = fetch_bbbc022_samples(count=5, seed=42)
    ```
"""

from .bbbc022 import (
    BBBC022Fetcher,
    fetch_bbbc022_samples,
    get_available_treatments,  # Legacy compatibility
    get_available_channels,
    get_available_roles,
    get_available_focal_planes
)

__all__ = [
    'BBBC022Fetcher',
    'fetch_bbbc022_samples',
    'get_available_treatments',  # Legacy compatibility
    'get_available_channels',
    'get_available_roles', 
    'get_available_focal_planes',
]

__version__ = '1.0.0'
