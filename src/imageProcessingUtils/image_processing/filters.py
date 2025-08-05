"""
Image filters utilities
"""

import cv2
import numpy as np

def apply_gaussian_filter(img: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian filter to an image.
    
    Args:
        img: Input image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Filtered image
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)

def apply_median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to remove noise.
    
    Args:
        img: Input image
        kernel_size: Size of the median filter kernel
        
    Returns:
        Filtered image
    """
    return cv2.medianBlur(img, kernel_size)
