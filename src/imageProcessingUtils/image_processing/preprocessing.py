"""
Image preprocessing utilities
"""

import cv2
import numpy as np

def preprocess_dapi(img: np.ndarray, blur_kernel: tuple = (3, 3), blur_sigma: float = 0) -> np.ndarray:
    """
    Preprocess DAPI images for nuclei segmentation.
    
    Args:
        img: Input grayscale image
        blur_kernel: Gaussian blur kernel size
        blur_sigma: Gaussian blur sigma (0 = auto)
        
    Returns:
        Preprocessed image
    """
    if blur_sigma == 0:
        # Auto-calculate sigma based on kernel size
        blur_sigma = 0.3 * ((blur_kernel[0] - 1) * 0.5 - 1) + 0.8
    
    return cv2.GaussianBlur(img, blur_kernel, blur_sigma)

def basic_preprocess(img: np.ndarray) -> np.ndarray:
    """Basic preprocessing fallback."""
    return cv2.GaussianBlur(img, (3, 3), 0)
