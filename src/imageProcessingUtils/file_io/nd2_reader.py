"""
ND2 file reading utilities.
"""
import numpy as np
from pathlib import Path


def load_nd2_image(nd2_path):
    """Load an ND2 image file.
    
    Args:
        nd2_path: Path to ND2 file
        
    Returns:
        numpy array of image data
    """
    try:
        from nd2reader import ND2Reader
    except ImportError:
        raise ImportError("nd2reader required. Install with: pip install nd2reader")
    
    nd2_path = Path(nd2_path)
    if not nd2_path.exists():
        raise FileNotFoundError(f"File not found: {nd2_path}")
    
    with ND2Reader(str(nd2_path)) as images:
        if len(images) == 0:
            raise ValueError(f"No images in file: {nd2_path}")
        
        # Get first image
        image = np.array(images[0])
        
        # Flatten multi-dimensional arrays
        while image.ndim > 2:
            image = image[0]
            
        return image


def get_nd2_info(nd2_path):
    """Get basic info about ND2 file.
    
    Args:
        nd2_path: Path to ND2 file
        
    Returns:
        dict with file information
    """
    try:
        from nd2reader import ND2Reader
    except ImportError:
        raise ImportError("nd2reader required. Install with: pip install nd2reader")
    
    nd2_path = Path(nd2_path)
    with ND2Reader(str(nd2_path)) as images:
        return {
            'filename': nd2_path.name,
            'frame_count': len(images),
            'sizes': getattr(images, 'sizes', {}),
            'shape': getattr(images, 'frame_shape', None)
        }
