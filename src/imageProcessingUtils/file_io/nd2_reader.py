"""
ND2 file reading utilities.
"""
import numpy as np
from pathlib import Path

def characterize_nd2(nd2_path, verbose=True):
    """Get a comprehensive characterization of the ND2 file.
    
    Args:
        nd2_path: Path to ND2 file
        verbose: If True, print detailed information (default: True)
        
    Returns:
        dict with detailed file characterization including:
        - type: File type ('ND2')
        - dimensions: Overall shape tuple
        - frame_count: Total number of frames
        - axes: List of dimension characters (e.g., 't', 'c', 'z', 'y', 'x')
        - axis_info: Dict mapping each axis to its size and description
        - channel_names: List of channel names (if channels present)
    """
    try:
        from nd2reader import ND2Reader
    except ImportError:
        raise ImportError("nd2reader required. Install with: pip install nd2reader")
    
    nd2_path = Path(nd2_path)
    if not nd2_path.exists():
        raise FileNotFoundError(f"File not found: {nd2_path}")
    
    with ND2Reader(str(nd2_path)) as images:
        # Get basic info
        frame_count = len(images)
        sizes = getattr(images, 'sizes', {})
        axes = getattr(images, 'axes', [])
        
        # Build axis information
        axis_info = {}
        axis_descriptions = {
            't': 'time/frames',
            'c': 'channels', 
            'z': 'z-stack/depth',
            'v': 'field of view',
            'y': 'height',
            'x': 'width'
        }
        
        for axis in axes:
            size = sizes.get(axis, 1)
            description = axis_descriptions.get(axis, f'dimension {axis}')
            axis_info[axis] = {
                'size': size,
                'description': description
            }
        
        # Get channel names if channels exist
        channel_names = []
        if 'c' in axes and hasattr(images, 'metadata') and images.metadata:
            try:
                # Try to extract channel names from metadata
                channels_info = images.metadata.get('channels', [])
                if channels_info:
                    channel_names = [ch.get('name', f'Channel_{i}') for i, ch in enumerate(channels_info)]
                elif hasattr(images, 'channel_names'):
                    channel_names = list(images.channel_names)
                else:
                    # Fallback to generic names
                    num_channels = sizes.get('c', 1)
                    channel_names = [f'Channel_{i}' for i in range(num_channels)]
            except Exception:
                # Fallback if channel name extraction fails
                num_channels = sizes.get('c', 1)
                channel_names = [f'Channel_{i}' for i in range(num_channels)]
        
        # Build result dictionary
        result = {
            'type': 'ND2',
            'filename': nd2_path.name,
            'dimensions': tuple(sizes.get(axis, 1) for axis in axes) if axes else (),
            'frame_count': frame_count,
            'axes': axes,
            'axis_info': axis_info,
            'channel_names': channel_names,
            'total_size_bytes': nd2_path.stat().st_size if nd2_path.exists() else 0
        }
        
        # Print verbose information if requested
        if verbose:
            print(f"\nND2 File Analysis: {nd2_path.name}")
            print("=" * 50)
            print(f"File size: {result['total_size_bytes'] / (1024*1024):.1f} MB")
            print(f"Total frames: {frame_count}")
            
            if axes:
                print(f"\nDimension structure: {' × '.join(axes)}")
                print(f"Shape: {result['dimensions']}")
                
                print("\nDimension details:")
                for axis in axes:
                    info = axis_info[axis]
                    print(f"  {axis.upper()}: {info['size']} ({info['description']})")
                
                if channel_names:
                    print(f"\nChannel information:")
                    for i, name in enumerate(channel_names):
                        print(f"  Channel {i}: {name}")
            else:
                print("\nNo dimension information available")
        
        return result

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
