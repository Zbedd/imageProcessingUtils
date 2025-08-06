#!/usr/bin/env python3
"""
Simple test script for ND2 file characterization.

Usage:
    python test_file_io.py path/to/your/file.nd2
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_nd2_characterization(nd2_path):
    """Test the ND2 characterization function."""
    try:
        from imageProcessingUtils.file_io.nd2_reader import characterize_nd2
        
        print(f"Testing ND2 characterization for: {nd2_path}")
        print("-" * 60)
        
        # Test with verbose output
        result = characterize_nd2(nd2_path, verbose=True)
        
        print("\n" + "=" * 60)
        print("Returned data structure:")
        print("=" * 60)
        for key, value in result.items():
            if key == 'axis_info':
                print(f"{key}: {dict(value)}")
            else:
                print(f"{key}: {value}")
                
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure nd2reader is installed: pip install nd2reader")
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python test_file_io.py path/to/your/file.nd2")
        print("\nExample:")
        print("  python test_file_io.py C:/data/microscopy/sample.nd2")
        sys.exit(1)
    
    nd2_path = sys.argv[1]
    
    if not Path(nd2_path).exists():
        print(f"File not found: {nd2_path}")
        sys.exit(1)
    
    if not nd2_path.lower().endswith('.nd2'):
        print(f"Warning: File doesn't have .nd2 extension: {nd2_path}")
    
    test_nd2_characterization(nd2_path)

if __name__ == "__main__":
    main()
