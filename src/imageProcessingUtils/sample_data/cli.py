#!/usr/bin/env python3
"""
Command-line interface for BBBC022 dataset fetching

This script provides a command-line interface to the BBBC022Fetcher class
for easy dataset acquisition and processing.

Usage:
    python cli.py --count 10 --seed 42 --output ./bbbc022_samples
    python cli.py --treatment DMSO Taxol --focal-plane 0 --count 5
"""

import argparse
import sys
from pathlib import Path

# Add the package to the path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from imageProcessingUtils.sample_data import BBBC022Fetcher


def main():
    """Command line interface for BBBC022 fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch and process BBBC022 sample images for YOLO segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get 10 random nuclei images from 2 plates
  python cli.py --count 10 --channel OrigHoechst --seed 42
  
  # Get multiple channels for comparison
  python cli.py --channel OrigHoechst OrigER --count 5 --seed 123
  
  # Get specific wells with all channels
  python cli.py --wells A01 A02 --include-all-channels --count 20
        """
    )
    
    parser.add_argument('--count', type=int, default=10,
                       help='Number of images to sample (default: 10)')
    
    parser.add_argument('--channel', nargs='+', 
                       choices=BBBC022Fetcher.AVAILABLE_CHANNELS,
                       help='Channels to include')
    
    parser.add_argument('--wells', nargs='+',
                       help='Specific wells to include (e.g., A01 B02)')
    
    parser.add_argument('--include-all-channels', action='store_true',
                       help='Include all channels (default: nuclei only)')
    
    parser.add_argument('--max-plates', type=int, default=2,
                       help='Maximum plates to download (default: 2)')
    
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible sampling')
    
    parser.add_argument('--output', default='./bbbc022_data',
                       help='Output directory (default: ./bbbc022_data)')
    
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force redownload of dataset')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = BBBC022Fetcher(args.output)
    
    # Force redownload if requested
    if args.force_redownload:
        try:
            fetcher.download_dataset(force_redownload=True, max_plates=args.max_plates, channels=args.channel)
        except Exception as e:
            print(f"Error during redownload: {e}")
            return 1
    
    # Fetch samples
    try:
        images, metadata = fetcher.fetch_samples(
            count=args.count,
            channels=args.channel,
            wells=args.wells,
            nuclei_only=not args.include_all_channels,
            seed=args.seed,
            max_plates=args.max_plates
        )
        
        if images:
            print(f"\n✓ Successfully fetched {len(images)} images")
            print(f"  Image shapes: {[img.shape for img in images[:3]]}...")
            print(f"  Data types: {[img.dtype for img in images[:3]]}...")
            print(f"\nImages are ready for yolo_segmentation_pipeline!")
            
            # Show example usage
            print(f"\nExample usage in Python:")
            print(f"```python")
            print(f"import numpy as np")
            print(f"from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo")
            print(f"")
            print(f"# Load a sample image")
            print(f"img = np.load('{args.output}/processed/bbbc022_sample_000.npy')")
            print(f"")
            print(f"# Run YOLO segmentation")
            print(f"labels, mask = segmentation_pipeline_yolo(img, conf_thres=0.1)")
            print(f"print(f'Detected {{labels.max()}} nuclei')")
            print(f"```")
        else:
            print("\n✗ No images were fetched. Check your filter criteria.")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
