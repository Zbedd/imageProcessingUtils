#!/usr/bin/env python3
"""
Example: Using imageProcessingUtils.sample_data in External Projects

This script demonstrates how to use the sample_data module from imageProcessingUtils
in other repositories or projects.
"""

import sys
from pathlib import Path
import numpy as np

def example_basic_usage():
    """Example 1: Basic usage with convenience function."""
    print("Example 1: Basic Usage")
    print("-" * 30)
    
    try:
        from imageProcessingUtils.sample_data import fetch_bbbc022_samples
        
        # Fetch a small sample for quick testing
        print("Fetching 3 sample images...")
        images, metadata = fetch_bbbc022_samples(
            count=3,
            focal_planes=[0],     # In-focus only
            treatments=['DMSO'],  # Control condition
            seed=42,              # Reproducible results
            output_dir='./example_data'
        )
        
        if images:
            print(f"✓ Successfully fetched {len(images)} images")
            for i, (img, meta) in enumerate(zip(images, metadata)):
                print(f"  Image {i+1}: {img.shape}, {img.dtype}, {meta['filename']}")
            return True
        else:
            print("✗ No images fetched (may need internet connection)")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure imageProcessingUtils is installed:")
        print("  pip install git+https://github.com/Zbedd/imageProcessingUtils.git")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_class_usage():
    """Example 2: Using the BBBC022Fetcher class directly."""
    print("\nExample 2: Class-based Usage")
    print("-" * 30)
    
    try:
        from imageProcessingUtils.sample_data import BBBC022Fetcher
        
        # Initialize fetcher with custom output directory
        fetcher = BBBC022Fetcher('./custom_data_location')
        
        print("Available treatments:", fetcher.AVAILABLE_TREATMENTS)
        print("Available focal planes:", fetcher.FOCAL_PLANES)
        
        # Fetch with more specific criteria
        print("\nFetching images with specific criteria...")
        images, metadata = fetcher.fetch_samples(
            count=2,
            treatments=['DMSO', 'Taxol'],  # Control vs treatment
            focal_planes=[0],              # In-focus only
            wells=['A01', 'A02'],          # Specific wells
            seed=123,                      # Different seed
            save_processed=True            # Save to disk
        )
        
        if images:
            print(f"✓ Fetched {len(images)} images")
            
            # Show metadata details
            for meta in metadata:
                print(f"  {meta['filename']}: {meta['treatment']} treatment, "
                      f"well {meta['well']}, focal plane {meta['focal_plane']}")
            return True
        else:
            print("✗ No images fetched")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_with_yolo():
    """Example 3: Integration with YOLO segmentation."""
    print("\nExample 3: YOLO Integration")
    print("-" * 30)
    
    try:
        from imageProcessingUtils.sample_data import fetch_bbbc022_samples
        from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo
        
        # Fetch sample images
        print("Fetching sample images for YOLO processing...")
        images, metadata = fetch_bbbc022_samples(
            count=2,
            focal_planes=[0],     # In-focus for better segmentation
            treatments=['DMSO'],  # Control condition
            seed=42,
            output_dir='./yolo_test_data'
        )
        
        if not images:
            print("✗ No images available for YOLO testing")
            return False
        
        # Process with YOLO
        print(f"\nProcessing {len(images)} images with YOLO...")
        for i, (img, meta) in enumerate(zip(images, metadata)):
            print(f"Processing {meta['filename']}...")
            
            try:
                labels, mask = segmentation_pipeline_yolo(img, conf_thres=0.1)
                nuclei_count = labels.max()
                
                print(f"  ✓ Detected {nuclei_count} nuclei")
                print(f"    Image shape: {img.shape}")
                print(f"    Mask coverage: {(mask > 0).sum() / mask.size * 100:.1f}%")
                
            except Exception as e:
                print(f"  ✗ YOLO processing failed: {e}")
                print(f"    (This may be due to missing YOLO model files)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_metadata_analysis():
    """Example 4: Analyzing dataset metadata."""
    print("\nExample 4: Metadata Analysis")
    print("-" * 30)
    
    try:
        from imageProcessingUtils.sample_data import (
            BBBC022Fetcher,
            get_available_treatments,
            get_available_focal_planes
        )
        
        # Get dataset information
        treatments = get_available_treatments()
        focal_planes = get_available_focal_planes()
        
        print(f"Dataset contains:")
        print(f"  Treatments: {len(treatments)} ({', '.join(treatments)})")
        print(f"  Focal planes: {len(focal_planes)} (range: {min(focal_planes)} to {max(focal_planes)})")
        
        # Fetch diverse sample for analysis
        fetcher = BBBC022Fetcher('./metadata_analysis')
        images, metadata = fetcher.fetch_samples(
            count=5,
            treatments=treatments[:3],    # First 3 treatments
            focal_planes=[-1, 0, 1],     # Out-of-focus to in-focus
            seed=456,
            save_processed=False         # Don't save to disk
        )
        
        if metadata:
            print(f"\nAnalyzing {len(metadata)} sample images:")
            
            # Group by treatment
            by_treatment = {}
            by_focal_plane = {}
            
            for meta in metadata:
                treatment = meta['treatment']
                focal_plane = meta['focal_plane']
                
                if treatment not in by_treatment:
                    by_treatment[treatment] = 0
                by_treatment[treatment] += 1
                
                if focal_plane not in by_focal_plane:
                    by_focal_plane[focal_plane] = 0
                by_focal_plane[focal_plane] += 1
            
            print("  By treatment:", dict(by_treatment))
            print("  By focal plane:", dict(by_focal_plane))
            
            # Image properties
            shapes = [meta['shape'] for meta in metadata]
            print(f"  Image shapes: {set(shapes)}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all examples."""
    print("imageProcessingUtils.sample_data Usage Examples")
    print("=" * 50)
    print("These examples show how to use the sample_data module")
    print("in external projects or repositories.\n")
    
    examples = [
        example_basic_usage,
        example_class_usage,
        example_with_yolo,
        example_metadata_analysis
    ]
    
    success_count = 0
    for example_func in examples:
        try:
            if example_func():
                success_count += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"Unexpected error in {example_func.__name__}: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Examples completed: {success_count}/{len(examples)} successful")
    
    if success_count > 0:
        print("\n✓ The sample_data module is working and ready for use!")
        print("\nTo use in your own projects:")
        print("1. Install imageProcessingUtils:")
        print("   pip install git+https://github.com/Zbedd/imageProcessingUtils.git")
        print("2. Import and use:")
        print("   from imageProcessingUtils.sample_data import fetch_bbbc022_samples")
        print("   images, metadata = fetch_bbbc022_samples(count=10, seed=42)")
    else:
        print("\n✗ Issues detected. Check your installation and internet connection.")


if __name__ == "__main__":
    main()
