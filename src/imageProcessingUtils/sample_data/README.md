# Sample Data Module

The `imageProcessingUtils.sample_data` module provides tools for acquiring and processing sample datasets for testing image processing and segmentation pipelines.

## Installation & Import

```python
# Install the package
pip install git+https://github.com/Zbedd/imageProcessingUtils.git

# Import the module
from imageProcessingUtils.sample_data import BBBC022Fetcher, fetch_bbbc022_samples
```

## Quick Start

### Convenience Function (Recommended)

```python
from imageProcessingUtils.sample_data import fetch_bbbc022_samples

# Fetch 5 in-focus nuclei images
images, metadata = fetch_bbbc022_samples(
    count=5,
    focal_planes=[0],     # In-focus only
    treatments=['DMSO'],  # Control condition
    seed=42              # Reproducible sampling
)

# Images are ready for YOLO segmentation
for img in images:
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
```

### Class-based Usage

```python
from imageProcessingUtils.sample_data import BBBC022Fetcher

# Initialize fetcher
fetcher = BBBC022Fetcher('./my_data_directory')

# Fetch with detailed control
images, metadata = fetcher.fetch_samples(
    count=10,
    treatments=['DMSO', 'Taxol'],  # Control vs treatment
    focal_planes=[0],              # In-focus only
    wells=['A01', 'A02'],          # Specific wells
    seed=123,                      # Reproducible results
    save_processed=True            # Save processed images to disk
)
```

## Integration with YOLO

The module is designed for seamless integration with the YOLO segmentation pipeline:

```python
from imageProcessingUtils.sample_data import fetch_bbbc022_samples
from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo

# Fetch sample data
images, metadata = fetch_bbbc022_samples(count=3, seed=42)

# Process with YOLO
for img, meta in zip(images, metadata):
    labels, mask = segmentation_pipeline_yolo(img, conf_thres=0.1)
    print(f"{meta['filename']}: Detected {labels.max()} nuclei")
```

## Available Functions

### Core Functions

- **`BBBC022Fetcher`** - Main class for dataset acquisition and processing
- **`fetch_bbbc022_samples()`** - Convenience function for quick data fetching

### Utility Functions

- **`get_available_treatments()`** - List available drug treatments
- **`get_available_focal_planes()`** - List available focal planes (-3 to +3)

## Filtering Options

### Treatments
- `DMSO` - Control
- `AZ138` - Aurora kinase inhibitor
- `AZ841` - Aurora kinase inhibitor  
- `BIRB796` - p38 MAPK inhibitor
- `BMS345541` - IKK inhibitor
- `Taxol` - Microtubule stabilizer

### Focal Planes
- `-3` to `+3` (0 = in focus, negative/positive = out of focus)

### Other Filters
- **Wells**: Specific plate wells (e.g., 'A01', 'B02')
- **Channels**: Nuclei only (default) or all imaging channels

## Command Line Interface

You can also use the CLI tool for dataset acquisition:

```bash
# Navigate to the module directory
cd src/imageProcessingUtils/sample_data

# Fetch data via command line
python cli.py --count 10 --focal-plane 0 --seed 42
python cli.py --treatment DMSO Taxol --count 5 --output ./my_data
```

## Output Format

- **Images**: 2D numpy arrays (uint8, 0-255 range) compatible with YOLO pipeline
- **Metadata**: Detailed information about each image including treatment, focal plane, well position
- **File Structure**: Organized directories with raw, processed, and metadata files

## Examples

See the following example scripts:
- `example_external_usage.py` - Complete usage examples for external projects
- `test_module.py` - Import and functionality tests
- `cli.py` - Command-line interface

## Dataset Information

**BBBC022**: Human U2OS cells - Out of focus fluorescence microscopy
- **Source**: Broad Bioimage Benchmark Collection
- **URL**: https://bbbc.broadinstitute.org/BBBC022
- **Size**: ~500MB (1,368 images)
- **Format**: 16-bit TIFF images converted to 8-bit for YOLO compatibility
- **Quality**: Multiple focal planes for robustness testing
