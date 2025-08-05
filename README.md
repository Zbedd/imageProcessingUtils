# imageProcessingUtils

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for advanced image processing, YOLO-based segmentation, and microscopy analysis.

## Features

### YOLO Segmentation
- **Fine-tuned nuclei detection** - Pre-trained models optimized for microscopy images
- **Automatic model discovery** - Works seamlessly across development and production environments
- **Device flexibility** - Automatic CPU/GPU detection with fallback support
- **High accuracy** - Specialized for biological image analysis

### Image Processing
- **Microscopy-focused tools** - Preprocessing and enhancement for biological images
- **Multi-format support** - Handle various image formats and bit depths
- **Efficient processing** - Optimized algorithms for large image datasets

### File I/O
- **ND2 file support** - Native reading of Nikon microscopy files
- **Batch processing** - Handle multiple files and formats efficiently
- **Metadata preservation** - Maintain important imaging parameters

## Quick Start

### Installation

```bash
# Install from GitHub (recommended)
pip install git+https://github.com/Zbedd/imageProcessingUtils.git

# Or install in development mode
git clone https://github.com/Zbedd/imageProcessingUtils.git
cd imageProcessingUtils
pip install -e .
```

### Basic Usage

```python
from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
import numpy as np

# Initialize segmentation (automatically finds best model)
segmenter = YOLOSegmentation()

# Load your microscopy image
image = np.load('your_image.npy')  # or use cv2.imread(), etc.

# Segment nuclei
if segmenter.is_available():
    labels, mask = segmenter.segment(image, conf_thres=0.05)
    print(f"Detected {labels.max()} nuclei")
```

### High-level Pipeline

```python
from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo

# One-line segmentation
labels, mask = segmentation_pipeline_yolo(image, conf_thres=0.05)
```

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running in minutes
- **[Installation Guide](docs/INSTALL_IN_OTHER_ENV.md)** - Detailed installation instructions
- **[API Reference](docs/api_reference.md)** - Complete function documentation
- **[YOLO Guide](docs/yolo.md)** - Deep dive into YOLO segmentation features

## Project Structure

```
imageProcessingUtils/
├── src/imageProcessingUtils/     # Main package
│   ├── yolo/                     # YOLO segmentation tools
│   │   ├── segmentation.py       # Main segmentation interface
│   │   ├── model_training.py     # Training utilities
│   │   └── models/               # Pre-trained model files
│   ├── image_processing/         # Image processing utilities
│   └── file_io/                  # File I/O operations
├── docs/                         # Documentation
├── examples/                     # Usage examples
└── tests/                        # Test suite
```

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 1.9+ with CPU/GPU support
- **OpenCV**: 4.5+ for image processing
- **Ultralytics**: 8.0+ for YOLO functionality

See [requirements.txt](requirements.txt) for complete dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
