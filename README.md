# imageProcessingUtils

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for advanced image processing, YOLO-based segmentation, and microscopy analysis. Consists of various tools I have found myself reusing across my projects.

## Features

### YOLO Segmentation

- **Fine-tuned nuclei detection** - Pre-trained models optimized for microscopy images
- **Automatic model discovery** - Works seamlessly across development and production environments
- **Device flexibility** - Automatic CPU/GPU detection with fallback support
- **High accuracy** - Specialized for biological image analysis
- **Model training utilities** - Build datasets and train custom YOLO models

### Image Processing

- **Basic preprocessing** - DAPI image preprocessing and enhancement
- **Gaussian filtering** - Noise reduction and image smoothing
- **Median filtering** - Salt-and-pepper noise removal
- **Multi-format support** - Handle various image formats and bit depths (planned)
- **Advanced filtering suite** - Edge detection, morphological operations (planned)

### File I/O

- **ND2 file support** - Native reading and characterization of Nikon microscopy files
- **Metadata preservation** - Extract and maintain important imaging parameters
- **Batch processing** - Handle multiple files and formats efficiently (planned)
- **Additional format support** - TIFF, CZI, and other microscopy formats (planned)

### Visualization (planned)

- **Static plotting** - Display images and segmentation results (planned)
- **Interactive tools** - Data exploration and analysis interfaces (planned)
- **Batch visualization** - Generate reports for multiple images (planned)

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
│   │   ├── preprocessing.py      # DAPI preprocessing functions
│   │   └── filters.py           # Gaussian and median filters
│   ├── file_io/                  # File I/O operations
│   │   ├── nd2_reader.py        # ND2 file reading and characterization
│   │   └── data_loaders.py      # General data loading (basic)
│   └── visualization/            # Visualization tools (planned)
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
