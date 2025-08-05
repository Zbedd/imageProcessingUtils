# YOLO Module Documentation

The YOLO module provides utilities for training and using YOLO (You Only Look Once) object detection and segmentation models.

## Components

### model_training.py
Contains functions for training YOLO models on custom datasets.

**Key Functions:**
- `create_model()` - Initialize a new YOLO model
- `train()` - Train the model on provided data
- `validate()` - Validate model performance

### segmentation.py
Provides segmentation utilities using trained YOLO models.

**Key Functions:**
- `segment_image()` - Perform segmentation on a single image
- `batch_segment()` - Process multiple images
- `extract_masks()` - Extract segmentation masks

## Data Structure

### models/
Directory for storing trained YOLO model files (.pt files).

### data/
Directory for training datasets, organized by:
- `images/` - Training images
- `labels/` - Corresponding annotation files
- `config.yaml` - Dataset configuration

## Usage Examples

See `examples/yolo_training_example.py` and `examples/segmentation_example.py` for detailed usage examples.
