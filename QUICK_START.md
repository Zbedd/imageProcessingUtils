# 🚀 imageProcessingUtils - Quick Start Guide

## ✅ **Correct Import Patterns**

The package is installed correctly! Here are the **exact import patterns** to use:

### Basic Import
```python
import imageProcessingUtils
print(f"Version: {imageProcessingUtils.__version__}")
```

### Import Submodules
```python
from imageProcessingUtils import yolo, image_processing, file_io
```

### Import Specific Functions/Classes
```python
# YOLO functionality
from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
from imageProcessingUtils.yolo.model_training import train_yolov8

# Image processing
from imageProcessingUtils.image_processing.preprocessing import preprocess_dapi
from imageProcessingUtils.image_processing.filters import median_filter

# File I/O
from imageProcessingUtils.file_io.nd2_reader import load_nd2_file
from imageProcessingUtils.file_io.data_loaders import load_image_batch
```

## 🧪 **Test Your Installation**

Run this command in your terminal to verify everything works:
```bash
python package_test_suite.py
```

## ⚠️ **Common Issues & Solutions**

### Issue 1: "No module named 'imageProcessingUtils'"
**Solution:** Make sure you're in the correct Python environment where the package is installed.

### Issue 2: Case sensitivity problems
**Solution:** Use exact casing: `imageProcessingUtils` (not `imageprocessingutils`)

### Issue 3: Import errors for submodules
**Solution:** Always use the full path:
```python
# ✅ Correct
from imageProcessingUtils.yolo import segmentation

# ❌ Wrong
from yolo import segmentation
```

## 🔧 **Reinstallation (if needed)**
```bash
pip uninstall imageProcessingUtils
pip install -e .
```

## 📁 **Package Structure**
```
imageProcessingUtils/
├── yolo/                    # YOLO model training and segmentation
│   ├── model_training.py    # train_yolov8() function
│   └── segmentation.py      # YOLOSegmentation class
├── image_processing/        # Image preprocessing utilities
│   ├── preprocessing.py     # preprocess_dapi(), basic_preprocess()
│   └── filters.py          # Various image filters
└── file_io/                # Data loading utilities
    ├── nd2_reader.py       # ND2 file handling
    └── data_loaders.py     # Batch loading functions
```

## 🎯 **Ready to Use!**

The package is working correctly. You can now use it in your TUNEL scripts or any other project!
