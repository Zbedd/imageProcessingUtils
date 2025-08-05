# imageProcessingUtils - New Import Examples

After restructuring to Option 2 (unified package), here are the new import patterns:

## ✅ **NEW Import Patterns (Recommended)**

```python
# Import the main package
import imageProcessingUtils

# Import submodules
from imageProcessingUtils import yolo, image_processing, file_io

# Import specific modules
from imageProcessingUtils.yolo import model_training, segmentation
from imageProcessingUtils.image_processing import preprocessing, filters
from imageProcessingUtils.file_io import nd2_reader, data_loaders

# Import specific classes and functions
from imageProcessingUtils.yolo.segmentation import YOLOSegmentation
from imageProcessingUtils.yolo.model_training import train_yolov8
from imageProcessingUtils.image_processing.preprocessing import preprocess_dapi, basic_preprocess
```

## 🔄 **Backward Compatibility (Still Works)**

```python
# Old style imports still work for backward compatibility
import src
from src import yolo, image_processing, file_io
```

## 🎯 **Benefits of New Structure**

1. **Clear Package Namespace**: When installed via pip, users import `imageProcessingUtils`
2. **Professional Structure**: Follows Python packaging best practices
3. **Backward Compatible**: Old import style still works during transition
4. **Consistent Naming**: Package name matches import name

## 📦 **Console Scripts Available**

After installation, these commands are available in terminal:
- `imageProcessingUtils-train` - Runs YOLO model training
- `imageProcessingUtils-segment` - Runs YOLO segmentation
