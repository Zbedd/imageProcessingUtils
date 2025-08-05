"""YOLO-based segmentation functionality for nuclei detection."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

# Import DEFAULTS from model_training
from .model_training import DEFAULTS


class YOLOSegmentation:
    """YOLO-based nuclei segmentation handler."""
    
    def __init__(self, model_path: Path | str | None = None):
        """Initialize YOLO segmentation with model path."""
        self.model = None
        self.model_path = None
        
        if model_path is None:
            # Auto-discover the latest trained model
            self.model_path = self._find_latest_model()
        else:
            self.model_path = Path(model_path)
            
        self._load_model()
    
    def _find_latest_model(self) -> Path | None:
        """Find the most recently trained YOLO model."""
        # First check for models in the models directory (preferred)
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            try:
                model_path = max(
                    models_dir.glob("*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                print(f"Found model in models directory: {model_path}")
                return model_path
            except ValueError:
                pass  # No models in models directory, continue to runs/
        
        # Fall back to runs directory for auto-discovered training outputs
        runs_dir = Path(DEFAULTS.get("yolo_runs_dir", "runs/segment")).expanduser()
        
        try:
            # Find the most recent best.pt model
            model_path = max(
                runs_dir.glob("train*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            print(f"Found model in runs directory: {model_path}")
            return model_path
        except ValueError:
            # No models found
            return None
    
    def _load_model(self):
        """Load the YOLO model."""
        if self.model_path is None or not self.model_path.exists():
            print(f"⚠️  Warning: YOLO model not found at '{self.model_path}'. "
                  "YOLO-based segmentation will be unavailable.")
            return
            
        try:
            self.model = YOLO(self.model_path)
            self.model.fuse()
            print(f"✅ YOLO model loaded from '{self.model_path}'")
        except Exception as e:
            print(f"⚠️  Warning: could not load YOLO model at '{self.model_path}': {e}\n"
                  "         YOLO-based segmentation will be unavailable.")
    
    def is_available(self) -> bool:
        """Check if YOLO model is available for inference."""
        return self.model is not None
    
    def segment(self, input_image: np.ndarray, *, conf_thres: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Perform YOLO-based segmentation.
        
        Args:
            input_image: Input image array
            conf_thres: Confidence threshold for detection
            
        Returns:
            Tuple of (instance_labels, binary_mask)
        """
        if not self.is_available():
            raise RuntimeError("YOLO model not loaded; cannot run YOLO segmentation.")
        
        # Prepare 3-channel uint8 input
        img8 = np.clip(input_image, 0, 255).astype(np.uint8)
        if img8.ndim == 2:
            img8 = np.stack([img8] * 3, axis=-1)
        
        # Run inference
        results = self.model(
            img8, 
            imgsz=768, 
            mask_ratio=1, 
            conf=conf_thres, 
            retina_masks=True, 
            verbose=False
        )[0]
        
        masks = results.masks.data  # Tensor (N, H, W)
        h, w = input_image.shape
        
        # Union mask and instance labels
        union_mask = np.zeros((h, w), dtype=bool)
        instance_labels = np.zeros((h, w), dtype=np.uint16)
        
        for i, mask in enumerate(masks):
            # Resize mask to match input image
            mask_np = mask.cpu().numpy()
            if mask_np.shape != (h, w):
                import cv2
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask_np > 0.5
            union_mask |= mask_bool
            instance_labels[mask_bool] = i + 1
        
        return instance_labels, union_mask


# Global instance for backward compatibility
_yolo_segmentation = None

def get_yolo_segmentation() -> YOLOSegmentation:
    """Get the global YOLO segmentation instance."""
    global _yolo_segmentation
    if _yolo_segmentation is None:
        _yolo_segmentation = YOLOSegmentation()
    return _yolo_segmentation


def segmentation_pipeline_yolo(input_image: np.ndarray, *, conf_thres: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """YOLO-v8 retina_masks segmentation → union mask + instance labels.
    
    Args:
        input_image: Input image array
        conf_thres: Confidence threshold for detection
        
    Returns:
        Tuple of (instance_labels, binary_mask)
    """
    yolo_seg = get_yolo_segmentation()
    return yolo_seg.segment(input_image, conf_thres=conf_thres)