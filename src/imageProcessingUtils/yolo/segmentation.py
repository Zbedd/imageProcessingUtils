"""YOLO-based segmentation functionality for nuclei detection."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

# Import DEFAULTS from model_training
from .model_training import DEFAULTS


class YOLOSegmentation:
    """YOLO-based nuclei segmentation handler for microscopy images.
    
    This class provides an interface to YOLO (You Only Look Once) segmentation models
    specifically fine-tuned for nuclei detection in microscopy images. It handles
    model loading, input preprocessing, and output post-processing.
    
    The class automatically discovers and loads the best available model:
    1. Fine-tuned models in models/ directory (prefix: best_*)
    2. Base models in models/ directory  
    3. Training outputs in runs/segment/ directory
    
    Attributes:
        model: Loaded YOLO model instance (None if unavailable)
        model_path: Path to the loaded model file
        
    Example:
        >>> segmenter = YOLOSegmentation()
        >>> if segmenter.is_available():
        ...     labels, mask = segmenter.segment(microscopy_image, conf_thres=0.05)
        ...     print(f"Detected {labels.max()} nuclei")
    """
    
    def __init__(self, model_path: Path | str | None = None):
        """Initialize YOLO segmentation with automatic or specified model discovery.
        
        Args:
            model_path: Optional path to specific YOLO model file (.pt)
                       If None, automatically discovers the best available model:
                       - Prioritizes fine-tuned models (best_*.pt) in models/ directory
                       - Falls back to base models in models/ directory  
                       - Finally checks runs/segment/train*/weights/best.pt
        """
        print(f"Debug: YOLOSegmentation init called with model_path: {model_path}")
        print(f"Debug: Current working directory: {Path.cwd()}")
        print(f"Debug: Module file location: {Path(__file__)}")
        print(f"Debug: Module parent directory: {Path(__file__).parent}")
        
        self.model = None
        self.model_path = None
        
        if model_path is None:
            # Auto-discover the latest trained model
            self.model_path = self._find_latest_model()
            print(f"Debug: Auto-discovered model_path: {self.model_path}")
        else:
            self.model_path = Path(model_path)
            print(f"Debug: Using provided model_path: {self.model_path}")
            
        print(f"Debug: Final model_path before loading: {self.model_path}")
        if self.model_path and not self.model_path.exists():
            print(f"Debug: Model file does not exist: {self.model_path}")
            self.model_path = None
            
        self._load_model()
    
    def _find_latest_model(self) -> Path | None:
        """Find the most recently trained YOLO model."""
        # First check for models in the models directory (preferred)
        models_dir = Path(__file__).parent / "models"
        print(f"Debug: Checking models directory: {models_dir}")
        print(f"Debug: Models directory exists: {models_dir.exists()}")
        
        if models_dir.exists():
            try:
                # Prioritize fine-tuned models (those starting with "best_")
                fine_tuned_models = list(models_dir.glob("best_*.pt"))
                print(f"Debug: Found {len(fine_tuned_models)} fine-tuned models")
                if fine_tuned_models:
                    model_path = max(fine_tuned_models, key=lambda p: p.stat().st_mtime)
                    print(f"Found fine-tuned model in models directory: {model_path}")
                    return model_path
                
                # Fall back to any .pt model if no fine-tuned models
                all_models = list(models_dir.glob("*.pt"))
                print(f"Debug: Found {len(all_models)} total .pt models")
                if all_models:
                    model_path = max(all_models, key=lambda p: p.stat().st_mtime)
                    print(f"Found model in models directory: {model_path}")
                    return model_path
            except ValueError as e:
                print(f"Debug: ValueError in models directory search: {e}")
                pass  # No models in models directory, continue to runs/
            except Exception as e:
                print(f"Debug: Unexpected error in models directory search: {e}")
                pass
        
        # Fall back to runs directory within the yolo module
        runs_dir = Path(__file__).parent / "runs" / "segment"
        print(f"Debug: Checking runs directory: {runs_dir}")
        print(f"Debug: Runs directory exists: {runs_dir.exists()}")
        
        try:
            # Find the most recent best.pt model
            training_models = list(runs_dir.glob("train*/weights/best.pt"))
            print(f"Debug: Found {len(training_models)} training models")
            if training_models:
                model_path = max(training_models, key=lambda p: p.stat().st_mtime)
                print(f"Found model in runs directory: {model_path}")
                return model_path
        except ValueError as e:
            print(f"Debug: ValueError in runs directory search: {e}")
            # No models found
            pass
        except Exception as e:
            print(f"Debug: Unexpected error in runs directory search: {e}")
            pass
            
        print("Debug: No models found in any location")
        return None
    
    def _load_model(self):
        """Load the YOLO model with proper device handling."""
        if self.model_path is None or not self.model_path.exists():
            print(f"Warning: YOLO model not found at '{self.model_path}'. "
                  "YOLO-based segmentation will be unavailable.")
            return
            
        try:
            # Clear any existing model first to avoid state issues
            self.model = None
            
            # Import torch to check device availability
            try:
                import torch
                # Use CUDA if available, otherwise CPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"Debug: Using device: {device}")
            except ImportError:
                # Fallback if torch not available
                device = 'cpu'
                print("Debug: PyTorch not available, defaulting to CPU")
            
            # Load model with explicit device specification
            self.model = YOLO(self.model_path)
            
            # Move model to appropriate device
            if hasattr(self.model, 'to'):
                self.model.to(device)
            elif hasattr(self.model.model, 'to'):
                self.model.model.to(device)
                
            self.model.fuse()
            print(f"YOLO model loaded from '{self.model_path}' on device: {device}")
            
        except Exception as e:
            print(f"Warning: could not load YOLO model at '{self.model_path}': {e}\n"
                  f"         Device: {device if 'device' in locals() else 'unknown'}")
            print(f"         Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.model = None  # Ensure model is None on failure
    
    def is_available(self) -> bool:
        """Check if YOLO model is available for inference."""
        return self.model is not None
    
    def reload_model(self):
        """Reload the YOLO model. Useful if model becomes corrupted or unavailable."""
        print("Reloading YOLO model...")
        self._load_model()
        
    def segment(self, input_image: np.ndarray, *, conf_thres: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Perform YOLO-based segmentation on nuclei/cellular structures.
        
        This method processes grayscale microscopy images for nuclei detection using a 
        fine-tuned YOLO segmentation model. The input image is automatically converted 
        to the required 3-channel RGB format that YOLO expects.
        
        Args:
            input_image: 2D grayscale numpy array (uint8 or float, any intensity range)
                        Expected formats:
                        - Microscopy images: nuclei appear as bright regions on dark background
                        - Any bit depth: will be normalized to uint8 [0-255] range
                        - Single channel: automatically converted to 3-channel RGB for YOLO
            conf_thres: Confidence threshold for object detection (0.0-1.0)
                       Lower values detect more objects but may include false positives
                       Typical range: 0.01-0.5 for nuclei detection
            
        Returns:
            Tuple of (instance_labels, binary_mask):
            - instance_labels: 2D uint16 array where each detected nucleus has unique ID (1, 2, 3...)
            - binary_mask: 2D boolean array indicating all detected regions (True = nucleus)
            
        Raises:
            RuntimeError: If YOLO model is not loaded or available
            
        Note:
            The YOLO model expects 3-channel input at 768x768 resolution internally.
            Input images are automatically resized and converted during inference.
        """
        if not self.is_available():
            print(f"Debug: YOLO model not available. Model path: {self.model_path}, Model loaded: {self.model is not None}")
            if self.model_path is None:
                print("Debug: Model path is None - model discovery failed")
            elif not self.model_path.exists():
                print(f"Debug: Model file does not exist: {self.model_path}")
            else:
                print("Debug: Model file exists but failed to load")
                
            raise RuntimeError(
                f"YOLO model not loaded; cannot run YOLO segmentation. "
                f"Model path: {self.model_path}. "
                f"Current working directory: {Path.cwd()}. "
                f"Module location: {Path(__file__).parent}"
            )
        
        try:
            # Prepare 3-channel uint8 input
            img8 = np.clip(input_image, 0, 255).astype(np.uint8)
            if img8.ndim == 2:
                img8 = np.stack([img8] * 3, axis=-1)
            
            # Run inference with device consistency check
            try:
                results = self.model(
                    img8, 
                    imgsz=768, 
                    mask_ratio=1, 
                    conf=conf_thres, 
                    retina_masks=True, 
                    verbose=False
                )[0]
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e) or "CUDA" in str(e):
                    print(f"Device mismatch detected: {e}")
                    print("Attempting to reload model with proper device handling...")
                    self.reload_model()
                    if self.is_available():
                        results = self.model(
                            img8, 
                            imgsz=768, 
                            mask_ratio=1, 
                            conf=conf_thres, 
                            retina_masks=True, 
                            verbose=False
                        )[0]
                    else:
                        raise e
                else:
                    raise e
            
        except Exception as e:
            print(f"Warning: YOLO inference failed: {e}")
            print("Attempting to reload model and retry...")
            
            # Try to reload the model once
            self.reload_model()
            
            if not self.is_available():
                raise RuntimeError(f"YOLO model failed and could not be reloaded: {e}")
            
            # Retry inference once
            try:
                results = self.model(
                    img8, 
                    imgsz=768, 
                    mask_ratio=1, 
                    conf=conf_thres, 
                    retina_masks=True, 
                    verbose=False
                )[0]
            except Exception as e2:
                raise RuntimeError(f"YOLO inference failed even after model reload: {e2}")
        
        # Check if any masks were detected
        if results.masks is None:
            # No objects detected, return empty masks
            h, w = input_image.shape
            return np.zeros((h, w), dtype=np.uint16), np.zeros((h, w), dtype=bool)
        
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
    """YOLO-v8 nuclei segmentation pipeline with retina masks.
    
    High-level interface for YOLO-based nuclei segmentation. This function provides
    a simple way to segment nuclei in microscopy images using a fine-tuned YOLO model.
    
    Args:
        input_image: 2D grayscale microscopy image as numpy array
                    - Supported formats: uint8, uint16, float32, float64
                    - Expected content: nuclei as bright regions on darker background
                    - Any resolution: automatically processed by YOLO at optimal size
        conf_thres: Detection confidence threshold (default: 0.01)
                   - Range: 0.0 to 1.0
                   - Lower values: more detections, possible false positives
                   - Higher values: fewer detections, missed small nuclei
                   - Recommended: 0.01-0.1 for nuclei, 0.1-0.3 for larger objects
        
    Returns:
        Tuple of (instance_labels, binary_mask):
        - instance_labels: 2D uint16 array with unique IDs for each nucleus (0=background)
        - binary_mask: 2D boolean array marking all detected nuclear regions
        
    Example:
        >>> import numpy as np
        >>> from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo
        >>> 
        >>> # Load your microscopy image (2D grayscale)
        >>> image = np.load('microscopy_image.npy')  # or cv2.imread(), etc.
        >>> 
        >>> # Segment nuclei
        >>> labels, mask = segmentation_pipeline_yolo(image, conf_thres=0.05)
        >>> 
        >>> # Results
        >>> num_nuclei = labels.max()  # Number of detected nuclei
        >>> coverage = mask.sum() / mask.size  # Fraction of image covered by nuclei
        
    Note:
        Uses a globally cached YOLO model instance for efficiency. The model is
        automatically loaded on first use and searches for fine-tuned models first.
    """
    yolo_seg = get_yolo_segmentation()
    return yolo_seg.segment(input_image, conf_thres=conf_thres)