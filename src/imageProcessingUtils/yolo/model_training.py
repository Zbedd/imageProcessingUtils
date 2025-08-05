"""YOLO model training utilities for nuclei segmentation.

This module handles downloading the DSB-2018 dataset, building YOLO-format datasets,
and training YOLOv8 models for nuclei segmentation.
"""
from __future__ import annotations
import os, zipfile, shutil, random, pathlib, sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Import preprocessing from within the utils package
from ..image_processing.preprocessing import preprocess_dapi

# Remove the tunel_quant import entirely
DEFAULTS = {
    "kaggle_config_dir": None  # Default fallback
}

# Import Kaggle API only when needed
_kaggle_api = None

def _get_kaggle_api():
    """Get Kaggle API instance (lazy import)."""
    global _kaggle_api
    if _kaggle_api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        _kaggle_api = KaggleApi()
        _kaggle_api.authenticate()
    return _kaggle_api

# ─── project paths ──────────────────────────────────────────────────────────
YOLO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = YOLO_ROOT / "data"
RAW_DIR = DATA_DIR / "dsb18" / "train_raw"
YOLO_DIR = DATA_DIR / "nuclei_yolo"
DATA_YAML = YOLO_DIR / "data.yaml"

def download_dsb() -> None:
    """Download DSB-2018 dataset from Kaggle."""
    try:
        api = _get_kaggle_api()
        dataset_name = "dsb2018"
        download_path = str(DATA_DIR)
        
        print(f"📥 Downloading DSB-2018 dataset to {download_path}...")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        # Move files to expected structure
        downloaded_path = DATA_DIR / "dsb2018"
        if downloaded_path.exists():
            print("✅ DSB-2018 dataset downloaded successfully")
        else:
            print("⚠️ Dataset download may have failed - check the path")
            
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("💡 Make sure you have Kaggle API configured and dataset access")

def masks_to_poly(masks_dir: Path, output_file: Path) -> None:
    """Convert mask images to YOLO polygon format."""
    polygons = []
    
    if not masks_dir.exists():
        # Create empty label file if no masks directory
        with open(output_file, "w") as f:
            pass
        return
    
    for mask_file in masks_dir.glob("*.png"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:  # Need at least 3 points for a polygon
                continue
                
            # Normalize coordinates to [0, 1]
            h, w = mask.shape
            normalized_contour = []
            
            for point in contour:
                x, y = point[0]
                normalized_contour.extend([x/w, y/h])
            
            # Format: class_id x1 y1 x2 y2 ... (class 0 for nucleus)
            if len(normalized_contour) >= 6:  # At least 3 points
                polygon_line = "0 " + " ".join(f"{coord:.6f}" for coord in normalized_contour)
                polygons.append(polygon_line)
    
    # Write polygons to file
    with open(output_file, "w") as f:
        for polygon in polygons:
            f.write(polygon + "\n")

# Make preprocessing configurable
def build_dataset(
    train_blur_ratio: float = 0.7, 
    preprocess_func=None,
    kaggle_config_dir: str = None
) -> None:
    """
    Create YOLO‑seg folder‑tree and YAML. Safe to call repeatedly.
    
    Args:
        train_blur_ratio: Fraction of training images to preprocess
        preprocess_func: Custom preprocessing function (defaults to preprocess_dapi)
        kaggle_config_dir: Custom Kaggle config directory
    """
    if not RAW_DIR.exists():
        if kaggle_config_dir:
            os.environ["KAGGLE_CONFIG_DIR"] = str(pathlib.Path(kaggle_config_dir))
        download_dsb()
    
    # Use provided preprocessing function or default
    if preprocess_func is None:
        preprocess_func = preprocess_dapi

    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    train_ids = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
    val_ids = {train_ids[i] for i in range(0, len(train_ids), 10)}  # 10 % val

    for sid in train_ids:
        split = "val" if sid in val_ids else "train"
        img_src = RAW_DIR / sid / "images" / f"{sid}.png"
        img_dst = YOLO_DIR / "images" / split / f"{sid}.png"
        lbl_dst = YOLO_DIR / "labels" / split / f"{sid}.txt"

        img = cv2.imread(str(img_src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("⚠️  skip", img_src)
            continue

        if split == "train" and random.random() < train_blur_ratio:
            img = preprocess_func(img)  # Use configurable preprocessing

        cv2.imwrite(str(img_dst), img)
        masks_to_poly(RAW_DIR / sid / "masks", lbl_dst)

    with open(DATA_YAML, "w") as f:
        f.write(
            f"path: {YOLO_DIR.resolve()}\n"
            "train: images/train\n"
            "val:   images/val\n"
            "nc: 1\n"
            "names: ['nucleus']\n"
        )
    print("✅ YOLO‑seg dataset ready →", YOLO_DIR)

# ─── model training ─────────────────────────────────────────────────────────

def train_yolov8(
    epochs: int = 50,
    imgsz: int = 768,
    batch: int = 8,
    device: int | str = 0,
) -> None:
    """Fine‑tune `yolov8m‑seg.pt` on the prepared dataset."""
    build_dataset()

    print("▶️  fine‑tuning yolov8m‑seg …")
    model = YOLO("yolov8m-seg.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        mask_ratio=2,
        retina_masks=True,
        device=device,
    )
    print("✅ training complete.  Checkpoints under runs/segment/")

# Optional: allow this file to be run directly for quick tests
if __name__ == "__main__":
    train_yolov8()
