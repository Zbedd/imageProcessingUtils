"""
BBBC022 Dataset Fetcher

This module provides tools for downloading and processing images from the BBBC022 dataset 
(Human U2OS cells - Out of focus fluorescence microscopy) for use in image processing pipelines.

BBBC022 contains human U2OS cells stained with Hoechst 33342 (nuclei) captured at different z-planes.
Dataset URL: https://bbbc.broadinstitute.org/BBBC022
"""

import os
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import json


class BBBC022Fetcher:
    """Fetcher for BBBC022 dataset with filtering and sampling capabilities.
    
    This class provides methods to download, filter, and sample images from the
    BBBC022 dataset in a format compatible with YOLO segmentation pipelines.
    
    Example:
        ```python
        from imageProcessingUtils.sample_data import BBBC022Fetcher
        
        fetcher = BBBC022Fetcher('./data')
        images, metadata = fetcher.fetch_samples(
            count=10,
            focal_planes=[0],     # In-focus only
            treatments=['DMSO'],  # Control condition
            seed=42               # Reproducible sampling
        )
        ```
    """
    
    # BBBC022 dataset information
    BASE_URL = "https://bbbc.broadinstitute.org/BBBC022"
    DATASET_BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC022"
    URLS_LIST_URL = "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_urls.txt"
    METADATA_URL = "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv"
    
    # Available treatments in BBBC022 (Cell Painting experiment with bioactive compounds)
    # Note: This dataset contains ~1600 bioactive compounds, not specific named treatments
    # The actual compounds are identified by BROAD_ID in the metadata
    AVAILABLE_ROLES = ["compound", "mock"]  # ASSAY_WELL_ROLE values
    
    # Available channels in BBBC022 Cell Painting
    AVAILABLE_CHANNELS = [
        "OrigHoechst",     # Hoechst 33342 (nuclei)
        "OrigER",          # con A (endoplasmic reticulum)
        "OrigSyto",        # SYTO 14 (cytoplasmic RNA)
        "OrigMito",        # MitoTracker Deep Red (mitochondria)
        "OrigPh_golgi"     # WGA + phalloidin (Golgi + F-actin)
    ]
    
    # Note: BBBC022 doesn't have focal planes like the original assumption
    # It's a Cell Painting experiment with different channels/organelles
    FOCAL_PLANES = [0]  # Single focal plane per field
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the fetcher.
        
        Args:
            output_dir: Directory to store downloaded and processed images.
                       If None, uses ~/.imageProcessingUtils/bbbc022_data
        """
        if output_dir is None:
            # Use user's home directory for data storage by default
            import os
            home_dir = Path.home()
            self.output_dir = home_dir / ".imageProcessingUtils" / "bbbc022_data"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed" 
        self.metadata_file = self.output_dir / "metadata.json"
        
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
    def download_dataset(self, 
                        force_redownload: bool = False,
                        max_plates: int = 2,
                        channels: list = None) -> Path:
        """Download the BBBC022 dataset (subset of plates/channels).
        
        Note: The full BBBC022 dataset is ~100GB (100 zip files). This method
        downloads only a subset for practical use.
        
        Args:
            force_redownload: If True, redownload even if files exist
            max_plates: Maximum number of plates to download (default: 2)
            channels: List of channels to download (default: ["OrigHoechst"] - nuclei only)
            
        Returns:
            Path to the extracted dataset directory
            
        Raises:
            Exception: If download fails
        """
        if channels is None:
            channels = ["OrigHoechst"]  # Nuclei channel only by default
            
        extract_path = self.raw_dir / "BBBC022_images"
        urls_file = self.raw_dir / "BBBC022_v1_images_urls.txt"
        
        # Download the URLs list file first
        if not urls_file.exists() or force_redownload:
            print(f"Downloading BBBC022 URLs list from {self.URLS_LIST_URL}")
            try:
                urllib.request.urlretrieve(self.URLS_LIST_URL, urls_file)
                print(f"Downloaded URLs list to {urls_file}")
            except Exception as e:
                print(f"Error downloading URLs list: {e}")
                raise
        
        # Read and parse the URLs
        with open(urls_file, 'r') as f:
            all_urls = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(all_urls)} zip files in dataset")
        
        # Filter URLs based on criteria
        selected_urls = []
        plates_seen = set()
        
        for url in all_urls:
            filename = url.split('/')[-1]
            
            # Parse BBBC022 filename format: BBBC022_v1_images_20585w1.zip
            # Where 20585 is plate ID, w1 is channel (w1=OrigHoechst, etc.)
            try:
                # Extract channel info
                channel_match = None
                if 'w1.zip' in filename:
                    channel_match = 'OrigHoechst'
                elif 'w2.zip' in filename:
                    channel_match = 'OrigER'
                elif 'w3.zip' in filename:
                    channel_match = 'OrigSyto'
                elif 'w4.zip' in filename:
                    channel_match = 'OrigMito'
                elif 'w5.zip' in filename:
                    channel_match = 'OrigPh_golgi'
                
                # Check if this channel is requested
                if channel_match and channel_match in channels:
                    # Extract plate ID (5 digits before 'w')
                    w_pos = filename.find('w')
                    if w_pos > 0:
                        # Look for plate number pattern
                        plate_part = filename[:w_pos]
                        # Extract the last sequence of digits
                        import re
                        plate_nums = re.findall(r'\d+', plate_part)
                        if plate_nums:
                            plate_id = plate_nums[-1]  # Take the last number found
                            
                            # Add if we haven't seen this plate or haven't reached limit
                            if len(plates_seen) < max_plates:
                                if plate_id not in plates_seen or len(selected_urls) < max_plates * len(channels):
                                    selected_urls.append(url)
                                    plates_seen.add(plate_id)
                                    
            except Exception as e:
                print(f"Warning: Could not parse filename {filename}: {e}")
                continue
                    
        print(f"Selected {len(selected_urls)} zip files to download (plates: {len(plates_seen)}, channels: {channels})")
        
        # If no URLs were selected but we have channels, take first few that match
        if not selected_urls and channels:
            print("Fallback: Taking first few URLs that match requested channels...")
            for url in all_urls[:max_plates * len(channels) * 2]:  # Check more URLs
                filename = url.split('/')[-1]
                for channel in channels:
                    if ('w1' in filename and channel == 'OrigHoechst') or \
                       ('w2' in filename and channel == 'OrigER') or \
                       ('w3' in filename and channel == 'OrigSyto') or \
                       ('w4' in filename and channel == 'OrigMito') or \
                       ('w5' in filename and channel == 'OrigPh_golgi'):
                        selected_urls.append(url)
                        if len(selected_urls) >= max_plates * len(channels):
                            break
                if len(selected_urls) >= max_plates * len(channels):
                    break
        
        print(f"Selected {len(selected_urls)} zip files to download (plates: {len(plates_seen)}, channels: {channels})")
        
        # Download and extract selected files
        extract_path.mkdir(exist_ok=True)
        
        for i, url in enumerate(selected_urls):
            filename = url.split('/')[-1]
            zip_path = self.raw_dir / filename
            
            # Download if needed
            if not zip_path.exists() or force_redownload:
                print(f"Downloading {i+1}/{len(selected_urls)}: {filename}")
                try:
                    urllib.request.urlretrieve(url, zip_path)
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    continue
            
            # Extract if needed
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Extracted {filename}")
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                continue
                
        print(f"Download and extraction complete. Images available in: {extract_path}")
        return extract_path
    
    def parse_filename(self, filename: str) -> dict:
        """Parse BBBC022 filename to extract metadata.
        
        BBBC022 filenames follow Cell Painting format with plate, well, site, and channel info.
        Files are typically named with patterns like: AS_09125_050116180001_A01f00d0.tif
        
        Args:
            filename: Image filename
            
        Returns:
            Dictionary with parsed metadata
        """
        try:
            # Remove extension
            base_name = filename.replace('.tif', '').replace('.png', '')
            
            # BBBC022 uses a different naming convention than originally assumed
            # Try to extract basic info from the filename
            metadata = {
                'filename': filename,
                'base_name': base_name,
                'channel': 'unknown',
                'well': 'unknown',
                'site': 'unknown',
                'plate': 'unknown',
                'is_nuclei': False  # Will be determined by channel
            }
            
            # Try to identify channel from common patterns
            filename_lower = filename.lower()
            if 'hoechst' in filename_lower or 'w1' in filename_lower:
                metadata['channel'] = 'OrigHoechst'
                metadata['is_nuclei'] = True
            elif 'er' in filename_lower or 'w2' in filename_lower:
                metadata['channel'] = 'OrigER'
            elif 'syto' in filename_lower or 'w3' in filename_lower:
                metadata['channel'] = 'OrigSyto'
            elif 'mito' in filename_lower or 'w4' in filename_lower:
                metadata['channel'] = 'OrigMito'
            elif 'golgi' in filename_lower or 'ph_golgi' in filename_lower or 'w5' in filename_lower:
                metadata['channel'] = 'OrigPh_golgi'
            
            # Try to extract well info (pattern like A01, B12, etc.)
            import re
            well_match = re.search(r'[A-P][0-9]{2}', base_name)
            if well_match:
                metadata['well'] = well_match.group()
            
            # Try to extract site info (pattern like f00, s1, etc.)
            site_match = re.search(r'[fs][0-9]+', base_name)
            if site_match:
                metadata['site'] = site_match.group()
            
            return metadata
            
        except Exception:
            # If parsing fails, return basic info
            return {
                'filename': filename,
                'channel': 'unknown',
                'well': 'unknown', 
                'site': 'unknown',
                'plate': 'unknown',
                'is_nuclei': 'hoechst' in filename.lower() or 'w1' in filename.lower()
            }
    
    def filter_images(self, 
                     image_dir: Path,
                     channels: Optional[List[str]] = None,
                     wells: Optional[List[str]] = None,
                     nuclei_only: bool = True) -> List[dict]:
        """Filter images based on criteria.
        
        Args:
            image_dir: Directory containing BBBC022 images
            channels: List of channels to include (None = all)
            wells: List of wells to include (None = all)
            nuclei_only: If True, only include nuclei channel (OrigHoechst)
            
        Returns:
            List of metadata dictionaries for filtered images
        """
        filtered_images = []
        
        # Get all image files recursively (TIFF and PNG)
        image_files = []
        # Search recursively in subdirectories
        for pattern in ["**/*.tif", "**/*.png"]:
            image_files.extend(image_dir.glob(pattern))
        
        print(f"Found {len(image_files)} total images in {image_dir}")
        
        for image_file in image_files:
            metadata = self.parse_filename(image_file.name)
            
            if not metadata:
                continue
                
            # Apply filters
            if nuclei_only and not metadata['is_nuclei']:
                continue
                
            if channels and metadata['channel'] not in channels:
                continue
                
            if wells and metadata['well'] not in wells:
                continue
                
            metadata['filepath'] = image_file
            filtered_images.append(metadata)
        
        print(f"After filtering: {len(filtered_images)} images")
        return filtered_images
    
    def sample_images(self, 
                     filtered_images: List[dict], 
                     count: int, 
                     seed: Optional[int] = None) -> List[dict]:
        """Randomly sample images from filtered list.
        
        Args:
            filtered_images: List of filtered image metadata
            count: Number of images to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled image metadata
        """
        if seed is not None:
            random.seed(seed)
            
        if count >= len(filtered_images):
            print(f"Requested {count} images but only {len(filtered_images)} available. Using all.")
            return filtered_images
            
        sampled = random.sample(filtered_images, count)
        print(f"Sampled {len(sampled)} images (seed: {seed})")
        return sampled
    
    def convert_to_yolo_format(self, image_path: Path) -> np.ndarray:
        """Convert image to format accepted by yolo_segmentation_pipeline.
        
        The YOLO pipeline expects 2D grayscale numpy arrays with uint8 dtype
        in the range [0, 255].
        
        Args:
            image_path: Path to input image
            
        Returns:
            2D numpy array ready for YOLO segmentation
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Ensure uint8 format
        if img_array.dtype != np.uint8:
            # Normalize to 0-255 range
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_normalized = (img_array - img_min) / (img_max - img_min) * 255
            else:
                img_normalized = img_array
            img_array = img_normalized.astype(np.uint8)
            
        return img_array
    
    def fetch_samples(self,
                     count: int = 10,
                     channels: Optional[List[str]] = None,
                     wells: Optional[List[str]] = None,
                     nuclei_only: bool = True,
                     seed: Optional[int] = None,
                     save_processed: bool = True,
                     max_plates: int = 2) -> Tuple[List[np.ndarray], List[dict]]:
        """Main method to fetch and process sample images.
        
        Args:
            count: Number of images to sample
            channels: List of channels to include (e.g., ['OrigHoechst', 'OrigER'])
            wells: List of wells to include (e.g., ['A01', 'B02'])
            nuclei_only: If True, only include nuclei channel (OrigHoechst)
            seed: Random seed for sampling
            save_processed: If True, save processed images to disk
            max_plates: Maximum number of plates to download (affects download time)
            
        Returns:
            Tuple of (image_arrays, metadata_list)
            
        Example:
            ```python
            fetcher = BBBC022Fetcher('./data')
            images, metadata = fetcher.fetch_samples(
                count=5,
                channels=['OrigHoechst'],  # Nuclei channel only
                wells=['A01', 'A02'],      # Specific wells
                seed=42
            )
            ```
        """
        print("BBBC022 Sample Fetcher")
        print("=" * 50)
        
        # Download dataset (subset)
        dataset_dir = self.download_dataset(max_plates=max_plates, channels=channels)
        
        # Filter images
        filtered_images = self.filter_images(
            dataset_dir,
            channels=channels,
            wells=wells,
            nuclei_only=nuclei_only
        )
        
        if not filtered_images:
            print("No images match the specified criteria!")
            return [], []
            
        # Sample images
        sampled_images = self.sample_images(filtered_images, count, seed)
        
        # Process images
        processed_images = []
        processed_metadata = []
        
        print("\nProcessing images...")
        for i, metadata in enumerate(sampled_images):
            print(f"Processing {i+1}/{len(sampled_images)}: {metadata['filename']}")
            
            # Convert to YOLO format
            img_array = self.convert_to_yolo_format(metadata['filepath'])
            processed_images.append(img_array)
            
            # Update metadata
            processed_meta = metadata.copy()
            processed_meta['shape'] = img_array.shape
            processed_meta['dtype'] = str(img_array.dtype)
            processed_meta['processed_filename'] = f"bbbc022_sample_{i:03d}.npy"
            
            # Save processed image if requested
            if save_processed:
                save_path = self.processed_dir / processed_meta['processed_filename']
                np.save(save_path, img_array)
                processed_meta['processed_path'] = str(save_path)
            
            processed_metadata.append(processed_meta)
        
        # Save metadata
        if save_processed:
            with open(self.metadata_file, 'w') as f:
                json.dump(processed_metadata, f, indent=2, default=str)
            print(f"\nMetadata saved to: {self.metadata_file}")
            print(f"Processed images saved to: {self.processed_dir}")
        
        print(f"\nSuccessfully processed {len(processed_images)} images")
        return processed_images, processed_metadata


# Convenience functions for quick access
def fetch_bbbc022_samples(count: int = 10,
                         channels: Optional[List[str]] = None,
                         wells: Optional[List[str]] = None,
                         nuclei_only: bool = True,
                         seed: Optional[int] = None,
                         output_dir: Optional[str] = None,
                         max_plates: int = 2) -> Tuple[List[np.ndarray], List[dict]]:
    """Convenience function to fetch BBBC022 samples with minimal setup.
    
    Args:
        count: Number of images to sample
        channels: List of channels to include (e.g., ['OrigHoechst'])
        wells: List of wells to include
        nuclei_only: If True, only include nuclei channel
        seed: Random seed for sampling
        output_dir: Directory to store downloaded data.
                   If None, uses ~/.imageProcessingUtils/bbbc022_data
        max_plates: Maximum number of plates to download
        
    Returns:
        Tuple of (image_arrays, metadata_list)
        
    Example:
        ```python
        from imageProcessingUtils.sample_data import fetch_bbbc022_samples
        
        # Quick fetch of 5 nuclei images
        images, metadata = fetch_bbbc022_samples(
            count=5,
            channels=['OrigHoechst'],
            seed=42
        )
        ```
    """
    fetcher = BBBC022Fetcher(output_dir)
    return fetcher.fetch_samples(
        count=count,
        channels=channels,
        wells=wells,
        nuclei_only=nuclei_only,
        seed=seed,
        max_plates=max_plates
    )


def get_available_channels() -> List[str]:
    """Get list of available channels in BBBC022 dataset.
    
    Returns:
        List of channel names
    """
    return BBBC022Fetcher.AVAILABLE_CHANNELS.copy()


def get_available_roles() -> List[str]:
    """Get list of available well roles in BBBC022 dataset.
    
    Returns:
        List of roles ('compound', 'mock')
    """
    return BBBC022Fetcher.AVAILABLE_ROLES.copy()


def get_available_focal_planes() -> List[int]:
    """Get list of available focal planes in BBBC022 dataset.
    
    Note: BBBC022 is a single focal plane dataset.
    
    Returns:
        List with single focal plane [0]
    """
    return BBBC022Fetcher.FOCAL_PLANES.copy()


# Legacy function name for compatibility
def get_available_treatments() -> List[str]:
    """Legacy function - BBBC022 uses compound roles, not specific treatments.
    
    Returns:
        List of available roles for backward compatibility
    """
    return get_available_roles()
