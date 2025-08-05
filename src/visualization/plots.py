"""
Plotting utilities for visualizing images and analysis results.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple


def show_image(image: np.ndarray, title: str = "", figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Display a single image.
    
    Args:
        image: Input image array
        title: Image title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_images_grid(images: List[np.ndarray], titles: Optional[List[str]] = None, 
                     cols: int = 3, figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Display multiple images in a grid.
    
    Args:
        images: List of image arrays
        titles: Optional list of titles
        cols: Number of columns in grid
        figsize: Figure size tuple
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        ax = axes[i] if rows > 1 or cols > 1 else axes
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_histogram(image: np.ndarray, bins: int = 256, title: str = "Image Histogram") -> None:
    """
    Plot histogram of image pixel values.
    
    Args:
        image: Input image array
        bins: Number of histogram bins
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    if len(image.shape) == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            plt.hist(image[:, :, i].ravel(), bins=bins, alpha=0.7, label=color, color=color)
        plt.legend()
    else:
        plt.hist(image.ravel(), bins=bins, alpha=0.7, color='gray')
    
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
