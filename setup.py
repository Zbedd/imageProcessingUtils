from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="imageProcessingUtils",
    version="0.1.0",
    author="Zbedd",
    author_email="",
    description="Image Processing and YOLO Utilities Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zbedd/imageProcessingUtils",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (always required)
        "numpy>=1.21.0",
        "opencv-python>=4.5.0", 
        "Pillow>=8.0.0",
        "PyYAML>=5.4.0",
        # Deep learning stack (essential for YOLO)
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "ultralytics>=8.0.0",
        # Scientific computing
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        # Data handling
        "pandas>=1.3.0",
        "h5py>=3.1.0",
        "tqdm>=4.60.0",
        # File I/O
        "imageio>=2.9.0",
        "nd2reader>=3.3.0",
        # CLI support
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
        "gpu": [
            "cupy-cuda11x",  # GPU acceleration
        ],
        "datasets": [
            "kaggle>=1.5.0",  # Kaggle dataset access
        ],
    },
    entry_points={
        "console_scripts": [
            "imageProcessingUtils-train=yolo.model_training:main",
            "imageProcessingUtils-segment=yolo.segmentation:main",
        ],
    },
    include_package_data=True,
)
