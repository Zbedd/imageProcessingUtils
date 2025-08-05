"""
Pytest configuration and fixtures for imageProcessingUtils tests
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path


@pytest.fixture
def sample_data_dir():
    """Path to sample test data directory"""
    return Path(__file__).parent / "data"
