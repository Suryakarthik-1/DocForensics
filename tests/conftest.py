import numpy as np
import pytest

from core.types import BBox, TextRegion
from detectors.base import Context


@pytest.fixture
def clean_image():
    rng = np.random.default_rng(42)
    img = rng.uniform(0.8, 1.0, (128, 128, 3)).astype(np.float32)
    return img


@pytest.fixture
def tampered_image():
    rng = np.random.default_rng(42)
    img = rng.uniform(0.8, 1.0, (128, 128, 3)).astype(np.float32)
    img[50:80, 50:80] = 0.1    # dark patch = simulated edit
    return img


@pytest.fixture
def sample_regions():
    return [
        TextRegion(bbox=BBox(10, 10, 60, 20), text='Invoice', confidence=0.95),
        TextRegion(bbox=BBox(10, 40, 60, 20), text='Total',   confidence=0.92),
        TextRegion(bbox=BBox(10, 70, 40, 20), text='1000',    confidence=0.90),
    ]


@pytest.fixture
def base_context(clean_image):
    return Context(file_path='test.jpg', text_regions=[], original_img=clean_image)