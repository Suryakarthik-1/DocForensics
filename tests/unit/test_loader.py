import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ingestion.loader import is_supported, load_document, to_normalized


def make_temp_jpg():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 200
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, img)
        return f.name


def test_is_supported_accepts_jpg():
    path = make_temp_jpg()
    assert is_supported(path)
    Path(path).unlink()


def test_is_supported_rejects_exe():
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        f.write(b'MZ')
        path = f.name
    assert not is_supported(path)
    Path(path).unlink()


def test_to_normalized_converts_uint8():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = to_normalized(img)
    assert out.dtype == np.float32
    assert out.max() <= 1.0


def test_to_normalized_grayscale_becomes_3_channel():
    img = np.ones((10, 10), dtype=np.float32) * 0.5
    out = to_normalized(img)
    assert out.shape == (10, 10, 3)


def test_load_document_returns_page():
    path = make_temp_jpg()
    pages = load_document(path)
    assert len(pages) == 1
    assert pages[0].image.dtype == np.float32
    Path(path).unlink()


def test_load_document_raises_on_missing():
    with pytest.raises(FileNotFoundError):
        load_document('/nonexistent/file.jpg')


def test_load_document_raises_on_unsupported():
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        f.write(b'MZ')
        path = f.name
    with pytest.raises(ValueError):
        load_document(path)
    Path(path).unlink()