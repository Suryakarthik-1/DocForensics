import numpy as np

from ingestion.preprocess import resize_keep_ratio, deskew


def test_small_image_is_unchanged():
    img = np.zeros((100, 80, 3), dtype=np.float32)
    out = resize_keep_ratio(img, max_side=2000)
    assert out.shape == img.shape


def test_large_image_is_downscaled_keeping_ratio():
    img = np.zeros((4000, 2000, 3), dtype=np.float32)
    out = resize_keep_ratio(img, max_side=1000)
    assert max(out.shape[:2]) == 1000
    # aspect ratio preserved (within rounding)
    assert abs(out.shape[0] / out.shape[1] - 2.0) < 0.05


def test_deskew_returns_image_and_angle(clean_image):
    out, angle = deskew(clean_image)
    assert out.shape == clean_image.shape
    assert isinstance(angle, float)


def test_deskew_output_is_normalised(clean_image):
    out, _ = deskew(clean_image)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
