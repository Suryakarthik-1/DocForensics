import numpy as np
import pytest

from detectors.ela import ELADetector, compute_ela, ela_score


def test_ela_output_is_float32():
    img = np.ones((64, 64, 3), dtype=np.float32) * 0.9
    result = compute_ela(img)
    assert result.dtype == np.float32


def test_ela_shape_matches_input():
    img = np.ones((64, 64, 3), dtype=np.float32) * 0.9
    result = compute_ela(img)
    assert result.shape == (64, 64)


def test_ela_values_between_0_and_1():
    img = np.ones((64, 64, 3), dtype=np.float32) * 0.9
    result = compute_ela(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_ela_detector_returns_correct_name(base_context, clean_image):
    det = ELADetector().run(clean_image, base_context)
    assert det.detector_name == 'ela'


def test_ela_detector_score_in_range(base_context, clean_image):
    det = ELADetector().run(clean_image, base_context)
    assert 0.0 <= det.score <= 1.0


def test_ela_heatmap_shape_matches_image(base_context, clean_image):
    det = ELADetector().run(clean_image, base_context)
    assert det.heatmap.shape == clean_image.shape[:2]