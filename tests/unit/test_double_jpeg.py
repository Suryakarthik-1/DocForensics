import numpy as np

from detectors.double_jpeg import (DoubleJpegDetector, dct_histogram,
                                   detect_double_compression)


def test_histogram_has_expected_length(clean_image):
    from core.config import DCT_HIST_BINS
    hist = dct_histogram(clean_image)
    assert hist.shape == (DCT_HIST_BINS,)


def test_score_in_range(clean_image):
    hist = dct_histogram(clean_image)
    score, regions = detect_double_compression(hist)
    assert 0.0 <= score <= 1.0
    assert regions == []


def test_empty_histogram_is_zero():
    score, _ = detect_double_compression(np.zeros(64, dtype=np.float32))
    assert score == 0.0


def test_score_is_capped_conservatively(clean_image):
    # double-JPEG is a weak hint on documents and must never read as certain
    hist = dct_histogram(clean_image)
    score, _ = detect_double_compression(hist)
    assert score <= 0.6


def test_detector_runs(clean_image, base_context):
    det = DoubleJpegDetector().run(clean_image, base_context)
    assert det.detector_name == 'double_jpeg'
    assert det.heatmap is None
    assert 0.0 <= det.score <= 1.0
