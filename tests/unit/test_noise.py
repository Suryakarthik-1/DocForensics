import numpy as np

from detectors.noise import (NoiseDetector, estimate_local_noise,
                             find_noise_anomalies)


def test_noise_map_shape_matches_image(clean_image):
    nmap = estimate_local_noise(clean_image)
    assert nmap.shape == clean_image.shape[:2]


def test_noise_map_is_normalised(clean_image):
    nmap = estimate_local_noise(clean_image)
    assert nmap.min() >= 0.0
    assert nmap.max() <= 1.0
    assert not np.isnan(nmap).any()


def test_find_anomalies_returns_score_and_regions(clean_image):
    nmap = estimate_local_noise(clean_image)
    score, regions = find_noise_anomalies(nmap)
    assert 0.0 <= score <= 1.0
    assert isinstance(regions, list)


def test_uniform_noise_map_yields_zero_score():
    flat = np.full((64, 64), 0.5, dtype=np.float32)
    score, regions = find_noise_anomalies(flat)
    assert score == 0.0
    assert regions == []


def test_detector_runs_and_reports(clean_image, base_context):
    det = NoiseDetector().run(clean_image, base_context)
    assert det.detector_name == 'noise'
    assert 0.0 <= det.score <= 1.0
    assert det.heatmap.shape == clean_image.shape[:2]
    assert 'mean_noise' in det.details
