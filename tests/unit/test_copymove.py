import numpy as np

from detectors.copymove import CopyMoveDetector, block_hash_features, match_duplicate_blocks


def make_cloned_image():
    img = np.random.default_rng(0).uniform(0, 1, (128, 128, 3)).astype(np.float32)
    img[80:96, 80:96] = img[10:26, 10:26]   # exact clone
    return img


def test_block_hash_returns_dict():
    img = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    features = block_hash_features(img, block=16, stride=8)
    assert isinstance(features, dict)
    assert len(features) > 0


def test_detects_cloned_block():
    img = make_cloned_image()
    features = block_hash_features(img, block=16, stride=8)
    matches = match_duplicate_blocks(features, min_distance=16)
    assert len(matches) >= 1


def test_score_in_range(base_context, clean_image):
    det = CopyMoveDetector().run(clean_image, base_context)
    assert 0.0 <= det.score <= 1.0


def test_heatmap_shape(base_context, clean_image):
    det = CopyMoveDetector().run(clean_image, base_context)
    assert det.heatmap.shape == clean_image.shape[:2]