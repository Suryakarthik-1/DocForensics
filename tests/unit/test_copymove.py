import numpy as np

from core.config import CM_BLOCK_SIZE
from detectors.copymove import (CopyMoveDetector, block_hash_features,
                                find_shifted_duplicates)


def make_cloned_image():
    img = np.random.default_rng(0).uniform(0, 1, (128, 128, 3)).astype(np.float32)
    img[80:112, 80:112] = img[0:32, 0:32]   # clone a textured 32x32 patch
    return img


def test_block_hash_returns_list_of_features():
    img = np.random.default_rng(1).uniform(0, 1, (64, 64, 3)).astype(np.float32)
    features = block_hash_features(img, block=16, stride=8)
    assert isinstance(features, list)
    assert len(features) > 0


def test_flat_blocks_are_skipped():
    flat = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    assert block_hash_features(flat, block=16, stride=8) == []


def test_detects_cloned_block():
    img = make_cloned_image()
    features = block_hash_features(img, block=CM_BLOCK_SIZE, stride=8)
    shifts   = find_shifted_duplicates(features, CM_BLOCK_SIZE, min_distance=16)
    assert any(len(boxes) >= 2 for boxes in shifts.values())


def test_score_in_range(base_context, clean_image):
    det = CopyMoveDetector().run(clean_image, base_context)
    assert 0.0 <= det.score <= 1.0


def test_heatmap_shape(base_context, clean_image):
    det = CopyMoveDetector().run(clean_image, base_context)
    assert det.heatmap.shape == clean_image.shape[:2]
