import numpy as np
import pytest

from detectors.base import Context
from detectors.ela import ELADetector
from detectors.copymove import CopyMoveDetector
from fusion.decision import fuse
from core.types import Detection


def test_all_detectors_return_detection(clean_image, base_context):
    from detectors.noise import NoiseDetector
    from detectors.double_jpeg import DoubleJpegDetector
    from detectors.font_forensics import FontForensicsDetector
    from detectors.metadata import MetadataDetector
    from detectors.ai_generated import AIGeneratedDetector

    detectors  = [
        ELADetector(), NoiseDetector(), CopyMoveDetector(),
        DoubleJpegDetector(), FontForensicsDetector(),
        MetadataDetector(), AIGeneratedDetector(),
    ]
    for d in detectors:
        result = d.run(clean_image, base_context)
        assert result.detector_name == d.name
        assert 0.0 <= result.score <= 1.0


def test_fuse_produces_verdict(clean_image):
    h, w = clean_image.shape[:2]
    detections = [
        Detection('ela',   0.1, heatmap=np.zeros((h, w), dtype=np.float32)),
        Detection('noise', 0.2, heatmap=np.zeros((h, w), dtype=np.float32)),
    ]
    verdict = fuse(detections)
    assert verdict.confidence >= 0.0
    assert isinstance(verdict.is_tampered, bool)
    assert verdict.fused_heatmap.shape == (h, w)


def test_clean_image_not_flagged(clean_image):
    h, w = clean_image.shape[:2]
    detections = [
        Detection(name, 0.0, heatmap=np.zeros((h, w), dtype=np.float32))
        for name in ('ela', 'noise', 'copy_move', 'double_jpeg', 'font', 'metadata', 'ai_generated')
    ]
    verdict = fuse(detections)
    assert not verdict.is_tampered


def test_high_scores_trigger_tampered(clean_image):
    h, w = clean_image.shape[:2]
    detections = [
        Detection(name, 0.9, heatmap=np.ones((h, w), dtype=np.float32))
        for name in ('ela', 'noise', 'copy_move', 'double_jpeg', 'font', 'metadata', 'ai_generated')
    ]
    verdict = fuse(detections)
    assert verdict.is_tampered


def test_evidence_list_not_empty(clean_image):
    h, w = clean_image.shape[:2]
    detections = [Detection('ela', 0.8, heatmap=np.ones((h, w), dtype=np.float32))]
    verdict = fuse(detections)
    assert len(verdict.evidence) > 0