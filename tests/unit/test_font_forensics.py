import numpy as np

from detectors.font_forensics import (FontForensicsDetector, baseline_alignment,
                                      analyze_text_consistency)
from core.types import BBox, TextRegion


def test_no_regions_scores_zero(clean_image, base_context):
    det = FontForensicsDetector().run(clean_image, base_context)
    assert det.detector_name == 'font'
    assert det.score == 0.0


def test_aligned_baselines_flag_nothing():
    # words on the SAME row share a baseline (y + h) → nothing suspicious
    regions = [
        TextRegion(BBox(10, 10, 50, 20), 'a', 0.9),
        TextRegion(BBox(70, 10, 50, 20), 'b', 0.9),
        TextRegion(BBox(130, 10, 50, 20), 'c', 0.9),
    ]
    assert baseline_alignment(regions) == []


def test_misaligned_baseline_is_detected():
    # one word sits off the shared baseline of the others → flagged
    regions = [
        TextRegion(BBox(10, 10, 50, 20), 'a', 0.9),
        TextRegion(BBox(70, 10, 50, 20), 'b', 0.9),
        TextRegion(BBox(130, 10, 50, 30), 'c', 0.9),   # baseline shifted down by 10
    ]
    flagged = baseline_alignment(regions)
    assert len(flagged) >= 1


def test_fewer_than_three_regions_not_flagged():
    regions = [TextRegion(BBox(10, 10, 50, 20), 'a', 0.9),
               TextRegion(BBox(10, 40, 50, 40), 'b', 0.9)]
    assert baseline_alignment(regions) == []


def test_detector_with_regions_in_range(clean_image, regions_context):
    det = FontForensicsDetector().run(clean_image, regions_context)
    assert 0.0 <= det.score <= 1.0


def test_consistency_returns_detection(clean_image, sample_regions):
    det = analyze_text_consistency(clean_image, sample_regions)
    assert det.detector_name == 'font'
    assert 0.0 <= det.score <= 1.0
