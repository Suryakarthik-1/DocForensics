import numpy as np

from core.types import BBox, TextRegion, Detection, Verdict


def test_bbox_fields():
    b = BBox(1, 2, 3, 4)
    assert (b.x, b.y, b.w, b.h) == (1, 2, 3, 4)


def test_detection_defaults():
    d = Detection('ela', 0.5)
    assert d.heatmap is None
    assert d.regions == []
    assert d.details == {}


def test_detection_mutable_defaults_are_independent():
    a, b = Detection('a', 0.1), Detection('b', 0.2)
    a.regions.append(BBox(0, 0, 1, 1))
    assert b.regions == []          # no shared mutable default


def test_verdict_default_label():
    v = Verdict(is_tampered=False, confidence=0.1,
                fused_heatmap=np.zeros((4, 4), np.float32),
                evidence=[], per_detector=[])
    assert v.label == 'AUTHENTIC'


def test_text_region_holds_bbox_and_text():
    r = TextRegion(BBox(0, 0, 5, 5), 'hi', 0.88)
    assert r.text == 'hi'
    assert r.confidence == 0.88
