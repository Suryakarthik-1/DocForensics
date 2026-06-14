import numpy as np

from core.types import Detection
from fusion.decision import fuse, build_evidence


def D(name, score, hm=None):
    return Detection(name, score, heatmap=hm)


def test_all_clean_is_authentic():
    v = fuse([D('ela', 0.0), D('noise', 0.0), D('ai_generated', 0.0)])
    assert v.label == 'AUTHENTIC'
    assert v.is_tampered is False


def test_strong_ai_signal_labels_ai_generated():
    v = fuse([D('ai_generated', 0.9)])
    assert v.is_tampered is True
    assert v.label == 'AI-GENERATED'


def test_strong_tamper_signal_labels_tampered():
    v = fuse([D('ela', 0.9)])
    assert v.is_tampered is True
    assert v.label == 'TAMPERED'


def test_model_confidence_drives_verdict():
    v = fuse([D('ela', 0.0)], model_confidence=0.9)
    assert v.is_tampered is True
    assert v.label == 'TAMPERED'


def test_moderate_ai_is_not_labelled_ai_generated():
    # ai must be a confident (>=0.6) and dominant signal to claim AI-GENERATED
    v = fuse([D('ai_generated', 0.5), D('ela', 0.9)])
    assert v.label in ('TAMPERED', 'AUTHENTIC')
    assert v.label != 'AI-GENERATED'


def test_strong_signal_not_diluted_to_zero():
    # one strong detector among many calm ones should still raise suspicion
    dets = [D('ai_generated', 0.85)] + [D(n, 0.0) for n in
            ('ela', 'noise', 'copy_move', 'double_jpeg', 'font', 'metadata')]
    v = fuse(dets)
    assert v.confidence >= 0.45


def test_heatmaps_are_merged(clean_image):
    h, w = clean_image.shape[:2]
    dets = [D('ela', 0.5, np.ones((h, w), np.float32)),
            D('copy_move', 0.5, np.zeros((h, w), np.float32))]
    v = fuse(dets)
    assert v.fused_heatmap.shape == (h, w)
    assert v.fused_heatmap.max() <= 1.0


def test_evidence_lists_significant_detectors():
    msgs = build_evidence([D('ela', 0.8), D('noise', 0.05)])
    assert any('ELA' in m for m in msgs)
    # the calm detector (0.05) should not appear
    assert not any('Noise' in m for m in msgs)
