import cv2
import numpy as np

from core.config import FUSION_WEIGHTS, TAMPER_THRESHOLD
from core.types import Detection, Verdict

def build_evidence(detections: list[Detection]) -> list[str]:
    labels = {
        'ela':         'ELA anomaly (pixel error inconsistency)',
        'noise':       'Noise fingerprint mismatch',
        'copy_move':   'Cloned/duplicated region detected',
        'double_jpeg': 'Double compression artifact',
        'font':        'Font or text baseline inconsistency',
        'metadata':    'Suspicious metadata',
    }
    messages = []
    for det in detections:
        if det.score > 0.2:
            base  = labels.get(det.detector_name, det.detector_name)
            flags = det.details.get('flags', [])
            if flags:
                messages.append(f'{base}: {"; ".join(flags)}')
            else:
                messages.append(f'{base} (score={det.score:.2f})')
    return messages or ['No significant tampering signals detected']

def fuse(detections: list[Detection], model_confidence: float = 0.0,
         model_heatmap: np.ndarray | None = None,
         weights: dict | None = None) -> Verdict:

    w            = weights or FUSION_WEIGHTS
    fused_score  = 0.0
    heatmaps     = []

    for det in detections:
        weight = w.get(det.detector_name, 0.0)
        fused_score += weight * det.score
        if det.heatmap is not None:
            heatmaps.append((det.heatmap, weight))

    fused_score += w.get('model', 0.2) * model_confidence
    if model_heatmap is not None:
        heatmaps.append((model_heatmap, w.get('model', 0.2)))

    # merge all heatmaps into one weighted average
    if heatmaps:
        ref_h, ref_w = heatmaps[0][0].shape
        merged  = np.zeros((ref_h, ref_w), dtype=np.float32)
        total_w = 0.0
        for hmap, hw in heatmaps:
            resized = cv2.resize(hmap.astype(np.float32), (ref_w, ref_h))
            merged  += resized * hw
            total_w += hw
        if total_w > 0:
            merged /= total_w
        merged = np.clip(merged, 0, 1)
    else:
        merged = np.zeros((256, 256), dtype=np.float32)

    fused_score = float(np.clip(fused_score, 0, 1))
    evidence    = build_evidence(detections)

    return Verdict(
        is_tampered=fused_score >= TAMPER_THRESHOLD,
        confidence=fused_score,
        fused_heatmap=merged,
        evidence=evidence,
        per_detector=detections,
    )