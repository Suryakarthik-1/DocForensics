import cv2
import numpy as np

from core.config import FUSION_WEIGHTS, TAMPER_THRESHOLD
from core.types import Detection, Verdict

LABELS = {
    'ela':          'ELA anomaly (pixel error inconsistency)',
    'noise':        'Noise fingerprint mismatch',
    'copy_move':    'Cloned/duplicated region detected',
    'double_jpeg':  'Double compression artifact',
    'font':         'Font or text baseline inconsistency',
    'metadata':     'Suspicious metadata',
    'ai_generated': 'AI-generated image signature',
    'model':        'CNN tampering localization',
}


def build_evidence(detections: list[Detection]) -> list[str]:
    messages = []
    for det in detections:
        if det.score > 0.3:
            base  = LABELS.get(det.detector_name, det.detector_name)
            flags = det.details.get('flags', [])
            if flags:
                messages.append(f'{base}: {"; ".join(flags)}')
            else:
                messages.append(f'{base} (score={det.score:.0%})')
    return messages or ['No significant tampering signals detected']


def fuse(detections: list[Detection], model_confidence: float = 0.0,
         model_heatmap: np.ndarray | None = None,
         weights: dict | None = None) -> Verdict:

    w      = weights or FUSION_WEIGHTS
    scores = {d.detector_name: d.score for d in detections}
    scores['model'] = model_confidence

    # ── Base: weighted average of every signal ──
    weighted = sum(w.get(name, 0.0) * s for name, s in scores.items())

    # ── Strong-signal boost: a single confident specialist shouldn't be diluted
    # to nothing by the calmer detectors. Only the RELIABLE detectors get to
    # drive this (copy_move / double_jpeg are too noisy on documents).
    strongest = max(scores.get('ai_generated', 0.0),
                    scores.get('model', 0.0),
                    scores.get('ela', 0.0))
    fused_score = float(np.clip(max(weighted, 0.55 * strongest + 0.45 * weighted), 0, 1))

    # ── Decide a human-readable label ──
    ai = scores.get('ai_generated', 0.0)
    is_flagged = fused_score >= TAMPER_THRESHOLD
    if is_flagged and ai >= 0.6 and ai >= strongest - 1e-6:
        label = 'AI-GENERATED'
    elif is_flagged:
        label = 'TAMPERED'
    else:
        label = 'AUTHENTIC'

    # ── Merge heatmaps (weighted) ──
    heatmaps = [(d.heatmap, w.get(d.detector_name, 0.0))
                for d in detections if d.heatmap is not None]
    if model_heatmap is not None:
        heatmaps.append((model_heatmap, w.get('model', 0.2)))

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
        merged = np.clip(np.nan_to_num(merged, nan=0.0), 0, 1)
    else:
        merged = np.zeros((256, 256), dtype=np.float32)

    return Verdict(
        is_tampered=is_flagged,
        confidence=fused_score,
        fused_heatmap=merged,
        evidence=build_evidence(detections),
        per_detector=detections,
        label=label,
    )
