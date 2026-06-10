import cv2
import numpy as np

from core.config import FONT_BASELINE_TOL
from core.types import BBox, Detection, TextRegion
from detectors.base import Context, Detector
from ingestion.ocr import crop_region

def antialiasing_profile(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(
        (crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def baseline_alignment(regions: list[TextRegion]) -> list[BBox]:
    if len(regions) < 3:
        return []
    baselines = [r.bbox.y + r.bbox.h for r in regions]
    median_y  = float(np.median(baselines))
    suspicious = []
    for r in regions:
        b = r.bbox.y + r.bbox.h
        if abs(b - median_y) > FONT_BASELINE_TOL and r.bbox.h < 60:
            suspicious.append(r.bbox)
    return suspicious

def analyze_text_consistency(img: np.ndarray,
                              regions: list[TextRegion]) -> Detection:
    if not regions:
        return Detection(detector_name='font', score=0.0,
                         details={'reason': 'no text regions found'})

    sharpness = []
    for r in regions:
        crop = crop_region(img, r.bbox)
        if crop.size == 0:
            continue
        sharpness.append(antialiasing_profile(crop))

    if not sharpness:
        return Detection(detector_name='font', score=0.0)

    mean_s = float(np.mean(sharpness))
    std_s  = float(np.std(sharpness))

    suspicious_regions = []
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)

    for i, r in enumerate(regions):
        if i >= len(sharpness):
            break
        deviation = abs(sharpness[i] - mean_s)
        if std_s > 0 and deviation > 2 * std_s:
            suspicious_regions.append(r.bbox)
            b = r.bbox
            heatmap[b.y:b.y+b.h, b.x:b.x+b.w] = float(
                np.clip(deviation / (mean_s + 1e-6), 0, 1)
            )

    baseline_anomalies = baseline_alignment(regions)
    suspicious_regions += baseline_anomalies
    for b in baseline_anomalies:
        heatmap[b.y:b.y+b.h, b.x:b.x+b.w] = 0.7

    score = float(np.clip(len(suspicious_regions) * 0.1 + heatmap.mean() * 5, 0, 1))

    return Detection(
        detector_name='font',
        score=score,
        heatmap=heatmap,
        regions=suspicious_regions,
        details={'mean_sharpness': mean_s, 'std_sharpness': std_s},
    )


class FontForensicsDetector(Detector):
    name = 'font'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            return analyze_text_consistency(img, ctx.text_regions)
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d