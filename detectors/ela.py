import io

import cv2
import numpy as np
from PIL import Image

from core.config import ELA_AMPLIFY, ELA_QUALITY, ELA_THRESHOLD
from core.types import BBox, Detection
from detectors.base import Context, Detector

def compute_ela(img: np.ndarray, quality: int = ELA_QUALITY) -> np.ndarray:
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8)

    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    recompressed = np.array(Image.open(buf)).astype(np.float32)

    original_f = img_uint8.astype(np.float32)
    diff       = np.abs(original_f - recompressed)
    amplified  = np.clip(diff * ELA_AMPLIFY / 255.0, 0, 1)
    return amplified.mean(axis=2)

def ela_score(error_map: np.ndarray, threshold: float = ELA_THRESHOLD
              ) -> tuple[float, list[BBox]]:
    mask = (error_map > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w   = error_map.shape
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:
            continue
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        regions.append(BBox(x, y, bw, bh))

    covered = mask.sum() / (h * w)
    mean_err = float(error_map[error_map > threshold].mean()) if mask.any() else 0.0
    score = float(np.clip(covered * 5 + mean_err, 0, 1))
    return score, regions


class ELADetector(Detector):
    name = 'ela'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            error_map = compute_ela(img)
            score, regions = ela_score(error_map)
            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=error_map,
                regions=regions,
                details={'mean_error': float(error_map.mean())},
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d