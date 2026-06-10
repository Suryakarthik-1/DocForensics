import warnings
import cv2
import numpy as np
from skimage.restoration import estimate_sigma

from core.config import NOISE_THRESHOLD, NOISE_WINDOW
from core.types import BBox, Detection
from detectors.base import Context, Detector

def estimate_local_noise(img: np.ndarray, window: int = NOISE_WINDOW) -> np.ndarray:
    gray = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    h, w      = gray.shape
    noise_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - window, window // 2):
        for x in range(0, w - window, window // 2):
            block = gray[y:y+window, x:x+window]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sigma = estimate_sigma(block, average_sigmas=True)
            noise_map[y:y+window, x:x+window] = float(sigma)

    if noise_map.max() > 0:
        noise_map = noise_map / noise_map.max()
    noise_map = np.nan_to_num(noise_map, nan=0.0)
    return noise_map

def find_noise_anomalies(noise_map: np.ndarray,
                         threshold: float = NOISE_THRESHOLD
                         ) -> tuple[float, list[BBox]]:
    mean_n = float(noise_map.mean())
    std_n  = float(noise_map.std())
    if std_n == 0:
        return 0.0, []

    anomaly = (np.abs(noise_map - mean_n) > (2 * std_n + threshold)).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(anomaly, connectivity=8)

    regions = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        regions.append(BBox(
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        ))

    score = float(np.clip(anomaly.mean() * 10 + len(regions) * 0.05, 0, 1))
    return score, regions

class NoiseDetector(Detector):
    name = 'noise'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            noise_map = estimate_local_noise(img)
            score, regions = find_noise_anomalies(noise_map)
            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=noise_map,
                regions=regions,
                details={
                    'mean_noise': float(np.nan_to_num(noise_map.mean())),
                    'std':        float(np.nan_to_num(noise_map.std())),
                },
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d