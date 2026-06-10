import cv2
import numpy as np
from scipy.fftpack import dct as scipy_dct

from core.config import DCT_HIST_BINS
from core.types import Detection
from detectors.base import Context, Detector

def dct_histogram(img: np.ndarray, bins: int = DCT_HIST_BINS) -> np.ndarray:
    gray   = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    coeffs = scipy_dct(scipy_dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    hist, _ = np.histogram(coeffs.flatten(), bins=bins, range=(-500, 500))
    return hist.astype(np.float32)

def detect_double_compression(hist: np.ndarray) -> tuple[float, list]:
    diff2     = np.diff(hist.astype(np.float64), n=2)
    periodicity = float(np.std(diff2))
    mean_val    = float(hist.mean())

    if mean_val > 0:
        score = float(np.clip(periodicity / (mean_val + 1e-6) * 0.1, 0, 1))
    else:
        score = 0.0
    return score, []

class DoubleJpegDetector(Detector):
    name = 'double_jpeg'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            hist  = dct_histogram(img)
            score, regions = detect_double_compression(hist)
            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=None,
                regions=regions,
                details={'histogram_std': float(hist.std())},
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d