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
    # Double compression shows up as periodic gaps/peaks in the DCT-coefficient
    # histogram. A document's huge white background creates one dominant bin that
    # would swamp the signal, so we log-compress and drop the single largest bin
    # before measuring periodicity in the tails.
    h = hist.astype(np.float64).copy()
    if h.sum() <= 0:
        return 0.0, []

    h[h.argmax()] = 0.0                 # remove the dominant peak
    h = np.log1p(h)
    peak = h.max()
    if peak <= 0:
        return 0.0, []
    h = h / peak                        # normalise to 0..1

    diff2 = np.diff(h, n=2)
    periodicity = float(np.std(diff2))

    # Single-compressed images sit low here; conservative mapping.
    score = float(np.clip((periodicity - 0.10) / 0.30, 0, 1))
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