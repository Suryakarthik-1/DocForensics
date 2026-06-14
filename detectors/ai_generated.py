import os

import cv2
import numpy as np

from core.types import Detection
from detectors.base import Context, Detector

# ── Cached HuggingFace pipeline (loaded once, reused across requests) ──────────
_classifier = None
_classifier_failed = False


def _get_classifier():
    """Lazily load the AI-image classifier exactly once.

    Set DOCFORENSICS_DISABLE_AI_MODEL=1 to skip the ~350 MB download entirely
    (useful for CI and memory-constrained hosts — the detector then relies on its
    frequency/EXIF signals only).
    """
    global _classifier, _classifier_failed
    if os.getenv('DOCFORENSICS_DISABLE_AI_MODEL') == '1':
        _classifier_failed = True
        return None
    if _classifier is not None or _classifier_failed:
        return _classifier
    try:
        from transformers import pipeline
        _classifier = pipeline(
            'image-classification',
            model='umm-maybe/AI-image-detector',
        )
    except Exception:
        _classifier_failed = True
        _classifier = None
    return _classifier


def frequency_analysis(img: np.ndarray) -> float:
    """AI/GAN images often have an unnaturally flat frequency spectrum."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    fft         = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude   = np.log1p(np.abs(fft_shifted))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    low_freq  = magnitude[cy - h // 8:cy + h // 8, cx - w // 8:cx + w // 8].mean()
    high_freq = magnitude.mean()

    ratio = float(low_freq / (high_freq + 1e-6))
    # Real photos have rich high-freq detail (ratio ~1.5-3+). A ratio near 1.0
    # means the spectrum is suspiciously flat. Only flag the genuinely-flat case.
    score = float(np.clip((1.3 - ratio) / 0.5, 0, 1))
    return score


def check_exif_absence(path: str) -> tuple[float, str]:
    try:
        import exifread
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        has_camera = any(k in str(tags) for k in ['Make', 'Model', 'DateTime', 'ExifIFD'])
        if not has_camera:
            return 0.3, 'No camera EXIF data — may be AI-generated or edited'
        return 0.0, ''
    except Exception:
        return 0.0, ''


def ai_classifier_score(img: np.ndarray) -> float:
    clf = _get_classifier()
    if clf is None:
        return 0.0
    try:
        from PIL import Image
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        results   = clf(Image.fromarray(img_uint8))
        for r in results:
            label = r['label'].lower()
            if label in ('artificial', 'fake', 'ai', 'ai-generated', 'generated'):
                return float(r['score'])
            if label in ('human', 'real', 'photo', 'photograph'):
                return float(1.0 - r['score'])
        return 0.0
    except Exception:
        return 0.0


class AIGeneratedDetector(Detector):
    name = 'ai_generated'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            freq_score           = frequency_analysis(img)
            exif_score, exif_msg = check_exif_absence(ctx.file_path)
            model_score          = ai_classifier_score(img)

            # The trained classifier is by far the most reliable signal.
            combined = float(np.clip(
                model_score * 0.75 + freq_score * 0.15 + exif_score * 0.10, 0, 1
            ))

            flags = []
            if model_score > 0.5:
                flags.append(f'AI classifier confidence: {model_score:.0%}')
            if freq_score > 0.6:
                flags.append('Unnaturally flat frequency spectrum')
            if exif_msg:
                flags.append(exif_msg)

            return Detection(
                detector_name=self.name,
                score=combined,
                heatmap=None,
                regions=[],
                details={
                    'frequency_score': round(freq_score, 4),
                    'exif_score':      round(exif_score, 4),
                    'model_score':     round(model_score, 4),
                    'flags':           flags,
                },
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d
