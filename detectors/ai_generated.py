import cv2
import numpy as np

from core.types import Detection
from detectors.base import Context, Detector

def frequency_analysis(img: np.ndarray) -> float:
    gray = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    fft    = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude   = np.log1p(np.abs(fft_shifted))

    h, w     = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # low frequency = center region, high frequency = outer ring
    low_freq  = magnitude[center_h-h//8:center_h+h//8,
                           center_w-w//8:center_w+w//8].mean()
    high_freq = magnitude.mean()

    # real photos: low_freq >> high_freq
    # AI images:   ratio closer to 1.0
    ratio = float(low_freq / (high_freq + 1e-6))
    # lower ratio = more suspicious
    score = float(np.clip(1.0 - (ratio - 1.0) / 10.0, 0, 1))
    return score

def check_exif_absence(path: str) -> tuple[float, str]:
    try:
        import exifread
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        has_camera = any(
            k in str(tags) for k in ['Make', 'Model', 'DateTime', 'ExifIFD']
        )
        if not has_camera:
            return 0.4, 'No camera EXIF data — may be AI generated'
        return 0.0, ''
    except Exception:
        return 0.0, ''
    
def ai_classifier_score(img: np.ndarray) -> float:
    try:
        from transformers import pipeline
        detector = pipeline(
            'image-classification',
            model='umm-maybe/AI-image-detector',
        )
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img_uint8)
        results = detector(pil_img)

        for r in results:
            if r['label'].lower() in ('artificial', 'fake', 'ai'):
                return float(r['score'])
        return 0.0
    except Exception:
        return 0.0
    
class AIGeneratedDetector(Detector):
    name = 'ai_generated'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            freq_score          = frequency_analysis(img)
            exif_score, exif_msg = check_exif_absence(ctx.file_path)
            model_score         = ai_classifier_score(img)

            # combine: model score is most reliable, freq is fast signal
            combined = float(np.clip(
                model_score * 0.6 + freq_score * 0.25 + exif_score * 0.15,
                0, 1
            ))

            flags = []
            if exif_msg:
                flags.append(exif_msg)
            if model_score > 0.5:
                flags.append(f'AI classifier confidence: {model_score:.0%}')
            if freq_score > 0.6:
                flags.append('Abnormal frequency distribution')

            return Detection(
                detector_name=self.name,
                score=combined,
                heatmap=None,
                regions=[],
                details={
                    'frequency_score':  round(freq_score, 4),
                    'exif_score':       round(exif_score, 4),
                    'model_score':      round(model_score, 4),
                    'flags':            flags,
                },
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d

