import numpy as np

from core.types import BBox, TextRegion


def extract_text_regions(img: np.ndarray) -> list[TextRegion]:
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        img_uint8 = (np.clip(img, 0, 1) * 255).astype('uint8')
        result = ocr.ocr(img_uint8, cls=True)
        regions = []
        if result and result[0]:
            for line in result[0]:
                pts, (text, conf) = line
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x, y = int(min(xs)), int(min(ys))
                w    = int(max(xs) - min(xs))
                h    = int(max(ys) - min(ys))
                regions.append(TextRegion(bbox=BBox(x, y, w, h), text=text, confidence=conf))
        return regions
    except (ImportError, Exception):
        return _tesseract(img)


def _tesseract(img: np.ndarray) -> list[TextRegion]:
    import pytesseract
    img_uint8 = (np.clip(img, 0, 1) * 255).astype('uint8')
    data = pytesseract.image_to_data(img_uint8, output_type=pytesseract.Output.DICT)
    regions = []
    for i, text in enumerate(data['text']):
        if text.strip() and int(data['conf'][i]) > 30:
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            conf = float(data['conf'][i]) / 100.0
            regions.append(TextRegion(bbox=BBox(x, y, w, h), text=text, confidence=conf))
    return regions


def crop_region(img: np.ndarray, bbox: BBox) -> np.ndarray:
    return img[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]