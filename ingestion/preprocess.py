import cv2
import numpy as np

from core.config import MAX_IMAGE_SIDE


def resize_keep_ratio(img: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    gray   = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    h, w   = img.shape[:2]
    center = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        (img * 255).astype(np.uint8), M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated.astype(np.float32) / 255.0, angle