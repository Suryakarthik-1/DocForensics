from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import MAX_FILE_MB, MAX_IMAGE_SIDE, SUPPORTED_EXTS


@dataclass
class PageImage:
    image: np.ndarray
    page_num: int
    source_path: str


def is_supported(path: str) -> bool:
    ext     = Path(path).suffix.lower()
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    return ext in SUPPORTED_EXTS and size_mb <= MAX_FILE_MB


def to_normalized(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def load_image_file(path: str) -> PageImage:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PageImage(image=to_normalized(img), page_num=0, source_path=path)


def load_pdf_file(path: str) -> list[PageImage]:
    import fitz
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        pages.append(PageImage(image=to_normalized(img), page_num=i, source_path=path))
    doc.close()
    return pages


def load_document(path: str) -> list[PageImage]:
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not is_supported(path):
        raise ValueError(f"Unsupported file type or too large: {path}")

    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf_file(path)
    return [load_image_file(path)]