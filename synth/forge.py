import random

import cv2
import numpy as np

from core.types import TextRegion


def forge_copy_move(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w   = img.shape[:2]
    bh     = random.randint(20, h // 4)
    bw     = random.randint(20, w // 4)
    sy     = random.randint(0, h - bh)
    sx     = random.randint(0, w - bw)
    patch  = img[sy:sy+bh, sx:sx+bw].copy()

    dy     = random.randint(0, h - bh)
    dx     = random.randint(0, w - bw)
    result = img.copy()
    result[dy:dy+bh, dx:dx+bw] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy:dy+bh, dx:dx+bw] = 255
    return result, mask


def forge_text_edit(img: np.ndarray,
                    regions: list[TextRegion]) -> tuple[np.ndarray, np.ndarray]:
    if not regions:
        return forge_copy_move(img)

    region  = random.choice(regions)
    b       = region.bbox
    result  = img.copy()
    mask    = np.zeros(img.shape[:2], dtype=np.uint8)

    result[b.y:b.y+b.h, b.x:b.x+b.w] = 255
    fake = str(random.randint(1000, 99999))
    cv2.putText(result, fake, (b.x + 2, b.y + b.h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    mask[b.y:b.y+b.h, b.x:b.x+b.w] = 255
    return result, mask


def forge_splice(img: np.ndarray,
                 donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w   = img.shape[:2]
    dh, dw = donor.shape[:2]
    bh     = random.randint(20, min(h, dh) // 3)
    bw     = random.randint(20, min(w, dw) // 3)
    sy     = random.randint(0, dh - bh)
    sx     = random.randint(0, dw - bw)
    patch  = cv2.resize(donor[sy:sy+bh, sx:sx+bw], (bw, bh))

    dy     = random.randint(0, h - bh)
    dx     = random.randint(0, w - bw)
    result = img.copy()
    result[dy:dy+bh, dx:dx+bw] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy:dy+bh, dx:dx+bw] = 255
    return result, mask


def apply_random_forgery(img: np.ndarray,
                          regions=None,
                          donor=None) -> tuple[np.ndarray, np.ndarray, str]:
    ops = ['copy_move']
    if regions:
        ops.append('text_edit')
    if donor is not None:
        ops.append('splice')

    op = random.choice(ops)
    if op == 'copy_move':
        result, mask = forge_copy_move(img)
    elif op == 'text_edit':
        result, mask = forge_text_edit(img, regions)
    else:
        result, mask = forge_splice(img, donor)

    return result, mask, op