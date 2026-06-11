"""Generate synthetic clean document images to balance the genuine class."""
import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np


def random_word(length: int) -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def random_line(width: int) -> str:
    words = [random_word(random.randint(3, 9)) for _ in range(random.randint(5, 12))]
    return ' '.join(words)


def draw_document(h: int = 400, w: int = 320) -> np.ndarray:
    """Draw a clean synthetic document with text lines, a header, and optional table."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_small = 0.35
    font_med   = 0.50
    color      = (30, 30, 30)
    y          = 30

    # Header block
    header = random_word(random.randint(6, 12)).upper()
    cv2.putText(img, header, (w // 2 - len(header) * 7, y), font, 0.65, (0, 0, 0), 1)
    y += 20
    cv2.line(img, (20, y), (w - 20, y), (80, 80, 80), 1)
    y += 15

    # Label-value pairs (like a form)
    fields = random.randint(3, 6)
    for _ in range(fields):
        label = random_word(random.randint(4, 8)).capitalize() + ':'
        value = random_word(random.randint(5, 12))
        cv2.putText(img, label, (20, y),        font, font_small, color, 1)
        cv2.putText(img, value, (120, y),       font, font_small, color, 1)
        y += 18

    y += 10
    cv2.line(img, (20, y), (w - 20, y), (180, 180, 180), 1)
    y += 15

    # Body text lines
    lines = random.randint(6, 12)
    for _ in range(lines):
        text = random_line(w)[:40]
        cv2.putText(img, text, (20, y), font, font_small, color, 1)
        y += 16
        if y > h - 60:
            break

    # Optional simple table
    if random.random() > 0.4 and y < h - 80:
        y += 10
        cols, rows = 3, random.randint(2, 4)
        cw = (w - 40) // cols
        for r in range(rows + 1):
            ry = y + r * 18
            cv2.line(img, (20, ry), (w - 20, ry), (120, 120, 120), 1)
            if r < rows:
                for c in range(cols):
                    cell = random_word(random.randint(3, 7))
                    cv2.putText(img, cell, (22 + c * cw, ry + 13), font, font_small, color, 1)
        for c in range(cols + 1):
            cv2.line(img, (20 + c * cw, y), (20 + c * cw, y + rows * 18), (120, 120, 120), 1)

    # Light background noise to simulate paper texture
    noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
    img   = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

    return img


def generate_genuine(out_dir: str, count: int = 200) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        h = random.randint(350, 500)
        w = random.randint(280, 400)
        img = draw_document(h, w)
        cv2.imwrite(str(out / f'genuine_synth_{i:04d}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, random.randint(85, 98)])
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{count} done')

    print(f'Generated {count} synthetic genuine documents in {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',   default='data/genuine')
    parser.add_argument('--count', type=int, default=200)
    args = parser.parse_args()
    generate_genuine(args.out, args.count)
