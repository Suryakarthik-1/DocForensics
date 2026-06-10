import argparse
from pathlib import Path

import cv2
import numpy as np

from synth.forge import apply_random_forgery


def generate(genuine_dir: str, out_dir: str, count: int = 200) -> None:
    genuine_path  = Path(genuine_dir)
    tampered_path = Path(out_dir) / 'tampered'
    masks_path    = Path(out_dir) / 'masks'
    tampered_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)

    sources = list(genuine_path.glob('*.jpg')) + list(genuine_path.glob('*.png'))
    if not sources:
        raise FileNotFoundError(f'No images in {genuine_dir}')

    for i in range(count):
        src = sources[i % len(sources)]
        img = cv2.imread(str(src))
        if img is None:
            print(f'  skipping unreadable: {src}')
            continue

        donor = cv2.imread(str(sources[(i + 1) % len(sources)]))

        try:
            forged, mask, op = apply_random_forgery(img, donor=donor)
        except Exception as e:
            print(f'  forge failed on {src.name}: {e}')
            continue

        name = f'{src.stem}_forge_{i:05d}'
        cv2.imwrite(str(tampered_path / f'{name}.jpg'), forged)
        cv2.imwrite(str(masks_path    / f'{name}.png'), mask)

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{count} done')

    print(f'Generated {count} tampered images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genuine', default='data/genuine')
    parser.add_argument('--out',     default='data')
    parser.add_argument('--count',   type=int, default=200)
    args = parser.parse_args()
    generate(args.genuine, args.out, args.count)