"""Generate a BALANCED tampering dataset.

Key idea: the genuine and tampered classes must contain the SAME source
documents (clean vs forged), otherwise the CNN learns document identity instead
of tampering. For every source page we emit, per round:
  * one augmented CLEAN copy   -> data/genuine   (label 0, blank mask)
  * one FORGED copy            -> data/tampered  (label 1, + mask)
Both get the same kind of photometric augmentation so the only systematic
difference between the classes is the forgery itself.
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from synth.forge import apply_random_forgery


def augment(img: np.ndarray) -> np.ndarray:
    """Light photometric augmentation: brightness, JPEG recompression, noise."""
    out = img.astype(np.float32)
    out = out * random.uniform(0.85, 1.15) + random.uniform(-12, 12)   # brightness
    out = np.clip(out, 0, 255).astype(np.uint8)

    if random.random() < 0.5:                                          # recompress
        q = random.randint(70, 95)
        ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if random.random() < 0.4:                                          # sensor noise
        noise = np.random.normal(0, 3, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def generate(genuine_dir: str, out_dir: str, per_class: int = 400, fresh: bool = True) -> None:
    src_path = Path(genuine_dir)
    sources  = sorted(src_path.glob('*.jpg')) + sorted(src_path.glob('*.png'))
    # Use only the real source documents, not previously-generated synthetic ones.
    sources  = [s for s in sources if 'synth' not in s.stem]
    if not sources:
        sources = sorted(src_path.glob('*.jpg')) + sorted(src_path.glob('*.png'))
    if not sources:
        raise FileNotFoundError(f'No source images in {genuine_dir}')

    gen_path  = Path(out_dir) / 'genuine'
    tam_path  = Path(out_dir) / 'tampered'
    mask_path = Path(out_dir) / 'masks'
    for p in (tam_path, mask_path):
        if fresh and p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    gen_path.mkdir(parents=True, exist_ok=True)
    if fresh:                       # drop previously-generated clean copies only
        for f in gen_path.glob('clean_*'):
            f.unlink()

    made = 0
    for i in range(per_class):
        src   = sources[i % len(sources)]
        donor = sources[(i + 1) % len(sources)]
        img   = cv2.imread(str(src))
        don   = cv2.imread(str(donor))
        if img is None:
            continue

        stem = f'{src.stem}_{i:05d}'

        # ── Genuine: augmented clean copy of the SAME source ──
        clean = augment(img)
        cv2.imwrite(str(gen_path / f'clean_{stem}.jpg'), clean)

        # ── Tampered: forge then apply the same kind of augmentation ──
        try:
            forged, mask, _ = apply_random_forgery(img, donor=don)
        except Exception as e:
            print(f'  forge failed on {src.name}: {e}')
            continue
        forged = augment(forged)
        cv2.imwrite(str(tam_path / f'{stem}.jpg'), forged)
        cv2.imwrite(str(mask_path / f'{stem}.png'), mask)
        made += 1

        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{per_class} pairs done')

    print(f'Done. {made} tampered + {per_class} clean copies '
          f'(plus any kept genuine_synth_*) in {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--genuine',   default='data/genuine')
    ap.add_argument('--out',       default='data')
    ap.add_argument('--per_class', type=int, default=400)
    args = ap.parse_args()
    generate(args.genuine, args.out, args.per_class)
