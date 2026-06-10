import cv2
import numpy as np

from core.config import CM_BLOCK_SIZE, CM_MATCH_THRESH, CM_STRIDE
from core.types import BBox, Detection
from detectors.base import Context, Detector

def block_hash_features(img: np.ndarray,
                         block: int = CM_BLOCK_SIZE,
                         stride: int = CM_STRIDE) -> dict[tuple, BBox]:
    gray = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    h, w     = gray.shape
    features = {}

    for y in range(0, h - block, stride):
        for x in range(0, w - block, stride):
            patch = gray[y:y+block, x:x+block]
            dct   = cv2.dct(patch)
            key   = tuple(np.round(dct[:4, :4].flatten(), 1))
            features[key] = BBox(x, y, block, block)

    return features

def _block_similarity(h1: tuple, h2: tuple) -> float:
    a, b  = np.array(h1), np.array(h2)
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1 - np.linalg.norm(a - b) / denom)


def match_duplicate_blocks(features: dict[tuple, BBox],
                            min_distance: int = 32) -> list[tuple[BBox, BBox]]:
    keys    = list(features.keys())
    matches = []
    seen    = set()

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if (i, j) in seen:
                continue
            b1, b2 = features[keys[i]], features[keys[j]]
            dist   = ((b1.x - b2.x)**2 + (b1.y - b2.y)**2) ** 0.5
            if dist < min_distance:
                continue
            if _block_similarity(keys[i], keys[j]) > CM_MATCH_THRESH:
                matches.append((b1, b2))
                seen.add((i, j))

    return matches

class CopyMoveDetector(Detector):
    name = 'copy_move'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            features = block_hash_features(img)
            matches  = match_duplicate_blocks(features)
            score    = float(np.clip(len(matches) * 0.05, 0, 1))

            heatmap = np.zeros(img.shape[:2], dtype=np.float32)
            regions = []
            for b1, b2 in matches:
                for b in (b1, b2):
                    heatmap[b.y:b.y+b.h, b.x:b.x+b.w] = 1.0
                    regions.append(b)

            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=heatmap,
                regions=regions,
                details={'match_count': len(matches)},
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d