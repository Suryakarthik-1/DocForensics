import cv2
import numpy as np

from core.config import CM_BLOCK_SIZE, CM_MATCH_THRESH, CM_STRIDE
from core.types import BBox, Detection
from detectors.base import Context, Detector

# A block must carry this much AC (texture) energy to be considered. This drops
# blank/near-blank regions AND the white document background, whose blocks would
# otherwise all hash to the same "mostly white" signature and match each other.
CM_MIN_AC_ENERGY = 40.0
# A real copy-move leaves many block pairs sharing the SAME shift vector.
# Repeated text/whitespace in a normal document gives scattered, inconsistent
# shifts — so we only trust a large cluster of matches with a common offset.
CM_MIN_CLUSTER = 10


def block_hash_features(img: np.ndarray,
                        block: int = CM_BLOCK_SIZE,
                        stride: int = CM_STRIDE) -> list[tuple[np.ndarray, int, int]]:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    feats = []
    for y in range(0, h - block, stride):
        for x in range(0, w - block, stride):
            patch = gray[y:y+block, x:x+block]
            dct = cv2.dct(patch)
            sig = dct[:4, :4].copy().flatten()
            sig[0] = 0.0                          # drop DC (average brightness)
            if np.linalg.norm(sig) < CM_MIN_AC_ENERGY:   # too little texture
                continue
            feats.append((sig, x, y))
    return feats


def find_shifted_duplicates(feats, block: int,
                            min_distance: int = 48) -> dict[tuple, list[BBox]]:
    """Group matching block pairs by their (dx, dy) shift vector."""
    shifts: dict[tuple, list[BBox]] = {}
    n = len(feats)
    for i in range(n):
        sig_i, xi, yi = feats[i]
        for j in range(i + 1, n):
            sig_j, xj, yj = feats[j]
            dx, dy = xj - xi, yj - yi
            if (dx * dx + dy * dy) ** 0.5 < min_distance:
                continue
            denom = np.linalg.norm(sig_i) + np.linalg.norm(sig_j)
            if denom == 0:
                continue
            if 1 - np.linalg.norm(sig_i - sig_j) / denom > CM_MATCH_THRESH:
                key = (round(dx / 8) * 8, round(dy / 8) * 8)   # quantize shift
                shifts.setdefault(key, []).extend(
                    [BBox(xi, yi, block, block), BBox(xj, yj, block, block)]
                )
    return shifts


class CopyMoveDetector(Detector):
    name = 'copy_move'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            feats  = block_hash_features(img)
            shifts = find_shifted_duplicates(feats, CM_BLOCK_SIZE)

            # The dominant shift vector = the suspected moved patch.
            best_boxes = max(shifts.values(), key=len) if shifts else []
            cluster    = len(best_boxes) // 2   # boxes come in pairs

            # Compactness separates a real forgery from repeated text. A moved
            # patch is a TIGHT cluster of blocks; legitimately-repeated document
            # text spreads matches across the whole page (low density).
            density = 0.0
            if best_boxes:
                xs = [b.x for b in best_boxes]; ys = [b.y for b in best_boxes]
                bb_area = (max(xs) - min(xs) + CM_BLOCK_SIZE) * \
                          (max(ys) - min(ys) + CM_BLOCK_SIZE)
                blocks_area = len(best_boxes) * CM_BLOCK_SIZE * CM_BLOCK_SIZE
                density = float(blocks_area / (bb_area + 1e-6))

            # NOTE: block matching is inherently weak on text documents, where
            # legitimately-repeated text mimics copy-move. We keep this as a soft,
            # low-weight hint (see FUSION_WEIGHTS) rather than a hard signal, and
            # cap the score so a normal page never reads as a confident forgery.
            flagged = cluster >= CM_MIN_CLUSTER and density >= 0.25
            score   = float(np.clip(cluster / 3000.0, 0, 0.6)) if flagged else 0.0

            heatmap = np.zeros(img.shape[:2], dtype=np.float32)
            regions = []
            if flagged:
                for b in best_boxes:
                    heatmap[b.y:b.y+b.h, b.x:b.x+b.w] = 1.0
                    regions.append(b)

            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=heatmap,
                regions=regions,
                details={'cluster_size': cluster,
                         'density': round(density, 3),
                         'shift_groups': len(shifts)},
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d
