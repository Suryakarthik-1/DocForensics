import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from core.config import GENUINE_DIR, MASKS_DIR, MODEL_INPUT_SIZE, TAMPERED_DIR


class TamperDataset(Dataset):
    def __init__(self, split: str = 'train', val_fraction: float = 0.15):
        genuine  = sorted(GENUINE_DIR.glob('*.jpg'))  + sorted(GENUINE_DIR.glob('*.png'))
        tampered = sorted(TAMPERED_DIR.glob('*.jpg')) + sorted(TAMPERED_DIR.glob('*.png'))

        all_items = [(p, 0) for p in genuine] + [(p, 1) for p in tampered]
        cut       = int(len(all_items) * (1 - val_fraction))

        self.items   = all_items[:cut] if split == 'train' else all_items[cut:]
        self.size    = MODEL_INPUT_SIZE
        self.augment = (split == 'train')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.items[i]

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if label == 1:
            mask_path = MASKS_DIR / (path.stem + '.png')
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.size, self.size))
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.ones((self.size, self.size), dtype=np.float32)
        else:
            mask = np.zeros((self.size, self.size), dtype=np.float32)

        if self.augment:
            img, mask = _augment(img, mask)

        img_t  = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)       # (1, H, W)
        return img_t, mask_t, label


def _augment(img: np.ndarray, mask: np.ndarray):
    if np.random.rand() > 0.5:
        img  = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    if np.random.rand() > 0.5:
        img  = img[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    return img, mask