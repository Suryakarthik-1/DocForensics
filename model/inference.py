from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from core.config import CHECKPOINTS_DIR, MODEL_INPUT_SIZE
from model.architecture import TamperNet


@dataclass
class ModelOut:
    confidence: float       # 0..1, probability of tampering
    heatmap: np.ndarray     # H x W float32, same spatial size as input image


_cached_model = None


def load_model(weights_path: str | None = None) -> TamperNet:
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    path = weights_path or str(CHECKPOINTS_DIR / 'best.pt')
    model = TamperNet()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    _cached_model = model
    return model


def predict(model: TamperNet, img: np.ndarray) -> ModelOut:
    """Run the model on a single HxWx3 float32 image (values 0..1)."""
    h, w = img.shape[:2]

    # Resize to model input size
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
    tensor = F.interpolate(tensor, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                           mode='bilinear', align_corners=False)

    with torch.no_grad():
        pred_mask, pred_logit = model(tensor)

    # Confidence: sigmoid of the classification logit
    confidence = torch.sigmoid(pred_logit).item()

    # Heatmap: resize back to original image dimensions
    heatmap = F.interpolate(pred_mask, size=(h, w), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy()  # (H, W)

    return ModelOut(confidence=confidence, heatmap=heatmap)
