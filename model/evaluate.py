import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score


def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_labels, all_scores = [], []
    total_iou, n = 0.0, 0

    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred_mask, pred_logit = model(imgs)

            probs = torch.sigmoid(pred_logit).view(-1).cpu().numpy()

            all_scores.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

            pred_bin = (pred_mask > 0.5).float()
            gt_bin   = (masks > 0.5).float()
            intersection = (pred_bin * gt_bin).sum()
            union        = pred_bin.sum() + gt_bin.sum() - intersection
            total_iou   += (intersection / (union + 1e-6)).item()
            n           += 1

    preds = [1 if s > 0.5 else 0 for s in all_scores]
    auc   = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0

    return {
        'auc':       round(float(auc), 4),
        'f1':        round(float(f1_score(all_labels, preds, zero_division=0)), 4),
        'pixel_iou': round(total_iou / max(n, 1), 4),
    }