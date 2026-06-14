import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import (BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOP_PATIENCE,
                          LEARNING_RATE, MAX_EPOCHS)
from model.architecture import TamperNet
from model.dataset import TamperDataset
from model.evaluate import evaluate


def compute_loss(pred_mask, pred_logit, gt_mask, gt_label, pos_weight: float = 1.0):
    # pos_weight scales the POSITIVE (tampered) class. Tampered is the majority
    # here, so pos_weight < 1 down-weights it and balances toward genuine.
    pw         = torch.tensor([pos_weight], device=pred_logit.device)
    label_loss = nn.BCEWithLogitsLoss(pos_weight=pw)(pred_logit.view(-1), gt_label.float())

    # Only apply mask loss on tampered samples (label=1); genuine masks are trivially zero
    tampered = gt_label.bool()
    if tampered.any():
        mask_loss = nn.BCELoss()(pred_mask[tampered], gt_mask[tampered])
    else:
        mask_loss = torch.tensor(0.0, device=pred_logit.device)

    return label_loss + 0.5 * mask_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    train_ds = TamperDataset(split='train')
    val_ds   = TamperDataset(split='val')
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Balance the classes: pos_weight = n_genuine / n_tampered
    labels    = [lbl for _, lbl in train_ds.items]
    n_tamper  = max(sum(labels), 1)
    n_genuine = max(len(labels) - n_tamper, 1)
    pos_weight = n_genuine / n_tamper
    print(f'Class balance: {n_genuine} genuine / {n_tamper} tampered → pos_weight={pos_weight:.3f}')

    model = TamperNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_auc = 0.0
    no_improve = 0
    history  = []

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks, labels in train_dl:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            opt.zero_grad()
            pred_mask, pred_logit = model(imgs)
            loss = compute_loss(pred_mask, pred_logit, masks, labels, pos_weight)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        metrics = evaluate(model, val_dl, device)
        sched.step(metrics['auc'])

        print(f"Epoch {epoch:03d} | loss={train_loss/len(train_dl):.4f} "
              f"| auc={metrics['auc']:.4f} | iou={metrics['pixel_iou']:.4f}")

        history.append({'epoch': epoch, **metrics})

        if metrics['auc'] > best_auc:
            best_auc   = metrics['auc']
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / 'best.pt')
            print(f'  saved best model (auc={best_auc:.4f})')
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    # Always save final weights so inference has something to load
    torch.save(model.state_dict(), CHECKPOINTS_DIR / 'last.pt')
    if not (CHECKPOINTS_DIR / 'best.pt').exists():
        torch.save(model.state_dict(), CHECKPOINTS_DIR / 'best.pt')
        print('Saved best.pt (fallback — auc never exceeded 0.0)')

    with open(CHECKPOINTS_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training done. Best AUC: {best_auc:.4f}')