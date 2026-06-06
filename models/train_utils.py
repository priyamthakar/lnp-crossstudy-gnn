"""
Training loop, evaluation metrics, and prediction utilities for LNP-MFGO.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr

from models.gnn import LIPID_NAMES


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_metrics(y_true, y_pred) -> dict:
    yt, yp = np.array(y_true), np.array(y_pred)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 3:
        return dict(R2=np.nan, RMSE=np.nan, MAE=np.nan,
                    Pearson_r=np.nan, Spearman_rho=np.nan, N=len(yt))
    return dict(
        R2=r2_score(yt, yp),
        RMSE=np.sqrt(mean_squared_error(yt, yp)),
        MAE=mean_absolute_error(yt, yp),
        Pearson_r=pearsonr(yt, yp)[0],
        Spearman_rho=spearmanr(yt, yp)[0],
        N=len(yt),
    )


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def to_device(graph_batch, tabular, targets, study_ids, device):
    gb = {}
    for name in LIPID_NAMES:
        x, ei, bna = graph_batch[name]
        gb[name] = (x.to(device), ei.to(device), bna)
    return gb, tabular.to(device), targets.to(device), study_ids.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 150,
    lr: float = 5e-4,
    patience: int = 30,
    weight_decay: float = 1e-4,
):
    """Train with AdamW + cosine LR schedule + early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for graph_batch, tabular, targets, study_ids in train_loader:
            gb, tab, tgt, sid = to_device(graph_batch, tabular, targets, study_ids, device)
            optimizer.zero_grad()
            pred, _ = model(gb, tab, sid)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for graph_batch, tabular, targets, study_ids in val_loader:
                gb, tab, tgt, sid = to_device(graph_batch, tabular, targets, study_ids, device)
                pred, _ = model(gb, tab, sid)
                val_loss += criterion(pred, tgt).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, epoch + 1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, loader, device):
    """Return (predictions, true_values, attention_weights) arrays."""
    model.eval()
    preds, trues, attns = [], [], []
    with torch.no_grad():
        for graph_batch, tabular, targets, study_ids in loader:
            gb, tab, tgt, sid = to_device(graph_batch, tabular, targets, study_ids, device)
            pred, attn_w = model(gb, tab, sid)
            preds.append(pred.cpu().numpy())
            trues.append(tgt.cpu().numpy())
            attns.append(attn_w.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(attns)
