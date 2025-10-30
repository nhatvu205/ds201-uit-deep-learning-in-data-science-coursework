from __future__ import annotations

import numpy as np
import torch


def update_confusion(conf: np.ndarray, preds: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        p = preds.detach().cpu().view(-1)
        t = labels.detach().cpu().view(-1)
    for true_c, pred_c in zip(t.tolist(), p.tolist()):
        conf[true_c, pred_c] += 1
    return conf


def precision_recall_f1(conf: np.ndarray) -> tuple[float, float, float]:
    eps = 1e-12
    tp = np.diag(conf)
    pred_pos = conf.sum(axis=0)
    real_pos = conf.sum(axis=1)

    prec = (tp / (pred_pos + eps))
    rec = (tp / (real_pos + eps))
    f1 = (2 * prec * rec) / (prec + rec + eps)

    return float(np.nanmean(prec)), float(np.nanmean(rec)), float(np.nanmean(f1))


def accuracy(conf: np.ndarray) -> float:
    return float(np.trace(conf) / max(conf.sum(), 1))


