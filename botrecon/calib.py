"""Calibration and confidence metrics (numpy only)."""
from __future__ import annotations

import numpy as np


def ece(y: np.ndarray, p_bot: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the predicted-class confidence, equal-width bins."""
    y = np.asarray(y).astype(int); p = np.asarray(p_bot, float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def brier(y, p_bot) -> float:
    return float(np.mean((np.asarray(p_bot, float) - np.asarray(y, float)) ** 2))


def selective_risk(y, p_bot, coverage: float) -> float:
    """Error rate on the `coverage` share of accounts with the highest confidence."""
    y = np.asarray(y).astype(int); p = np.asarray(p_bot, float)
    conf = np.abs(p - 0.5)
    k = max(1, int(round(coverage * len(y))))
    keep = np.argsort(-conf, kind="stable")[:k]
    return float((((p[keep] >= 0.5).astype(int)) != y[keep]).mean())


def reliability_curve(y, p_bot, bins: int = 10):
    """(mean confidence, accuracy, count) per bin of predicted-class confidence."""
    y = np.asarray(y).astype(int); p = np.asarray(p_bot, float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = []
    for i in range(bins):
        sel = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        if sel.any():
            out.append((float(conf[sel].mean()), float((pred[sel] == y[sel]).mean()), int(sel.sum())))
    return out


def confidence_metrics(y, p_bot, prefix: str = "") -> dict:
    return {prefix + "ece": ece(y, p_bot), prefix + "brier": brier(y, p_bot),
            prefix + "risk_at_80": selective_risk(y, p_bot, 0.8),
            prefix + "risk_at_90": selective_risk(y, p_bot, 0.9)}
