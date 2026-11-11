"""Classification metrics. Bot (label 1) is the positive class.

Averaging conventions (stated explicitly because the original manuscript mixed them):
  *_bot      : bot class only (binary, pos_label=1). F1 = 2PR/(P+R) holds exactly.
  *_macro    : unweighted mean over the two classes (F1 is the mean of per-class F1, so it is
               NOT the harmonic mean of macro P and macro R).
  *_weighted : support-weighted mean over classes; weighted recall always equals accuracy.

Headline: bot-class precision/recall/F1 and MCC. Weighted averages are recorded
only to compare with the manuscript, whose numbers appear to be weighted
(weighted recall always equals accuracy).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)


def evaluate(y_true: np.ndarray, p_bot: np.ndarray, threshold: float = 0.5, prefix: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    assert set(np.unique(y_true)).issubset({0, 1})
    y_pred = (np.asarray(p_bot) >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two = len(np.unique(y_true)) == 2
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_bot": p, "recall_bot": r, "f1_bot": f,
        "precision_macro": mp, "recall_macro": mr, "f1_macro": mf,
        "precision_weighted": wp, "recall_weighted": wr, "f1_weighted": wf,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, p_bot) if two else float("nan"),
        "pr_auc": average_precision_score(y_true, p_bot) if two else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y_true)),
    }
    return {prefix + k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def check_consistency(m: dict, prefix: str = "", tol: float = 1e-9) -> None:
    """Raise if the reported numbers are internally inconsistent."""
    g = lambda k: m[prefix + k]
    tp, fp, fn, tn = g("tp"), g("fp"), g("fn"), g("tn")
    assert tp + fp + fn + tn == g("n")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    assert abs(p - g("precision_bot")) < tol and abs(r - g("recall_bot")) < tol and abs(f - g("f1_bot")) < tol
    assert abs((tp + tn) / g("n") - g("accuracy")) < tol
    assert abs(g("recall_weighted") - g("accuracy")) < 1e-9
