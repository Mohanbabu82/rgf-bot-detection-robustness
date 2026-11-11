"""Statistics for multi-seed results.

- mean_ci: mean, SD and 95% t-interval over seeds.
- paired_t: paired t-test on per-seed differences (assumes roughly normal differences).
  With 5 seeds do NOT use Wilcoxon signed-rank: its smallest two-sided p is 0.0625.
- mcnemar_exact: per-test-set comparison of two models' errors.
- paired_bootstrap_f1: CI of the F1 difference by resampling test accounts.
- holm: family-wise correction for several comparisons against one model.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st
from sklearn.metrics import f1_score


def mean_ci(values, level: float = 0.95) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        h = st.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)
    else:
        h = float("nan")
    return {"mean": m, "sd": sd, "ci_low": m - h, "ci_high": m + h, "n": n}


def paired_t(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    if sd == 0:                       # identical differences: no variance to test against
        return {"mean_diff": float(d.mean()), "t": float("nan"),
                "p": 1.0 if d.mean() == 0 else float("nan"), "cohen_dz": float("nan"), "n": len(d)}
    t, p = st.ttest_rel(a, b)
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(p),
            "cohen_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n": len(d)}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    ca, cb = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0}
    if b + c < 25:
        p = st.binomtest(b, b + c, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = st.chi2.sf(chi2, 1)
    return {"b": b, "c": c, "p": float(p)}


def paired_bootstrap_f1(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = f1_score(y_true[s], pred_a[s], zero_division=0) - f1_score(y_true[s], pred_b[s], zero_division=0)
    obs = f1_score(y_true, pred_a, zero_division=0) - f1_score(y_true, pred_b, zero_division=0)
    return {"diff": float(obs), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    # NaN p-values (untestable comparisons) are treated as 1.0: never significant
    items = sorted(((k, 1.0 if p != p else p) for k, p in pvalues.items()), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    return out
