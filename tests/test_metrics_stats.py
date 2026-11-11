import numpy as np
import pytest

from botrecon.metrics import evaluate
from botrecon.stats import holm, mcnemar_exact, mean_ci, paired_bootstrap_f1, paired_t


def test_bot_class_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1])       # tp=2 fn=1 fp=1 tn=1
    m = evaluate(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision_bot"] == pytest.approx(2 / 3)
    assert m["recall_bot"] == pytest.approx(2 / 3)
    assert m["f1_bot"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(0.6)


def test_weighted_recall_equals_accuracy():
    """The property that identifies the manuscript's averaging convention."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    m = evaluate(y, p)
    assert m["recall_weighted"] == pytest.approx(m["accuracy"])


def test_mean_ci_uses_t_distribution():
    r = mean_ci([0.80, 0.82, 0.81, 0.79, 0.83])
    assert r["mean"] == pytest.approx(0.81)
    half = 2.7764 * np.std([0.80, 0.82, 0.81, 0.79, 0.83], ddof=1) / np.sqrt(5)
    assert r["ci_high"] - r["mean"] == pytest.approx(half, rel=1e-3)


def test_paired_and_mcnemar_and_bootstrap_and_holm():
    t = paired_t([0.9, 0.91, 0.92, 0.9, 0.91], [0.85, 0.86, 0.88, 0.86, 0.85])
    assert t["mean_diff"] > 0 and t["p"] < 0.01
    y = np.array([1] * 50 + [0] * 50)
    a = y.copy(); b = y.copy(); b[:10] = 0
    assert mcnemar_exact(y, a, b)["b"] == 10
    assert mcnemar_exact(y, a, b)["p"] < 0.01
    bs = paired_bootstrap_f1(y, a, b, n_boot=300)
    assert bs["ci_low"] > 0
    h = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert h["x"]["p_holm"] == pytest.approx(0.03)
    assert h["z"]["p_holm"] == pytest.approx(0.06) and h["y"]["p_holm"] == pytest.approx(0.06)


def test_identical_results_are_not_significant():
    t = paired_t([0.9, 0.9, 0.9], [0.9, 0.9, 0.9])
    assert t["p"] == 1.0
    h = holm({"a": float("nan"), "b": 0.01})
    assert h["a"]["reject"] is False or h["a"]["reject"] == False
    assert h["a"]["p_holm"] == 1.0


def test_metric_definitions_are_consistent():
    from botrecon.metrics import check_consistency
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, 300); p = np.clip(rng.random(300) * 0.6 + y * 0.3, 0, 1)
        m = evaluate(y, p)
        check_consistency(m)
        for k in ("precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted",
                  "f1_weighted", "mcc", "roc_auc", "pr_auc"):
            assert k in m
    m = evaluate(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]))   # tp=1 fn=1 fp=1 tn=1
    assert m["precision_macro"] == pytest.approx(0.5) and m["recall_macro"] == pytest.approx(0.5)
    bad = dict(m); bad["f1_bot"] = 0.9
    with pytest.raises(AssertionError):
        check_consistency(bad)
