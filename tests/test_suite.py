"""Suite plumbing, perturbations and calibration, with numpy stand-ins for the models."""
import argparse
import os

import numpy as np
import pandas as pd
import pytest

from botrecon import provenance as P
from botrecon.calib import brier, ece, reliability_curve, selective_risk
from botrecon.data import load_cresci2015, load_cresci2017, load_twibot20
from botrecon.graph import symmetrise
from botrecon.inputs import Inputs, drop_edges, metadata_mimicry, missing_text, noisy_edges
from botrecon.text import HashingTestEncoder


def _inp(n=50, seed=0):
    rng = np.random.default_rng(seed)
    e = symmetrise(rng.integers(0, n, (2, 150)))
    d = rng.integers(0, n, (2, 100)); d = d[:, d[0] != d[1]]
    return Inputs(rng.normal(size=(n, 4)).astype(np.float32), rng.normal(size=(n, 8)).astype(np.float32),
                  np.ones(n, bool), e, d)


def test_perturbations_touch_only_test_nodes():
    inp, test = _inp(), np.arange(30, 50)
    rng = np.random.default_rng(0)
    out = missing_text(inp, test, 0.5, rng)
    assert (~out.text_avail).sum() == 10 and (~out.text_avail[:30]).sum() == 0
    assert np.allclose(out.t[:30], inp.t[:30]) and np.allclose(inp.t, _inp().t)      # original untouched
    out = drop_edges(inp, test, 1.0, rng)
    m = np.zeros(50, bool); m[test] = True
    assert not (m[out.sym_edges[0]] | m[out.sym_edges[1]]).any()                       # all test edges gone
    keep = ~(m[inp.sym_edges[0]] | m[inp.sym_edges[1]])
    assert out.sym_edges.shape[1] == keep.sum()                                        # others intact
    assert not (m[out.directed[0]] | m[out.directed[1]]).any()
    out = noisy_edges(inp, test, 0.5, rng)
    assert out.sym_edges.shape[1] > inp.sym_edges.shape[1]
    added = set(map(tuple, out.sym_edges.T)) - set(map(tuple, inp.sym_edges.T))
    assert all(a in test or b in test for a, b in added)
    y = (np.arange(20) % 2).astype(int)
    out = metadata_mimicry(inp, test, y, 1.0, np.zeros(4), rng)
    assert np.allclose(out.m[test[y == 1]], 0) and np.allclose(out.m[test[y == 0]], inp.m[test[y == 0]])
    assert np.allclose(missing_text(inp, test, 0.0, rng).t, inp.t)


def test_calibration_metrics():
    y = np.array([1, 1, 0, 0])
    assert ece(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0)
    assert ece(y, np.array([0.6, 0.6, 0.4, 0.4])) == pytest.approx(0.4)
    assert brier(y, np.array([1, 1, 0, 0])) == 0
    p = np.array([0.99, 0.6, 0.45, 0.01])
    assert selective_risk(np.array([1, 0, 1, 0]), p, 0.5) == 0.0
    assert selective_risk(np.array([1, 0, 1, 0]), p, 1.0) == 0.5
    assert len(reliability_curve(y, p)) >= 2


class StubMethod:
    """Numpy stand-in: logistic regression on [m, t*avail, degree-feature]."""

    def __init__(self, name, group="gnn"):
        self.name, self.group, self.provenance, self.needs_directed = name, group, P.THIS_WORK, False

    def _X(self, inp):
        deg = np.log1p(inp.degree())[:, None]
        return np.hstack([inp.m, inp.t * inp.text_avail[:, None], deg])

    def fit(self, inp, y_view, split, seed, device="cpu"):
        from sklearn.linear_model import LogisticRegression
        assert (y_view[split.train] >= 0).all() and (y_view[split.test] == -1).all()
        self.clf = LogisticRegression(max_iter=300).fit(self._X(inp)[split.train], y_view[split.train])
        return {"seconds": 0.0, "size": 10, "best_epoch": 1}

    def predict(self, inp):
        p = self.clf.predict_proba(self._X(inp))[:, 1]
        out = {"p": p}
        if self.name == "RGF (proposed)":
            w = np.tile([0.5, 0.3, 0.2], (inp.n, 1))
            out.update(weights=w, expert_p={"m": p, "t": p, "g": p}, p_uncalibrated=np.clip(p * 1.1, 0, 1))
        return out

    def importances(self):
        return np.abs(self.clf.coef_[0][:-1])


def factory(dataset, setting):
    ms = [StubMethod("RGF (proposed)", "proposed"), StubMethod("GCN"), StubMethod("LightGBM (meta+text)", "tabular"),
          StubMethod("Reconstructed hybrid (P3, hidden)", "hybrid"), StubMethod("RGF w/o gate (mean of experts)", "ablation")]
    return ms if setting == "transductive" else ms[:2]


def _args(tmp, ds, root, cross=None, **kw):
    a = argparse.Namespace(dataset=ds, data_root=root, cross_root=cross, seeds=[0, 1, 2], robust_seeds=[0],
                           settings=["transductive", "inductive"], methods=None, text="B", out_dir=str(tmp),
                           cache_dir=str(tmp / "cache"), device="cpu", max_minutes=None, no_resume=False, smoke=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _loader(ds, root, cfg):
    return {"twibot20": load_twibot20, "cresci2015": load_cresci2015,
            "cresci2017": load_cresci2017}[ds](root, load_tweets=True)


def test_suite_end_to_end_and_report(cresci2015_root, cresci_root, tmp_path):
    from botrecon.suite import read_rows, run_suite
    from botrecon.suite_report import build
    enc = lambda cfg: HashingTestEncoder()
    n = run_suite(_args(tmp_path, "cresci2015", cresci2015_root[0], cresci_root[0]), _loader, enc, factory)
    assert n == 5 * 3 + 2 * 3
    run_suite(_args(tmp_path, "cresci2017", cresci_root[0], cresci2015_root[0]), _loader, enc, factory)
    main = read_rows(tmp_path / "suite_main.jsonl")
    assert len(main) == 2 * (15 + 6)
    assert (main.provenance == P.THIS_WORK).all()
    rob = read_rows(tmp_path / "suite_robust.jsonl")
    assert set(rob.perturbation) == {"missing_text", "drop_edges", "noisy_edges", "metadata_mimicry"}
    assert set(rob.seed) == {0}                                     # robustness only on robust seeds
    assert "Reconstructed hybrid (P3, hidden)" in set(rob.method)
    cross = read_rows(tmp_path / "suite_cross.jsonl")
    assert set(zip(cross.source, cross.target)) == {("cresci2015", "cresci2017"), ("cresci2017", "cresci2015")}
    assert (cross.overlap_removed >= 0).all() and (cross.target_bots + cross.target_humans == cross.n).all()
    ex = read_rows(tmp_path / "suite_explain.jsonl")
    assert {"gate_weight", "expert_alone", "permutation_importance", "lgbm_split_importance"} <= set(ex.kind)
    # resume: nothing re-run
    assert run_suite(_args(tmp_path, "cresci2015", cresci2015_root[0], cresci_root[0]), _loader, enc, factory) == 0
    md = build(str(tmp_path), str(tmp_path / "paper"))
    for t in ("T2 Main comparison", "T3 Ablation", "T4 Robustness", "T5 Unseen", "T6 Calibration",
              "T7 Cross-dataset", "T8a", "T8c", "T9 Efficiency", "T10 Paired"):
        assert t in md, t
    # new paper: no numbers from the earlier manuscript or TwiBot-20 literature in the tables
    assert P.MANUSCRIPT_REPORTED not in md and "92.28" not in md and "98.57" not in md and "TwiBot" not in md
    assert "Cresci-2015" in md and "Shared accounts removed" in md
    assert os.path.exists(tmp_path / "paper" / "paper_facts.json")
    assert any(f.startswith("fig_robustness") for f in os.listdir(tmp_path / "paper"))
    assert any(f.startswith("fig_curves") for f in os.listdir(tmp_path / "paper"))


def test_suite_time_budget(twibot_root, tmp_path):
    from botrecon.suite import run_suite
    n = run_suite(_args(tmp_path, "twibot20", twibot_root[0], max_minutes=0), _loader,
                  lambda cfg: HashingTestEncoder(), factory)
    assert n == 0


def test_inductive_prepared_hides_test_nodes(cresci_root):
    from botrecon import config as C
    from botrecon.suite import Prepared
    data = load_cresci2017(cresci_root[0])
    emb = HashingTestEncoder().encode(list(data.description))
    prep = Prepared(data, emb, C.corrected("cresci2017"), 0, "inductive")
    test = set(prep.split.test)
    e = prep.train_inp.sym_edges
    assert not (set(e[0]) | set(e[1])) & test
    s, t = prep.infer_inp.sym_edges
    assert not any(a in test and b not in test for a, b in zip(s, t))   # test nodes send nothing to seen nodes
    assert (prep.y_view[prep.split.test] == -1).all()


def test_checkpoints_saved_only_for_requested_seeds(twibot_root, tmp_path):
    from botrecon.suite import read_rows, run_suite
    saved = []

    class Savable(StubMethod):
        def save(self, path):
            saved.append(path); open(path + ".stub", "w").write("x"); return True

    def fac(dataset, setting):
        return [Savable("RGF (proposed)", "proposed"), Savable("GCN")]

    a = _args(tmp_path, "twibot20", twibot_root[0], settings=["transductive"], seeds=[0, 1])
    a.checkpoint_seeds = [1]
    run_suite(a, _loader, lambda cfg: HashingTestEncoder(), fac)
    assert len(saved) == 2 and all(p.endswith("_1") for p in saved)
    main = read_rows(tmp_path / "suite_main.jsonl")
    assert main[main.seed == 1].checkpoint.notna().all()
    assert main[main.seed == 0].get("checkpoint", pd.Series([None] * 2)).isna().all()


def test_modality_table_rows():
    from botrecon.suite_report import modality_table
    rows = []
    for name in ("LightGBM (meta)", "GCN (text)", "Reconstructed hybrid (P1, hidden)", "RGF (proposed)", "Unlisted"):
        for seed in range(2):
            rows.append({"dataset": "twibot20", "setting": "transductive", "method": name, "seed": seed,
                         "test_f1_bot": 0.8, "test_mcc": 0.6, "test_roc_auc": 0.9})
    t = modality_table(pd.DataFrame(rows), "twibot20")
    assert list(t.Method) == ["LightGBM (meta)", "GCN (text)", "Reconstructed hybrid (P1, hidden)", "RGF (proposed)"]
    assert t.set_index("Method").loc["Reconstructed hybrid (P1, hidden)", "Architecture"].startswith("P1")


@pytest.mark.parametrize("which", ["cresci", "cresci2015", "twibot"])
def test_protocol_audit_passes_on_synthetic_releases(which, cresci_root, cresci2015_root, twibot_root):
    from botrecon.audit import protocol_audit
    data = {"cresci": lambda: load_cresci2017(cresci_root[0]), "cresci2015": lambda: load_cresci2015(cresci2015_root[0]),
            "twibot": lambda: load_twibot20(twibot_root[0], load_tweets=True)}[which]()
    rows = protocol_audit(data)
    failed = [r for r in rows if r["status"] == "FAIL"]
    # the synthetic Cresci folders have 30 accounts per group, so only the published-size check may fail
    assert all("published sizes" in r["check"] for r in failed), failed
    assert any("scaler fitted on training rows only" in r["check"] and r["status"] == "PASS" for r in rows)


def test_protocol_audit_detects_literal_label_rule(cresci_root):
    from botrecon.audit import label_audit
    rows = {r["check"]: r["status"] for r in label_audit(load_cresci2017(cresci_root[0], "literal_bot_substring"))}
    assert rows["Cresci-2017: fake_followers labelled bot"] == "FAIL"
    assert rows["Cresci-2017: only genuine_accounts labelled human (explicit map)"] == "FAIL"


def test_cross_dataset_removes_shared_accounts(cresci_root, cresci2015_root):
    """The Cresci-2015 fake followers re-appear in Cresci-2017; the cross test must not score them."""
    from botrecon import config as C
    from botrecon.suite import Prepared, target_inputs
    src, tgt = load_cresci2017(cresci_root[0]), load_cresci2015(cresci2015_root[0])
    cfg = C.corrected("cresci2017")
    prep = Prepared(src, np.zeros((src.n, 4), np.float32), cfg, 0, "transductive")
    _, test, n_shared = target_inputs(tgt, np.zeros((tgt.n, 4), np.float32), prep, cfg, 0)
    assert not np.isin(tgt.ids[test], src.ids).any()
    full = C.corrected("cresci2015")
    from botrecon.splits import make_split
    assert n_shared == int(np.isin(tgt.ids[make_split(tgt, full, 0).test], src.ids).sum())
