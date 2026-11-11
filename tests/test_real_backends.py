"""Tests for the real PyTorch GCN and LightGBM stages.

SKIPPED where PyTorch/LightGBM are not installed (as in the environment this
package was written in). Run them first on Colab/Kaggle:  pytest -q tests/test_real_backends.py
"""
from dataclasses import replace

import numpy as np
import pytest

from botrecon import config as C
from botrecon import graph as G


def _two_cluster_graph(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    x = rng.normal(size=(n, 8)) + y[:, None] * 1.5
    gp = G.transductive_knn(x, 5)
    return x.astype(np.float32), y, gp


def test_gcn_forward_matches_dense_formula():
    torch = pytest.importorskip("torch")
    from botrecon.gcn import train_gcn
    x, y, gp = _two_cluster_graph()
    a = G.normalised_adjacency(gp.train_edges, len(y))
    cfg = replace(C.corrected("cresci2017"), gcn_epochs=1, gcn_selection="fixed", gcn_lr=0.0)
    y_view = y.copy()
    res = train_gcn(x, a, a, y_view, np.arange(150), np.arange(150, 200), cfg, seed=0, device="cpu")
    # with lr = 0 the weights stay at their seeded initialisation; rebuild them and compare
    torch.manual_seed(0)
    l0 = torch.nn.Linear(8, 64, bias=False)
    h_ref = np.maximum(a.toarray() @ (x @ l0.weight.detach().numpy().T), 0)
    assert np.allclose(res.hidden, h_ref, atol=1e-4)
    assert np.allclose(res.prob.sum(1), 1, atol=1e-5)


def _sbm_graph(n=600, seed=0, p_in=0.03, p_out=0.002, shift=0.3, d=8):
    """Two-block stochastic block model whose label signal is mostly in the GRAPH:
    features alone are weak (logistic regression ~0.65-0.70 test accuracy), edges are
    ~94% same-class (mean degree ~9). A GCN can only score highly by propagating
    information along edges, which is exactly what this test must establish.

    Replaces the earlier two-cluster kNN graph (homophily 0.87, 40 test nodes), on which
    even a correct GCN scores 0.825-0.975 depending on the seed: an independent NumPy
    replica of this GCN reproduced the 0.825 seen on Kaggle, and plain logistic regression
    on the raw features scored 0.975 there. That test measured the dataset, not the model."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    prob = np.where(y[:, None] == y[None, :], p_in, p_out)
    s, t = np.nonzero(np.triu(rng.random((n, n)) < prob, 1))
    edges = G.symmetrise(np.array([s, t]))
    x = (rng.normal(size=(n, d)) + y[:, None] * shift).astype(np.float32)
    perm = np.random.default_rng(100 + seed).permutation(n)
    # 50% train / ~17% validation / ~33% test at any n (n = 600 gives the original 300 / 100 / 200).
    # Fixed indices 300/400 left the validation and test sets EMPTY for n <= 300 (Kaggle failure of
    # test_gcn_seed_reproducibility: 'validation set contains masked labels' on an empty set).
    a, b = n // 2, n // 2 + n // 6
    return x, y, edges, perm[:a], perm[a:b], perm[b:]


@pytest.mark.parametrize("graph_seed", [0, 1])
def test_gcn_learns_and_respects_selection(graph_seed):
    pytest.importorskip("torch")
    from botrecon.gcn import train_gcn
    x, y, edges, tr, va, te = _sbm_graph(seed=graph_seed)
    n = len(y)
    a = G.normalised_adjacency(edges, n)
    a_none = G.normalised_adjacency(np.zeros((2, 0), np.int64), n)     # self-loops only: no message passing
    y_view = y.copy(); y_view[te] = -1
    cfg = C.corrected("cresci2017")
    res = train_gcn(x, a, a, y_view, tr, va, cfg, seed=0, device="cpu")
    acc = ((res.prob[te, 1] >= 0.5) == y[te]).mean()
    res_none = train_gcn(x, a_none, a_none, y_view, tr, va, cfg, seed=0, device="cpu")
    acc_none = ((res_none.prob[te, 1] >= 0.5) == y[te]).mean()
    # 1. the GCN learns (NumPy replica: 0.985-1.000 over 15 runs; 200 test nodes)
    assert acc >= 0.93, acc
    # 2. ...and it does so through the graph: the same model without edges stays near feature-only level
    assert acc - acc_none >= 0.15, (acc, acc_none)
    # 3. validation-loss selection returns the epoch with the lowest validation loss
    assert 1 <= res.best_epoch <= cfg.gcn_epochs
    assert res.val_loss_selected <= res.val_loss_last + 1e-6
    # 4. shapes are the two clearly separated outputs
    assert res.hidden.shape == (n, cfg.gcn_hidden) and res.prob.shape == (n, 2)
    # 5. masked (test) labels can never be used for fitting
    with pytest.raises(AssertionError):
        train_gcn(x, a, a, y_view, te, va, cfg, seed=0, device="cpu")


def test_gcn_seed_reproducibility():
    pytest.importorskip("torch")
    from botrecon.gcn import train_gcn
    x, y, edges, tr, va, te = _sbm_graph(n=300, seed=3)
    assert len(va) > 0 and len(te) > 0
    a = G.normalised_adjacency(edges, len(y))
    y_view = y.copy(); y_view[te] = -1
    cfg = C.corrected("cresci2017")
    r1 = train_gcn(x, a, a, y_view, tr, va, cfg, seed=7, device="cpu")
    r2 = train_gcn(x, a, a, y_view, tr, va, cfg, seed=7, device="cpu")
    r3 = train_gcn(x, a, a, y_view, tr, va, cfg, seed=8, device="cpu")
    assert np.allclose(r1.prob, r2.prob) and r1.best_epoch == r2.best_epoch     # same seed, same result
    assert not np.allclose(r1.hidden, r3.hidden)                                 # different seed differs


def test_gcn_gradients_update_every_parameter():
    torch = pytest.importorskip("torch")
    from botrecon import nn as N
    from botrecon.inputs import Inputs
    x, y, edges, tr, va, te = _sbm_graph(n=200, seed=4)
    inp = Inputs(x[:, :4], x[:, 4:], np.ones(len(y), bool), edges)
    model = N.GCN(8)
    B = N.Bundle(inp, "cpu")
    before = [p.detach().clone() for p in model.parameters()]
    out = model(B)
    loss = torch.nn.functional.cross_entropy(out["logits"][torch.as_tensor(tr)], torch.as_tensor(y[tr]))
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    torch.optim.AdamW(model.parameters(), lr=0.01).step()
    assert all(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_lightgbm_stage():
    pytest.importorskip("lightgbm")
    from botrecon.classifier import LGBMStage
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)); y = (X[:, 0] + 0.3 * rng.normal(size=600) > 0).astype(int)
    m = LGBMStage(C.corrected("twibot20"), 0).fit(X[:400], y[:400], X[400:500], y[400:500])
    assert 1 <= m.best_iteration <= 1000
    p = m.predict_proba(X[500:])
    assert ((p >= 0.5) == y[500:]).mean() > 0.85


# ---------------------------------------------------------------- suite models (run on Kaggle)

def _bundle_data(n=240, seed=0, isolated=40, no_text=60):
    from botrecon.inputs import Inputs
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    m = (rng.normal(size=(n, 4)) + y[:, None] * 1.2).astype(np.float32)
    t = (rng.normal(size=(n, 16)) + y[:, None] * 0.8).astype(np.float32)
    avail = np.ones(n, bool); avail[:no_text] = False; t[~avail] = 0
    src, dst = [], []
    for i in range(isolated, n):                       # homophilous edges; first `isolated` nodes have none
        same = np.where((y == y[i]) & (np.arange(n) >= isolated))[0]
        for j in rng.choice(same, 4):
            if j != i:
                src.append(i); dst.append(j)
    d = np.array([src, dst])
    inp = Inputs(m, t, avail, G.symmetrise(d), d)
    perm = rng.permutation(n)
    return inp, y, perm[:140], perm[140:180], perm[180:]


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat", "rgcn"])
def test_gnn_baselines_learn(kind):
    pytest.importorskip("torch")
    from botrecon.methods import GNN
    from botrecon.splits import Split
    inp, y, tr, va, te = _bundle_data()
    yv = y.copy(); yv[te] = -1
    m = GNN(kind)
    info = m.fit(inp, yv, Split(tr, va, te, "t"), seed=0)
    p = m.predict(inp)["p"]
    assert p.shape == (len(y),) and np.isfinite(p).all()
    assert ((p[te] >= 0.5) == y[te]).mean() > 0.8, kind
    assert info["size"] > 0


def test_rgf_gating_and_masks():
    torch = pytest.importorskip("torch")
    from botrecon import nn as N
    inp, y, tr, va, te = _bundle_data()
    yv = y.copy(); yv[te] = -1
    model = N.RGF(4, 16)
    B = N.Bundle(inp, "cpu")
    N.fit(model, B, yv, tr, va, seed=0, lr=0.005, epochs=150, patience=30)
    T = N.fit_temperature(model, B, yv, va)
    assert 0.25 <= T <= 5.0
    out = N.predict(model, B)
    w = out["weights"]
    assert np.allclose(w.sum(1), 1, atol=1e-5)
    assert np.allclose(w[~inp.text_avail, 1], 0)                 # no text -> text expert unused
    assert np.allclose(w[inp.degree() == 0, 2], 0)               # isolated -> graph expert unused
    assert ((out["p"][te] >= 0.5) == y[te]).mean() > 0.8
    with pytest.raises(AssertionError):
        N.fit(N.RGF(4, 16), B, yv, te, va, seed=0)               # masked labels refused
    m2 = N.RGF(4, 16, experts=("t", "g"))                        # metadata expert excluded
    m2.eval()
    with torch.no_grad():
        o = m2(B)
    assert torch.allclose(o["weights"][:, 0], torch.zeros(len(y)))
    none = (~inp.text_avail) & (inp.degree() == 0)
    if none.any():                                               # no usable expert -> p = 0.5
        p = torch.softmax(o["logits"], 1)[:, 1].numpy()
        assert np.allclose(p[none], 0.5, atol=1e-6)


def test_every_method_end_to_end_on_synthetic(cresci2015_root, cresci_root, tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("lightgbm"); pytest.importorskip("xgboost")
    import argparse
    from botrecon.data import load_cresci2015, load_cresci2017
    from botrecon.methods import make_all
    from botrecon.suite import read_rows, run_suite
    from botrecon.suite_report import build
    from botrecon.text import HashingTestEncoder

    def loader(ds, root, cfg):
        return (load_cresci2015 if ds == "cresci2015" else load_cresci2017)(root, load_tweets=True)

    for ds, root, other in (("cresci2015", cresci2015_root[0], cresci_root[0]),
                            ("cresci2017", cresci_root[0], cresci2015_root[0])):
        a = argparse.Namespace(dataset=ds, data_root=root, cross_root=other, seeds=[0], robust_seeds=[0],
                               settings=["transductive", "inductive"], methods=None, text="B", out_dir=str(tmp_path),
                               cache_dir=str(tmp_path / "c"), device="cpu", max_minutes=None, no_resume=False, smoke=True)
        run_suite(a, loader, lambda cfg: HashingTestEncoder(), make_all)
    main = read_rows(tmp_path / "suite_main.jsonl")
    expected = len(make_all("cresci2015", "transductive")) + len(make_all("cresci2015", "inductive")) + \
        len(make_all("cresci2017", "transductive")) + len(make_all("cresci2017", "inductive"))
    assert len(main) == expected
    assert main.test_f1_bot.notna().all()
    assert (main.provenance == "SMOKE-TEST-NOT-A-RESULT").all()
    build(str(tmp_path), str(tmp_path / "paper"))



def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("lightgbm"); pytest.importorskip("xgboost")
    from botrecon.methods import GNN, Hybrid, RGFMethod, Tabular
    from botrecon.splits import Split
    inp, y, tr, va, te = _bundle_data()
    yv = y.copy(); yv[te] = -1
    split = Split(tr, va, te, "t")
    for make in (lambda: GNN("gcn"), lambda: GNN("gcn", "meta"), lambda: GNN("rgcn"), lambda: Hybrid("P1", "hidden"),
                 lambda: Hybrid("P3", "prob"), lambda: RGFMethod(), lambda: RGFMethod("RGF w/o gate (mean of experts)"),
                 lambda: Tabular("lgbm", "meta+text"), lambda: Tabular("xgb", "meta+text")):
        m = make()
        m.fit(inp, yv, split, seed=0)
        p = m.predict(inp)["p"]
        path = str(tmp_path / m.name.replace(" ", "_").replace("/", "-"))
        assert m.save(path), m.name
        p2 = make().load(path).predict(inp)["p"]
        assert np.allclose(p, p2, atol=1e-5), m.name


def test_feature_subset_gcn_uses_only_its_features():
    pytest.importorskip("torch")
    from botrecon.methods import GNN
    from botrecon.splits import Split
    inp, y, tr, va, te = _bundle_data()
    yv = y.copy(); yv[te] = -1
    m = GNN("gcn", "meta")
    m.fit(inp, yv, Split(tr, va, te, "t"), seed=0)
    assert m.d == inp.m.shape[1]
    changed = inp.copy(); changed.t[:] = 123.0                  # text must not matter to Metadata + GCN
    assert np.allclose(m.predict(inp)["p"], m.predict(changed)["p"])


def test_walkthrough_gcn_lightgbm_evaluation(tmp_path, monkeypatch, cresci_root, cresci2015_root):
    pytest.importorskip("torch"); pytest.importorskip("lightgbm")
    import json, os
    from botrecon import walkthrough as W
    from botrecon.notebook import ctx, start_session
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(tmp_path); os.makedirs("config")
    cfg = json.load(open(os.path.join(repo, "config", "kaggle_cpu.json")))
    cfg.update(input_dir=str(tmp_path), work_dir=str(tmp_path), smoke_test=True,
               paths={"cresci2017": cresci_root[0], "cresci2015": cresci2015_root[0]})
    json.dump(cfg, open("config/kaggle_cpu.json", "w")); start_session(40)
    c = ctx()
    W.step_metadata(c); W.step_text(c); W.step_graph(c)
    g = W.step_gcn(c)
    assert g["val_loss_selected"] <= g["val_loss_last"] + 1e-6
    e = W.step_embeddings(c)
    assert e["hidden_embedding"]["shape"][1] == 64 and e["class_probabilities"]["shape"][1] == 2
    lg = W.step_lightgbm(c)
    d_m, d_t = 4, 32                                          # core metadata features, hashing test encoder
    dims = {k[:2]: v["input_dim"] for k, v in lg.items()}
    assert dims == {"P3": 64, "P1": d_m + d_t + 64}
    ev = W.step_evaluate(c)
    assert ev["label"].startswith("WALKTHROUGH") and 0 <= ev["P1"]["f1_bot"] <= 1
