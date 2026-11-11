"""Protocol tests with stub GCN/classifier (no PyTorch or LightGBM needed).
They check WHICH data each stage may use, not model quality."""
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from botrecon import config as C
from botrecon import provenance as P
from botrecon.data import load_cresci2017, load_twibot20
from botrecon.pipeline import run_one
from botrecon.splits import make_split
from botrecon.text import HashingTestEncoder, empty_rates, text_inputs


def _emb(data, variant):
    return HashingTestEncoder().encode(text_inputs(data, variant))


@pytest.fixture(scope="module")
def tw(twibot_root):
    return load_twibot20(twibot_root[0], load_tweets=True)


@pytest.fixture(scope="module")
def cr(cresci_root):
    return load_cresci2017(cresci_root[0], load_tweets=True)


def _check_label_discipline(data, cfg, seed, gcn, clf_log):
    split = make_split(data, cfg, seed)
    test = set(split.test)
    for call in gcn.calls:
        assert (call["y_view"][split.test] == -1).all(), "test labels visible to GCN"
        assert not (set(call["fit_idx"]) & test)
        assert not (set(call["fit_idx"]) & set(split.val)), "validation nodes used in GCN loss"
    return split


@pytest.mark.parametrize("dataset", ["twibot20", "cresci2017"])
@pytest.mark.parametrize("fusion,tap", [("P3", "hidden"), ("P3", "prob"), ("P1", "hidden"), ("P1", "prob")])
def test_corrected_blend(dataset, fusion, tap, tw, cr, stub_gcn, stub_clf):
    data = tw if dataset == "twibot20" else cr
    cfg = replace(C.corrected(dataset, fusion=fusion, tap=tap, text="B"), smoke_test=True)
    rec, preds = run_one(data, cfg, 0, _emb(data, "B"), stub_gcn, stub_clf)
    split = _check_label_discipline(data, cfg, 0, stub_gcn, stub_clf.log)
    assert rec["provenance"] == P.SMOKE_TEST
    # blending: GCN and LightGBM fitted on disjoint halves of the training set
    assert rec["n_gcn_fit"] + rec["n_lgbm_fit"] == len(split.train)
    assert stub_clf.log[-1]["n_es"] == len(split.val)
    d_tap = 64 if tap == "hidden" else 2
    assert rec["d_q"] == (d_tap if fusion == "P3" else rec["d_meta"] + rec["d_text"] + d_tap)
    assert len(preds["y"]) == len(split.test)
    assert 0 <= rec["test_f1_bot"] <= 1
    if dataset == "twibot20":
        assert rec["split_kind"] == "official"


def test_literal_insample_and_scaler_all(cr, stub_gcn, stub_clf):
    cfg = replace(C.literal("cresci2017", fusion="P3", tap="prob", text="A",
                            cresci_knn_features="all"), smoke_test=True)
    rec, _ = run_one(cr, cfg, 1, _emb(cr, "A"), stub_gcn, stub_clf)
    split = _check_label_discipline(cr, cfg, 1, stub_gcn, stub_clf.log)
    assert rec["n_gcn_fit"] == rec["n_lgbm_fit"] == len(split.train)   # in-sample stacking
    assert rec["scaler_fit"] == "all"


def test_oof_requires_prob_tap():
    with pytest.raises(ValueError):
        C.corrected("twibot20", stacking="oof", tap="hidden")


def test_oof_runs_with_prob_tap(tw, stub_gcn, stub_clf):
    cfg = replace(C.corrected("twibot20", stacking="oof", tap="prob"), smoke_test=True)
    rec, _ = run_one(tw, cfg, 0, _emb(tw, "B"), stub_gcn, stub_clf)
    assert len(stub_gcn.calls) == 5
    split = make_split(tw, cfg, 0)
    for call in stub_gcn.calls:                          # each fold model excludes its held-out fold
        assert len(call["fit_idx"]) < len(split.train)
    _check_label_discipline(tw, cfg, 0, stub_gcn, stub_clf.log)


@pytest.mark.parametrize("dataset", ["twibot20", "cresci2017"])
def test_inductive_training_graph_excludes_test(dataset, tw, cr, stub_gcn, stub_clf):
    data = tw if dataset == "twibot20" else cr
    cfg = replace(C.corrected(dataset, setting="inductive"), smoke_test=True)
    run_one(data, cfg, 0, _emb(data, "B"), stub_gcn, stub_clf)
    split = make_split(data, cfg, 0)
    a_tr = stub_gcn.calls[0]["a_train"].tocoo()
    touched = set(a_tr.row) | set(a_tr.col)
    assert not (touched & set(split.test)), "test nodes present in training graph"
    a_in = stub_gcn.calls[0]["a_infer"].tocsc()
    # columns of test nodes (their outgoing messages) reach no seen node
    seen = np.concatenate([split.train, split.val])
    sub = a_in[seen][:, split.test]
    assert sub.nnz == 0


def test_diagnostic_is_labelled_invalid(tw, stub_gcn, stub_clf):
    cfg = replace(C.diagnostic(C.literal("twibot20")), smoke_test=False)
    assert cfg.provenance == P.DIAGNOSTIC_INVALID
    rec, _ = run_one(tw, cfg, 0, _emb(tw, "B"), stub_gcn, stub_clf)
    split = make_split(tw, cfg, 0)
    assert stub_clf.log[-1]["n_es"] == len(split.test)
    assert rec["provenance"] == P.DIAGNOSTIC_INVALID


def test_provenance_labels():
    assert C.literal("twibot20").provenance == P.REPRODUCED_LITERAL
    assert C.corrected("twibot20").provenance == P.REPRODUCED_CORRECTED
    assert replace(C.corrected("twibot20"), smoke_test=True).provenance == P.SMOKE_TEST


def test_grid_sizes():
    assert len(list(C.literal_grid("cresci2017"))) == 32
    assert len(list(C.literal_grid("twibot20"))) == 16
    assert len(list(C.corrected_grid("twibot20"))) == 8


def test_empty_text_rates(cr):
    r = empty_rates(text_inputs(cr, "A"), cr.y)
    assert r["empty_text_rate_human"] == 0.0 and r["empty_text_rate_bot"] > 0   # trad spambots 2-4


def test_report_builds_and_keeps_provenance_apart(tw, stub_gcn, stub_clf, tmp_path):
    from botrecon.report import build
    rows = []
    for cfg in list(C.corrected_grid("twibot20"))[:3] + [C.literal("twibot20")]:
        for seed in range(3):
            rec, _ = run_one(tw, cfg, seed, _emb(tw, cfg.text if cfg.text == "B" else "A"), stub_gcn, stub_clf)
            rows.append(rec)
    smoke = dict(rows[0]); smoke["provenance"] = P.SMOKE_TEST; smoke["test_accuracy"] = 0.999
    rows.append(smoke)
    path = tmp_path / "runs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    md = build(str(path))
    assert P.MANUSCRIPT_REPORTED in md and P.REPRODUCED_CORRECTED in md and P.LITERATURE in md
    assert "99.90" not in md                         # smoke rows excluded
    assert "92.28" in md                             # manuscript row present, in its own labelled row
