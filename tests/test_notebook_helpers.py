"""Notebook helpers: context, walkthrough steps 8-10 (no PyTorch needed), reproducibility record."""
import json
import os

import numpy as np
import pytest


def _setup(tmp_path, monkeypatch, cresci_root, cresci2015_root, c15=True):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(tmp_path)
    os.makedirs("config")
    cfg = json.load(open(os.path.join(repo, "config", "kaggle_cpu.json")))
    cfg.update(input_dir=str(tmp_path / "none"), work_dir=str(tmp_path), smoke_test=True,
               paths={"cresci2017": cresci_root[0], "cresci2015": cresci2015_root[0] if c15 else None})
    json.dump(cfg, open("config/kaggle_cpu.json", "w"))
    from botrecon.notebook import ctx, start_session
    start_session(40)
    return ctx()


def test_ctx_and_budget(tmp_path, monkeypatch, cresci_root, cresci2015_root):
    c = _setup(tmp_path, monkeypatch, cresci_root, cresci2015_root, c15=False)
    assert c.available == ["cresci2017"] and c.walk_dataset == "cresci2017"
    assert 39 <= c.left() <= 40 and c.left(reserve=5) <= 35
    assert json.load(open("results/paths.json"))["cresci2017"] == cresci_root[0]


@pytest.mark.parametrize("c15", [True, False])
def test_walkthrough_steps_without_torch(tmp_path, monkeypatch, cresci_root, cresci2015_root, c15):
    from botrecon import walkthrough as W
    c = _setup(tmp_path, monkeypatch, cresci_root, cresci2015_root, c15)
    md = W.step_metadata(c)
    assert md["scaler_fitted_on_rows"] == md["train_rows"] and md["non_finite_values"] == 0
    assert np.allclose(md["train_mean_after_scaling"], 0, atol=1e-3)
    tx = W.step_text(c)
    assert tx["smoke_test_encoder"] is True and tx["zero_rows_equal_empty_texts"]
    g = W.step_graph(c)
    assert g["self_loops_in_edge_list"] == 0 and g["nodes"] > 0
    assert ("kNN" in g["graph"]) == (c.walk_dataset == "cresci2017")
    for f in ("08_metadata.json", "09_text.json", "10_graph.json"):
        assert os.path.exists(os.path.join("results", "walkthrough", f))


def test_reproducibility_record_lists_pending_runs(tmp_path, monkeypatch, cresci_root, cresci2015_root):
    from botrecon.repro_report import write
    c = _setup(tmp_path, monkeypatch, cresci_root, cresci2015_root, c15=False)
    text = write(c)
    assert "Still pending" in text and "cresci2017 / transductive" in text and "RGF (proposed)" in text
    assert "cresci2015 /" not in text
