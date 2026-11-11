"""Notebook steps 8-14: the corrected reconstructed pipeline, step by step, on ONE dataset and ONE seed.

Same procedure as methods.Hybrid in the suite (GCN on training half A with validation-loss selection;
LightGBM on half B, early-stopped on validation), shown explicitly so each tensor and dimension can be
inspected. Results are labelled WALKTHROUGH and are NOT reported results (single seed).
Every step reads its inputs from disk (results/walkthrough/), so cells are restart-safe.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from . import graph as G
from .features import raw_features
from .metrics import check_consistency, evaluate
from .splits import blend_halves
from .text import cached_encode, empty_rates, make_encoder, text_inputs

OUT = "results/walkthrough"
LABEL = "WALKTHROUGH (single seed, corrected protocol; not a reported result)"


def _cfg(ctx):
    return C.corrected(ctx.walk_dataset, text=ctx.cfg["text_variant"])


def _data(ctx):
    from .run import cached_load
    cfg = _cfg(ctx)
    return cached_load(cfg, ctx.paths[ctx.walk_dataset], "cache", cfg.text == "A")


def _emb(ctx, data):
    cfg = _cfg(ctx)
    texts = text_inputs(data, cfg.text, cfg.max_tweets)
    enc = make_encoder(cfg, "cpu", test_encoder=ctx.smoke)
    return texts, cached_encode(enc, texts, "cache", f"suite_{data.name}_{cfg.text}"), enc.name


def _prep(ctx):
    from .suite import Prepared
    data = _data(ctx)
    texts, emb, _ = _emb(ctx, data)
    return data, Prepared(data, emb, _cfg(ctx), ctx.walk_seed, "transductive",
                          text_avail=[len(t.strip()) > 0 for t in texts])


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    return obj


def step_metadata(ctx) -> dict:
    data, p = _prep(ctx)
    m = p.train_inp.m
    names = list(raw_features(data.meta.iloc[:2]).columns)
    tr = p.split.train
    return _save("08_metadata.json", {
        "dataset": data.name, "features": names, "shape": list(m.shape),
        "scaler_fitted_on_rows": int(p.transformer.fit_idx_size_), "train_rows": int(len(tr)),
        "train_mean_after_scaling": np.round(m[tr].mean(0), 4).tolist(),
        "train_sd_after_scaling": np.round(m[tr].std(0), 4).tolist(),
        "non_finite_values": int((~np.isfinite(m)).sum())})


def step_text(ctx) -> dict:
    data = _data(ctx)
    texts, emb, name = _emb(ctx, data)
    return _save("09_text.json", {"encoder": name, "variant": _cfg(ctx).text, "shape": list(emb.shape),
                                  "smoke_test_encoder": ctx.smoke, **empty_rates(texts, data.y),
                                  "zero_rows_equal_empty_texts": int((np.abs(emb).sum(1) == 0).sum()) ==
                                  int(sum(len(t.strip()) == 0 for t in texts))})


def step_graph(ctx) -> dict:
    data, p = _prep(ctx)
    e = p.train_inp.sym_edges
    deg = p.infer_inp.degree()
    kind = "follow edges (follower + following, undirected)" if data.edges is not None else \
        f"kNN, k={_cfg(ctx).k}, cosine on standardised metadata, symmetrised"
    return _save("10_graph.json", {"graph": kind, "setting": "transductive (test accounts' features and edges "
                                   "present, labels hidden)", "nodes": data.n, "undirected_edges": int(e.shape[1] // 2),
                                   "self_loops_in_edge_list": int((e[0] == e[1]).sum()),
                                   "isolated_nodes": int((deg == 0).sum()),
                                   "edge_homophily_train_nodes": p.log["edge_homophily_train"]})


def step_gcn(ctx) -> dict:
    from .gcn import train_gcn
    data, p = _prep(ctx)
    cfg = _cfg(ctx)
    a_half, b_half = blend_halves(p.split.train, p.y_view, ctx.walk_seed)
    n = data.n
    A_tr = G.normalised_adjacency(p.train_inp.sym_edges, n)
    A_in = G.normalised_adjacency(p.infer_inp.sym_edges, n)
    res = train_gcn(p.train_inp.x, A_tr, A_in, p.y_view, a_half, p.split.val, cfg, ctx.walk_seed, "cpu")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "11_gcn.npz"), hidden=res.hidden, prob=res.prob, half_b=b_half)
    return _save("11_gcn.json", {"architecture": f"GCN({p.train_inp.x.shape[1]} -> {cfg.gcn_hidden}) -> ReLU -> "
                                 f"Dropout({cfg.gcn_dropout}) -> GCN({cfg.gcn_hidden} -> 2)",
                                 "trained_on": f"training half A ({len(a_half)} accounts)",
                                 "selection": "lowest validation loss", "best_epoch": res.best_epoch,
                                 "val_loss_selected": res.val_loss_selected, "val_loss_last": res.val_loss_last,
                                 "seconds": res.seconds})


def step_embeddings(ctx) -> dict:
    z = np.load(os.path.join(OUT, "11_gcn.npz"))
    return _save("12_embeddings.json", {
        "hidden_embedding": {"shape": list(z["hidden"].shape), "meaning": "64-d node embedding after ReLU (eval mode)"},
        "class_probabilities": {"shape": list(z["prob"].shape), "meaning": "softmax of the second GCN layer",
                                "rows_sum_to_one": bool(np.allclose(z["prob"].sum(1), 1, atol=1e-5))}})


def step_lightgbm(ctx) -> dict:
    import lightgbm as lgb
    data, p = _prep(ctx)
    z = np.load(os.path.join(OUT, "11_gcn.npz"))
    b_half, val, test = z["half_b"], p.split.val, p.split.test
    variants = {"P3 (GCN 64-d embedding -> LightGBM)": z["hidden"],
                "P1 (metadata + text + GCN 64-d embedding -> LightGBM)": np.hstack([p.infer_inp.m, p.infer_inp.t, z["hidden"]])}
    out = {}
    for name, q in variants.items():
        clf = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=31, learning_rate=0.05, n_estimators=1000,
                                 objective="binary", random_state=ctx.walk_seed, verbose=-1)
        clf.fit(q[b_half], p.y_view[b_half], eval_set=[(q[val], p.y_view[val])], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        np.save(os.path.join(OUT, f"13_{name[:2]}_ptest.npy"), clf.predict_proba(q[test])[:, 1])
        np.save(os.path.join(OUT, f"13_{name[:2]}_pval.npy"), clf.predict_proba(q[val])[:, 1])
        out[name] = {"input_dim": int(q.shape[1]), "trained_on": f"training half B ({len(b_half)} accounts)",
                     "early_stopping": "validation set", "best_iteration": int(clf.best_iteration_ or 1000)}
    return _save("13_lightgbm.json", out)


def step_evaluate(ctx) -> dict:
    data, p = _prep(ctx)
    out = {"label": LABEL, "dataset": data.name, "seed": ctx.walk_seed}
    for v in ("P3", "P1"):
        pt = np.load(os.path.join(OUT, f"13_{v}_ptest.npy"))
        m = evaluate(data.y[p.split.test], pt, prefix="")
        check_consistency(m)
        out[v] = m
    return _save("14_evaluation.json", out)
