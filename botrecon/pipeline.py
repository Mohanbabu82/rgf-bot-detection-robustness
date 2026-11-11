"""One run = one (dataset, config, seed).

Label discipline: the pipeline builds y_view, a copy of the labels with every
test label replaced by -1. All fitting (GCN loss, GCN model selection,
LightGBM fit, LightGBM early stopping) reads y_view only. True test labels are
read in exactly two places: final evaluation, and LightGBM early stopping when
cfg.es_on == "test" (DIAGNOSTIC-INVALID runs only).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from . import graph as G
from .features import metadata_matrix
from .metrics import evaluate
from .splits import blend_halves, kfold, make_split


def _tap(res, tap: str) -> np.ndarray:
    return res.hidden if tap == "hidden" else res.prob


def build_graph(data, cfg, x: np.ndarray, m: np.ndarray, split) -> G.GraphPair:
    seen = np.sort(np.concatenate([split.train, split.val]))
    if data.edges is not None:
        if cfg.setting == "transductive":
            return G.transductive_follow(data.edges, data.n)
        return G.inductive_follow(data.edges, data.n, seen, split.test)
    z = m if cfg.cresci_knn_features == "meta" else x
    if cfg.setting == "transductive":
        return G.transductive_knn(z, cfg.k)
    return G.inductive_knn(z, cfg.k, seen, split.test)


def run_one(data, cfg, seed: int, text_emb: np.ndarray, gcn_fn: Callable, clf_factory: Callable,
            extra: Optional[dict] = None) -> tuple[dict, dict]:
    """Returns (record, predictions). gcn_fn and clf_factory are injected so the
    protocol logic can be tested without PyTorch/LightGBM."""
    cfg.validate()
    t0 = time.time()
    split = make_split(data, cfg, seed)
    y = data.y
    y_view = y.copy()
    y_view[split.test] = -1                                   # test labels hidden from fitting

    fit_rows = np.arange(data.n) if cfg.scaler_fit == "all" else split.train
    m, _ = metadata_matrix(data.meta, fit_rows, cfg.feature_set)
    assert text_emb.shape[0] == data.n
    x = np.hstack([m, text_emb]).astype(np.float32)

    gp = build_graph(data, cfg, x, m, split)
    a_train = G.normalised_adjacency(gp.train_edges, data.n, gp.train_nodes)
    a_infer = G.normalised_adjacency(gp.infer_edges, data.n)
    homophily = G.edge_homophily(gp.train_edges, y_view, split.train)

    # ---- stage 1: GCN, stage 2 inputs ----
    gcn_epochs, gcn_secs = [], 0.0
    if cfg.stacking == "insample":
        gcn_fit, lgbm_fit = split.train, split.train
        res = gcn_fn(x, a_train, a_infer, y_view, gcn_fit, split.val, cfg, seed)
        r = _tap(res, cfg.tap); gcn_epochs.append(res.best_epoch); gcn_secs += res.seconds
    elif cfg.stacking == "blend":
        gcn_fit, lgbm_fit = blend_halves(split.train, y_view, seed)
        res = gcn_fn(x, a_train, a_infer, y_view, gcn_fit, split.val, cfg, seed)
        r = _tap(res, cfg.tap); gcn_epochs.append(res.best_epoch); gcn_secs += res.seconds
    else:  # oof, probability tap only (validated in config)
        lgbm_fit = split.train
        r = np.zeros((data.n, 2), np.float32)
        others = np.setdiff1d(np.arange(data.n), split.train)
        folds = list(kfold(split.train, y_view, seed))
        for f_idx, held in enumerate(folds):
            fit = np.setdiff1d(split.train, held)
            res = gcn_fn(x, a_train, a_infer, y_view, fit, split.val, cfg, seed * 100 + f_idx)
            r[held] = res.prob[held]
            r[others] += res.prob[others] / len(folds)
            gcn_epochs.append(res.best_epoch); gcn_secs += res.seconds
        gcn_fit = split.train

    q = r if cfg.fusion == "P3" else np.hstack([m, text_emb, r])
    q = q.astype(np.float32)

    # ---- stage 2: LightGBM ----
    assert (y_view[lgbm_fit] >= 0).all()
    if cfg.es_on == "val":
        es_idx, es_y = split.val, y_view[split.val]
    else:                                                     # DIAGNOSTIC-INVALID
        es_idx, es_y = split.test, y[split.test]
    clf = clf_factory(cfg, seed).fit(q[lgbm_fit], y_view[lgbm_fit], q[es_idx], es_y)
    p_val = clf.predict_proba(q[split.val])
    p_test = clf.predict_proba(q[split.test])

    rec = cfg.to_dict()
    rec.update(seed=seed, split_kind=split.kind, n=data.n, n_train=len(split.train),
               n_val=len(split.val), n_test=len(split.test), n_gcn_fit=len(gcn_fit),
               n_lgbm_fit=len(lgbm_fit), n_edges_train=int(gp.train_edges.shape[1]),
               n_edges_infer=int(gp.infer_edges.shape[1]), edge_homophily_train=homophily,
               gcn_best_epoch=float(np.mean(gcn_epochs)), gcn_seconds=gcn_secs,
               lgbm_best_iteration=getattr(clf, "best_iteration", None),
               d_meta=m.shape[1], d_text=text_emb.shape[1], d_q=q.shape[1],
               seconds_total=time.time() - t0)
    rec.update(evaluate(y_view[split.val], p_val, prefix="val_"))
    rec.update(evaluate(y[split.test], p_test, prefix="test_"))
    if extra:
        rec.update(extra)
    preds = {"ids": data.ids[split.test], "y": y[split.test], "p": p_test}
    return rec, preds
