"""Full experiment suite for the revised paper.

For each (dataset, setting, seed) it prepares data once, then trains and evaluates every
method in botrecon.methods, writing labelled rows to:

  results/suite_main.jsonl     clean test (and validation) metrics, calibration, efficiency
  results/suite_robust.jsonl   test-time perturbations (missing text, dropped/noisy edges, metadata mimicry)
  results/suite_cross.jsonl    train on one dataset, test on the other
  results/suite_explain.jsonl  gate weights, expert accuracy, permutation importance, LightGBM gain
  results/preds/*.npz        test predictions (for curves, confusion matrices, paired tests)

Resumable (skips finished (dataset, setting, method, seed)) and time-limited (--max-minutes).

  python -m botrecon.suite --dataset twibot20 --data-root ... --cross-root <cresci root> --seeds 0 1 2 3 4 --max-minutes 35
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import config as C
from . import graph as G
from . import provenance as P
from .calib import confidence_metrics
from .features import MetadataTransformer, raw_features
from .inputs import PERTURBATIONS, Inputs, drop_edges, metadata_mimicry, missing_text, noisy_edges
from .methods import CROSS_SET, ROBUST_SET, make_all
from .metrics import evaluate
from .splits import make_split
from .text import cached_encode, empty_rates, make_encoder, text_inputs

def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def append(path, rows):
    """JSON Lines: one self-describing row per line, so rows may carry different fields."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: _jsonable(v) for k, v in r.items()}) + "\n")


def read_rows(path) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def done_keys(path):
    d = read_rows(path)
    if d.empty:
        return set()
    return set(map(tuple, d[["dataset", "setting", "method", "seed"]].astype(str).to_numpy()))


# ---------------------------------------------------------------- preparation

class Prepared:
    """Everything label-independent or train-fitted for one (dataset, setting, seed)."""

    def __init__(self, data, emb, cfg, seed, setting, text_avail=None):
        self.data, self.seed, self.setting = data, seed, setting
        self.split = make_split(data, cfg, seed)
        self.y_view = data.y.copy(); self.y_view[self.split.test] = -1
        phi = raw_features(data.meta, cfg.feature_set).to_numpy(float)
        self.transformer = MetadataTransformer().fit(phi, self.split.train)
        m = self.transformer.transform(phi).astype(np.float32)
        # availability comes from the text itself (non-empty), not from the embedding values
        avail = (np.asarray(text_avail, bool) if text_avail is not None
                 else np.array([np.abs(r).sum() > 0 for r in emb]))
        seen = np.sort(np.concatenate([self.split.train, self.split.val]))
        if data.edges is not None:
            gp = (G.transductive_follow(data.edges, data.n) if setting == "transductive"
                  else G.inductive_follow(data.edges, data.n, seen, self.split.test))
            directed = data.edges if setting == "transductive" else None
        else:
            gp = (G.transductive_knn(m, cfg.k) if setting == "transductive"
                  else G.inductive_knn(m, cfg.k, seen, self.split.test))
            directed = None
        self.graph = gp
        self.train_inp = Inputs(m, emb, avail, gp.train_edges, directed)
        self.infer_inp = Inputs(m, emb, avail, gp.infer_edges, directed)
        human_train = self.split.train[self.y_view[self.split.train] == 0]
        self.human_ref = np.median(m[human_train], axis=0)
        deg = self.infer_inp.degree()
        self.log = {
            "edge_homophily_train": G.edge_homophily(gp.train_edges, self.y_view, self.split.train),
            "isolated_share_test": float((deg[self.split.test] == 0).mean()),
            "text_missing_share_test": float((~avail[self.split.test]).mean()),
        }


def target_inputs(target, emb, source: Prepared, cfg, seed):
    """Target dataset seen through the SOURCE model's preprocessing (scaler fitted on source train)."""
    phi = raw_features(target.meta, cfg.feature_set).to_numpy(float)
    m = source.transformer.transform(phi).astype(np.float32)
    avail = np.array([np.abs(r).sum() > 0 for r in emb])
    if target.edges is not None:
        e, directed = G.symmetrise(target.edges), target.edges
    else:
        e, directed = G.transductive_knn(m, cfg.k).train_edges, None
    split = make_split(target, C.corrected(target.name), seed)
    # Accounts present in BOTH datasets (e.g. the Cresci-2015 fake followers re-used in Cresci-2017)
    # would let the source model be tested on accounts it was trained on: remove them by user ID.
    shared = np.isin(target.ids[split.test], source.data.ids)
    return Inputs(m, emb, avail, e, directed), split.test[~shared], int(shared.sum())


# ---------------------------------------------------------------- evaluation helpers

def score(y, p, prefix="test_"):
    r = evaluate(y, p, prefix=prefix)
    return r


def robustness_rows(method, prep, base, rng_seed):
    rows, test, y = [], prep.split.test, prep.data.y
    y_test = y[test]
    for kind, levels in PERTURBATIONS.items():
        if kind in ("drop_edges", "noisy_edges") and prep.infer_inp.sym_edges.size == 0:
            continue
        for lvl in levels:
            rng = np.random.default_rng(rng_seed * 1000 + int(lvl * 100))
            if kind == "missing_text":
                inp = missing_text(prep.infer_inp, test, lvl, rng)
            elif kind == "drop_edges":
                inp = drop_edges(prep.infer_inp, test, lvl, rng)
            elif kind == "noisy_edges":
                inp = noisy_edges(prep.infer_inp, test, lvl, rng)
            else:
                inp = metadata_mimicry(prep.infer_inp, test, y_test, lvl, prep.human_ref, rng)
            p = method.predict(inp)["p"][test]
            r = evaluate(y_test, p, prefix="")
            rows.append({**base, "perturbation": kind, "level": lvl, "accuracy": r["accuracy"],
                         "f1_bot": r["f1_bot"], "mcc": r["mcc"], "roc_auc": r["roc_auc"]})
    return rows


def explain_rows(method, prep, base, out_clean):
    rows, test, y = [], prep.split.test, prep.data.y
    y_test = y[test]
    if method.name == "RGF (proposed)":
        w = out_clean["weights"][test]
        avail = prep.infer_inp
        groups = {"all": np.ones(len(test), bool), "bots": y_test == 1, "humans": y_test == 0,
                  "text missing": ~avail.text_avail[test], "isolated (no graph)": avail.degree()[test] == 0}
        for g, mask in groups.items():
            if mask.any():
                rows.append({**base, "kind": "gate_weight", "group": g, "n": int(mask.sum()),
                             "w_meta": float(w[mask, 0].mean()), "w_text": float(w[mask, 1].mean()),
                             "w_graph": float(w[mask, 2].mean())})
        for k, name in (("m", "metadata"), ("t", "text"), ("g", "graph")):
            pe = out_clean["expert_p"][k][test]
            ok = np.ones(len(test), bool) if k == "m" else (avail.text_avail[test] if k == "t" else avail.degree()[test] > 0)
            if ok.sum() > 1 and len(np.unique(y_test[ok])) == 2:
                r = evaluate(y_test[ok], pe[ok], prefix="")
                rows.append({**base, "kind": "expert_alone", "group": name, "n": int(ok.sum()),
                             "f1_bot": r["f1_bot"], "mcc": r["mcc"]})
        base_r = evaluate(y_test, out_clean["p"][test], prefix="")
        names = list(raw_features(prep.data.meta.iloc[:2]).columns)
        rng = np.random.default_rng(prep.seed)
        for j, fname in enumerate(names):
            d_f1, d_mcc = [], []
            for _ in range(3):
                inp = prep.infer_inp.copy()
                inp.m[test, j] = inp.m[rng.permutation(test), j]
                r = evaluate(y_test, method.predict(inp)["p"][test], prefix="")
                d_f1.append(base_r["f1_bot"] - r["f1_bot"]); d_mcc.append(base_r["mcc"] - r["mcc"])
            rows.append({**base, "kind": "permutation_importance", "group": f"metadata: {fname}",
                         "delta_f1": float(np.mean(d_f1)), "delta_mcc": float(np.mean(d_mcc))})
        for label, fn in (("text (all test accounts)", lambda: missing_text(prep.infer_inp, test, 1.0, rng)),
                          ("graph (all test edges)", lambda: drop_edges(prep.infer_inp, test, 1.0, rng))):
            r = evaluate(y_test, method.predict(fn())["p"][test], prefix="")
            rows.append({**base, "kind": "permutation_importance", "group": f"remove {label}",
                         "delta_f1": base_r["f1_bot"] - r["f1_bot"], "delta_mcc": base_r["mcc"] - r["mcc"]})
    elif method.name == "LightGBM (meta+text)":
        imp = method.importances()
        if imp is not None:
            names = list(raw_features(prep.data.meta.iloc[:2]).columns)
            tot = imp.sum() or 1.0
            for j, fname in enumerate(names):
                rows.append({**base, "kind": "lgbm_split_importance", "group": f"metadata: {fname}",
                             "share": float(imp[j] / tot)})
            rows.append({**base, "kind": "lgbm_split_importance", "group": "text (all dimensions)",
                         "share": float(imp[len(names):].sum() / tot)})
    return rows


# ---------------------------------------------------------------- main loop

def run_suite(args, data_loader, encoder_factory, methods_factory=make_all, cross_loader=None):
    t_start = time.time()
    cfg = C.corrected(args.dataset, text=args.text)
    data = data_loader(args.dataset, args.data_root, cfg)
    enc = encoder_factory(cfg)
    texts = text_inputs(data, cfg.text, cfg.max_tweets)
    emb = cached_encode(enc, texts, args.cache_dir, f"suite_{args.dataset}_{cfg.text}")
    rates = empty_rates(texts, data.y)
    cross = None
    if args.cross_root:
        other = C.CROSS_PARTNER[args.dataset]
        tdata = (cross_loader or data_loader)(other, args.cross_root, C.corrected(other, text=args.text))
        temb = cached_encode(enc, text_inputs(tdata, cfg.text, cfg.max_tweets), args.cache_dir,
                             f"suite_{other}_{cfg.text}")
        cross = (other, tdata, temb)
    prov_override = P.SMOKE_TEST if args.smoke else None
    finished = set() if args.no_resume else done_keys(os.path.join(args.out_dir, "suite_main.jsonl"))
    out = {k: os.path.join(args.out_dir, f"suite_{k}.jsonl") for k in ("main", "robust", "cross", "explain")}
    os.makedirs(os.path.join(args.out_dir, "preds"), exist_ok=True)
    n_run = 0
    for setting in args.settings:
        for seed in args.seeds:
            todo = [m for m in methods_factory(args.dataset, setting)
                    if (args.dataset, setting, m.name, str(seed)) not in finished
                    and (not args.methods or m.name in args.methods)]
            if not todo:
                continue
            prep = Prepared(data, emb, cfg, seed, setting, text_avail=[len(t.strip()) > 0 for t in texts])
            for method in todo:
                elapsed = (time.time() - t_start) / 60
                if args.max_minutes is not None and elapsed >= args.max_minutes:
                    print(f"Time budget reached after {elapsed:.1f} min. Rerun the same command to resume.")
                    return n_run
                base = {"dataset": args.dataset, "setting": setting, "method": method.name, "seed": seed,
                        "group": method.group, "provenance": prov_override or method.provenance,
                        "text_variant": cfg.text, "text_encoder": enc.name,
                        "text_encoder_source": getattr(enc, "source", "")}
                t0 = time.time()
                info = method.fit(prep.train_inp, prep.y_view, prep.split, seed, args.device)
                t1 = time.time()
                outp = method.predict(prep.infer_inp)
                pred_ms = (time.time() - t1) * 1000 / prep.data.n * 1000
                p, test, val = outp["p"], prep.split.test, prep.split.val
                row = dict(base, **info, fit_seconds=t1 - t0, predict_ms_per_1k=pred_ms,
                           n_train=len(prep.split.train), n_val=len(val), n_test=len(test), **prep.log, **rates)
                row.update(evaluate(prep.y_view[val], p[val], prefix="val_"))
                row.update(evaluate(prep.data.y[test], p[test], prefix="test_"))
                row.update(confidence_metrics(prep.data.y[test], p[test], prefix="test_"))
                if "p_uncalibrated" in outp:
                    row.update(confidence_metrics(prep.data.y[test], outp["p_uncalibrated"][test], prefix="test_uncal_"))
                tag = f"{args.dataset}_{setting}_{method.name}_{seed}".replace(" ", "_").replace("/", "-")
                np.savez_compressed(os.path.join(args.out_dir, "preds", tag + ".npz"),
                                    ids=prep.data.ids[test], y=prep.data.y[test], p=p[test],
                                    **({"weights": outp["weights"][test]} if "weights" in outp else {}))
                if seed in getattr(args, "checkpoint_seeds", []):
                    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
                    if method.save(os.path.join(args.out_dir, "checkpoints", tag)):
                        row["checkpoint"] = os.path.join("checkpoints", tag)
                if setting == "transductive" and seed in args.robust_seeds and method.name in ROBUST_SET:
                    append(out["robust"], robustness_rows(method, prep, base, seed))
                if setting == "transductive" and seed in args.robust_seeds:
                    append(out["explain"], explain_rows(method, prep, base, outp))
                if cross and setting == "transductive" and seed in args.robust_seeds and method.name in CROSS_SET:
                    other, tdata, temb = cross
                    tinp, ttest, n_shared = target_inputs(tdata, temb, prep, cfg, seed)
                    if len(ttest):
                        r = evaluate(tdata.y[ttest], method.predict(tinp)["p"][ttest], prefix="")
                        append(out["cross"], [{**base, "source": args.dataset, "target": other,
                                               "overlap_removed": n_shared, "target_bots": int(tdata.y[ttest].sum()),
                                               "target_humans": int((tdata.y[ttest] == 0).sum()), **r}])
                append(out["main"], [row])           # written last: a crash above means the run is redone
                n_run += 1
                print(f"[{base['provenance']}] {args.dataset}/{setting}/seed {seed}: {method.name:38s} "
                      f"F1={row['test_f1_bot']:.4f} MCC={row['test_mcc']:.4f} AUC={row['test_roc_auc']:.4f} "
                      f"({row['fit_seconds']:.0f}s fit; {(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)
    return n_run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cross-root", default=None, help="root of the OTHER dataset, for cross-dataset tests")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--robust-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--checkpoint-seeds", type=int, nargs="*", default=[0],
                    help="save trained models for these seeds (results/checkpoints/)")
    ap.add_argument("--settings", nargs="+", default=["transductive", "inductive"])
    ap.add_argument("--methods", nargs="*", default=None, help="restrict to these method names")
    ap.add_argument("--text", default="B", choices=["A", "B"])
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="hashing text encoder; rows labelled SMOKE-TEST")
    args = ap.parse_args(argv)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    from .run import cached_load

    def loader(ds, root, cfg):
        return cached_load(cfg, root, args.cache_dir, cfg.text == "A")

    def enc_factory(cfg):
        return make_encoder(cfg, args.device, test_encoder=args.smoke)

    n = run_suite(args, loader, enc_factory)
    print(f"done: {n} method runs this call")


if __name__ == "__main__":
    main()
