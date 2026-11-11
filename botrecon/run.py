"""Command-line runner.

Examples (Colab/Kaggle, GPU):
  python -m botrecon.run --dataset twibot20 --data-root /data/TwiBot-20 --family corrected --seeds 0 1 2 3 4
  python -m botrecon.run --dataset cresci2017 --data-root /data/cresci-2017/datasets_full.csv --family literal --seeds 0
  python -m botrecon.run --dataset twibot20 --data-root ... --family corrected --inductive --only fusion=P3,tap=hidden,text=B
  python -m botrecon.run --dataset twibot20 --data-root ... --family literal --diagnostic --only fusion=P3,tap=prob,text=B,twibot_split=random

Every row appended to --out carries a 'provenance' column.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from . import config as C
from .data import load
from .pipeline import run_one
from .text import cached_encode, empty_rates, make_encoder, text_inputs


def _parse_only(s: str | None) -> dict:
    if not s:
        return {}
    return dict(kv.split("=", 1) for kv in s.split(","))


def _matches(cfg, only: dict) -> bool:
    return all(str(getattr(cfg, k)) == v for k, v in only.items())


def configs_for(args):
    only = _parse_only(args.only)
    grid = C.literal_grid(args.dataset) if args.family == "literal" else C.corrected_grid(args.dataset)
    for cfg in grid:
        if not _matches(cfg, only):
            continue
        if args.inductive:
            cfg = replace(cfg, setting="inductive")
        if args.diagnostic:
            cfg = C.diagnostic(cfg)
        if args.smoke:
            cfg = replace(cfg, smoke_test=True)
        cfg.validate()
        yield cfg


TAG_KEYS = ("dataset", "family", "fusion", "tap", "text", "cresci_label_rule",
            "cresci_knn_features", "twibot_split", "setting", "es_on")


def run_tag(cfg, seed: int) -> str:
    tag = "_".join(str(getattr(cfg, k)) for k in TAG_KEYS) + f"_{seed}"
    return ("SMOKE_" + tag) if cfg.smoke_test else tag


def done_tags(out_csv: str) -> set:
    if not os.path.exists(out_csv):
        return set()
    try:
        return set(pd.read_csv(out_csv, usecols=["run_tag"])["run_tag"].dropna())
    except (ValueError, pd.errors.EmptyDataError):
        return set()


def cached_load(cfg, data_root: str, cache_dir: str | None, need_tweets: bool):
    """Loading TwiBot-20 JSON (and Cresci-2017 tweets) is slow on CPU; cache the parsed data.
    The cache key includes the data path, label rule and whether tweets were loaded."""
    import hashlib
    import pickle
    key = f"{cfg.dataset}|{os.path.abspath(data_root)}|{cfg.cresci_label_rule}|{need_tweets}|{cfg.max_tweets}"
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "data_" + hashlib.sha1(key.encode()).hexdigest()[:12] + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    data = load(cfg.dataset, data_root, label_rule=cfg.cresci_label_rule,
                load_tweets=need_tweets, max_tweets=cfg.max_tweets)
    if path:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    return data


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "lightgbm", "transformers", "sentence_transformers", "sklearn", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["gpu"] = None
    return env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--family", required=True, choices=C.FAMILIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--only", default=None, help="filter, e.g. fusion=P3,tap=hidden,text=B")
    ap.add_argument("--inductive", action="store_true")
    ap.add_argument("--diagnostic", action="store_true", help="LightGBM early-stopped on TEST (invalid)")
    ap.add_argument("--smoke", action="store_true", help="hashing text encoder; rows labelled SMOKE-TEST")
    ap.add_argument("--out", default="results/runs.csv")
    ap.add_argument("--pred-dir", default="results/preds")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop starting new runs after this many minutes; rerun the same command to resume")
    ap.add_argument("--no-resume", action="store_true", help="rerun runs already present in --out")
    args = ap.parse_args(argv)
    t_start = time.time()
    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass

    from .classifier import LGBMStage
    from .gcn import train_gcn

    def gcn_fn(*a, **k):
        return train_gcn(*a, device=args.device, **k)

    env = environment()
    cfgs = list(configs_for(args))
    if not cfgs:
        raise SystemExit("No configuration matches --only")
    finished = set() if args.no_resume else done_tags(args.out)
    todo = [(c, s) for c in cfgs for s in args.seeds if run_tag(c, s) not in finished]
    print(f"{len(cfgs)} configs x {len(args.seeds)} seeds = {len(cfgs) * len(args.seeds)} runs; "
          f"{len(cfgs) * len(args.seeds) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return
    n_done, run_secs = 0, []

    datasets, encoders, embeddings = {}, {}, {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    for cfg in cfgs:
        seeds = [s for s in args.seeds if (cfg, s) in todo]
        if not seeds:
            continue
        need_tweets = cfg.text == "A"
        dkey = (cfg.cresci_label_rule if cfg.dataset == "cresci2017" else "-", need_tweets)
        if dkey not in datasets:
            datasets[dkey] = cached_load(cfg, args.data_root, args.cache_dir, need_tweets)
            print(json.dumps(datasets[dkey].log, default=str)[:2000])
        data = datasets[dkey]
        texts = text_inputs(data, cfg.text, cfg.max_tweets)
        ekey = (dkey, cfg.text)
        if ekey not in embeddings:
            if cfg.text not in encoders:
                encoders[cfg.text] = make_encoder(cfg, args.device, test_encoder=args.smoke)
            enc = encoders[cfg.text]
            embeddings[ekey] = cached_encode(enc, texts, args.cache_dir, f"{cfg.dataset}_{cfg.text}_{dkey[0]}")
        emb = embeddings[ekey]
        extra = {"text_encoder": encoders[cfg.text].name, **empty_rates(texts, data.y),
                 "data_n_bot": data.log.get("n_bot"), "data_n_human": data.log.get("n_human"),
                 "env": json.dumps(env)}
        for seed in seeds:
            elapsed = (time.time() - t_start) / 60
            if args.max_minutes is not None and elapsed >= args.max_minutes:
                print(f"Time budget reached ({elapsed:.1f} min): {len(todo) - n_done} runs left. "
                      f"Rerun the same command to resume.")
                return
            t_run = time.time()
            rec, preds = run_one(data, cfg, seed, emb, gcn_fn, LGBMStage, extra)
            tag = run_tag(cfg, seed)
            np.savez_compressed(os.path.join(args.pred_dir, tag + ".npz"), **preds)
            rec["pred_file"] = tag + ".npz"
            rec["run_tag"] = tag
            pd.DataFrame([rec]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
            n_done += 1
            run_secs.append(time.time() - t_run)
            eta = np.mean(run_secs) * (len(todo) - n_done) / 60
            print(f"[{rec['provenance']}] {tag}: test acc={rec['test_accuracy']:.4f} "
                  f"F1(bot)={rec['test_f1_bot']:.4f} MCC={rec['test_mcc']:.4f} val F1={rec['val_f1_bot']:.4f} "
                  f"| {time.time() - t_run:.0f}s, {n_done}/{len(todo)} done, ~{eta:.1f} min left")


if __name__ == "__main__":
    main()
