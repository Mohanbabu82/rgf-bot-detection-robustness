"""Shared context for the Kaggle notebook. Every cell calls ctx(): it reads the config and the
detected dataset paths from disk, so cells can be re-run in any order after a restart."""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

from .datapaths import locate
from .settings import load_config

STATE = "results/session.json"


def start_session(budget_minutes: float | None = None) -> None:
    os.makedirs("results", exist_ok=True)
    with open(STATE, "w") as f:
        json.dump({"t0": time.time(), "budget": budget_minutes}, f)


def ctx(config_path: str = "config/kaggle_cpu.json", refresh_paths: bool = False) -> SimpleNamespace:
    cfg = load_config(config_path)
    if os.path.exists(STATE):
        st = json.load(open(STATE))
    else:
        start_session(); st = json.load(open(STATE))
    budget = st.get("budget") or cfg["session_budget_minutes"]
    work = cfg["work_dir"] if os.path.isdir(cfg["work_dir"]) else os.getcwd()
    paths_file = "results/paths.json"
    if refresh_paths or not os.path.exists(paths_file):
        found = locate(cfg["input_dir"], work, cfg["paths"]["cresci2017"], cfg["paths"]["cresci2015"])
        json.dump(found, open(paths_file, "w"), indent=1)
    paths = json.load(open(paths_file))
    available = [d for d in ("cresci2017", "cresci2015") if paths.get(d)]

    def left(reserve: float = 0.0) -> float:
        return max(0.0, round(budget - reserve - (time.time() - st["t0"]) / 60, 1))

    wd = cfg["walkthrough"]["dataset"]
    walk_ds = wd if wd != "auto" else (available[0] if available else None)
    return SimpleNamespace(cfg=cfg, paths=paths, available=available, left=left, budget=budget,
                           threads=cfg["threads"] or os.cpu_count(), walk_dataset=walk_ds,
                           walk_seed=cfg["walkthrough"]["seed"], smoke=bool(cfg.get("smoke_test", False)))
