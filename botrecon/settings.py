"""Load and validate the experiment configuration (config/kaggle_cpu.json)."""
from __future__ import annotations

import json

DEFAULTS = {
    "input_dir": "/kaggle/input", "work_dir": "/kaggle/working",
    "paths": {"cresci2017": None, "cresci2015": None},
    "session_budget_minutes": 40, "threads": None,
    "seeds": [0, 1, 2, 3, 4], "robust_seeds": [0, 1, 2], "checkpoint_seeds": [0],
    "text_variant": "B", "run_bert_tweets": False,
    "walkthrough": {"dataset": "auto", "seed": 0},
}


def load_config(path: str) -> dict:
    with open(path) as f:
        user = json.load(f)
    cfg = {**DEFAULTS, **{k: v for k, v in user.items() if not k.startswith("_")}}
    cfg["paths"] = {**DEFAULTS["paths"], **(user.get("paths") or {})}
    cfg["walkthrough"] = {**DEFAULTS["walkthrough"], **(user.get("walkthrough") or {})}
    assert cfg["text_variant"] in ("A", "B"), "text_variant must be A (BERT on tweets) or B (SBERT on bios)"
    assert len(cfg["seeds"]) >= 1 and set(cfg["robust_seeds"]) <= set(cfg["seeds"])
    assert set(cfg["checkpoint_seeds"]) <= set(cfg["seeds"])
    assert cfg["session_budget_minutes"] > 0
    return cfg
