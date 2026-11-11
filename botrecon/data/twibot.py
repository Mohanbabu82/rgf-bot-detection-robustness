"""TwiBot-20 loader (official release, obtained from the dataset authors).

Expected files under <root>: train.json, dev.json, test.json (support.json is
not used: only the 11,826 labelled users and the follow edges among them).

Each record: {"ID", "profile": {...Twitter API fields as strings...},
"tweet": [str, ...] | null, "neighbor": {"following": [...], "follower": [...]} | null,
"domain": [...], "label": "0" | "1"}  (label 1 = bot).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .common import AccountData, META_COLUMNS, clean_text, parse_twitter_time, to_number

SPLIT_FILES = {"train": "train.json", "val": "dev.json", "test": "test.json"}


def _norm_id(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def load_twibot20(root: str, load_tweets: bool = False, max_tweets: int = 20) -> AccountData:
    recs, splits = [], []
    for split, fname in SPLIT_FILES.items():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        recs.extend(data)
        splits.extend([split] * len(data))

    ids = np.array([_norm_id(r["ID"]) for r in recs])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs across official TwiBot-20 files")
    y = np.array([int(str(r["label"]).strip()) for r in recs], dtype=int)
    prof = [r.get("profile") or {} for r in recs]

    meta = pd.DataFrame({
        "created_at": parse_twitter_time([p.get("created_at") for p in prof]),
        "followers_count": to_number([p.get("followers_count") for p in prof]),
        "friends_count": to_number([p.get("friends_count") for p in prof]),
        "statuses_count": to_number([p.get("statuses_count") for p in prof]),
        "favourites_count": to_number([p.get("favourites_count") for p in prof]),
        "screen_name": [clean_text(p.get("screen_name")) for p in prof],
    })[list(META_COLUMNS)]
    desc = [clean_text(p.get("description")) for p in prof]

    tweets = None
    if load_tweets:
        # [C] The release does not document tweet order; the first entries are taken.
        tweets = [[clean_text(t) for t in (r.get("tweet") or []) if clean_text(t)][:max_tweets]
                  for r in recs]

    index = {u: i for i, u in enumerate(ids)}
    src, dst = [], []
    for i, r in enumerate(recs):
        nb = r.get("neighbor") or {}
        for j in nb.get("following") or []:       # i follows j : edge i -> j
            j = index.get(_norm_id(j))
            if j is not None and j != i:
                src.append(i); dst.append(j)
        for j in nb.get("follower") or []:        # j follows i : edge j -> i
            j = index.get(_norm_id(j))
            if j is not None and j != i:
                src.append(j); dst.append(i)
    edges = np.unique(np.array([src, dst], dtype=np.int64), axis=1) if src else np.zeros((2, 0), np.int64)

    split = np.array(splits)
    log = {
        "n": int(len(ids)), "n_bot": int(y.sum()), "n_human": int((y == 0).sum()),
        "split_sizes": {s: int((split == s).sum()) for s in SPLIT_FILES},
        "directed_edges_among_labelled": int(edges.shape[1]),
        "isolated_nodes": int(len(ids) - len(np.unique(edges))) if edges.size else int(len(ids)),
    }
    return AccountData(name="twibot20", ids=ids, y=y, meta=meta, description=desc, tweets=tweets,
                       edges=edges, official_split=split, group=None, log=log)
