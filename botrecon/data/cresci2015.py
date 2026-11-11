"""Cresci-2015 loader (official release, Botometer bot repository; "Fame for sale", Cresci et al. 2015).

Expected layout (the release's own folder names may or may not end in ".csv"):

    <root>/E13.csv/users.csv   tweets.csv   followers.csv   friends.csv
    <root>/TFP.csv/...
    <root>/FSF.csv/...
    <root>/INT.csv/...
    <root>/TWT.csv/...

Labels [C, following the release]: E13 and TFP are verified humans (0); FSF, INT and TWT are
fake followers bought from three markets (1).

Graph: followers.csv / friends.csv hold one relation per row. The column names differ between
copies (source_id/target_id is the common form); the first two ID columns are used. Only
relations where BOTH ends are labelled accounts in this release are kept. The direction is
stored as given but every graph model symmetrises it, so the direction convention does not
change results except for the relational model (RGCN), which is documented as such.

IMPORTANT for cross-dataset tests: the 3,351 Cresci-2015 fake followers are the same
accounts as the Cresci-2017 "fake_followers" group. Overlap is removed by user ID at
evaluation time (botrecon.suite.target_inputs); see EXPERIMENT_PROTOCOL.md.

Group sizes published in Cresci et al. (2015): TFP 469, E13 1,481 (humans, 1,950);
FSF 1,169, INT 1,337, TWT 845 (fake followers, 3,351); total 5,301.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import AccountData, META_COLUMNS, clean_text, parse_twitter_time, resolve_group_dir, to_number
from .cresci import _read_csv, _recent_tweets

LABELS = {"E13": 0, "TFP": 0, "FSF": 1, "INT": 1, "TWT": 1}
GROUPS = tuple(LABELS)
PUBLISHED_GROUP_SIZES = {"TFP": 469, "E13": 1481, "FSF": 1169, "INT": 1337, "TWT": 845}
RELATION_FILES = ("followers.csv", "friends.csv")


def group_dir(root: str, group: str):
    return resolve_group_dir(root, group)


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _relation_pairs(path: str) -> pd.DataFrame:
    df = _read_csv(path, dtype=str)
    cols = [c for c in ("source_id", "target_id") if c in df.columns]
    if len(cols) < 2:
        cols = list(df.columns[:2])
    out = pd.DataFrame({"s": _norm(df[cols[0]]), "t": _norm(df[cols[1]])})
    return out.dropna()


def load_cresci2015(root: str, load_tweets: bool = False, max_tweets: int = 20, groups=GROUPS,
                    **_ignored) -> AccountData:
    frames, log = [], {"groups": {}}
    for g in groups:
        d = group_dir(root, g)
        if d is None or not os.path.exists(os.path.join(d, "users.csv")):
            log["groups"][g] = {"found": False}
            continue
        u = _read_csv(os.path.join(d, "users.csv"))
        u["id"] = _norm(u["id"])
        u["_group"], u["_label"] = g, LABELS[g]
        log["groups"][g] = {"found": True, "rows": int(len(u)), "label": LABELS[g],
                            "published_size": PUBLISHED_GROUP_SIZES[g],
                            "relation_files": [f for f in RELATION_FILES if os.path.exists(os.path.join(d, f))]}
        frames.append(u)
    if not frames:
        raise FileNotFoundError(f"No Cresci-2015 groups (E13, TFP, FSF, INT, TWT) found under {root}")
    df = pd.concat(frames, ignore_index=True)

    n_before = len(df)
    lab_per_id = df.groupby("id")["_label"].nunique()
    conflicting = set(lab_per_id[lab_per_id > 1].index)
    df = df[~df["id"].isin(conflicting)].drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    log.update(rows_before_dedup=int(n_before), conflicting_ids_dropped=len(conflicting),
               rows_after_dedup=int(len(df)))

    meta = pd.DataFrame({
        "created_at": parse_twitter_time(df.get("created_at")),
        "followers_count": to_number(df.get("followers_count")),
        "friends_count": to_number(df.get("friends_count")),
        "statuses_count": to_number(df.get("statuses_count")),
        "favourites_count": to_number(df.get("favourites_count")),
        "screen_name": df["screen_name"].map(clean_text) if "screen_name" in df else "",
    })[list(META_COLUMNS)]
    desc = [clean_text(x) for x in df.get("description", pd.Series([""] * len(df)))]

    index = {u: i for i, u in enumerate(df["id"])}
    pairs, raw_rows = [], 0
    for g in groups:
        d = group_dir(root, g)
        if d is None:
            continue
        for f in RELATION_FILES:
            p = os.path.join(d, f)
            if os.path.exists(p):
                r = _relation_pairs(p)
                raw_rows += len(r)
                s, t = r["s"].map(index), r["t"].map(index)
                keep = s.notna() & t.notna()
                pairs.append(np.vstack([s[keep].to_numpy(np.int64), t[keep].to_numpy(np.int64)]))
    if pairs:
        e = np.concatenate(pairs, axis=1)
        e = e[:, e[0] != e[1]]
        edges = np.unique(e, axis=1) if e.size else np.zeros((2, 0), np.int64)
    else:
        edges = np.zeros((2, 0), np.int64)

    tweets = None
    if load_tweets:
        by_user: dict = {}
        for g in groups:
            d = group_dir(root, g)
            if d is not None and os.path.exists(os.path.join(d, "tweets.csv")):
                by_user.update(_recent_tweets(os.path.join(d, "tweets.csv"),
                                              set(df.loc[df["_group"] == g, "id"]), max_tweets))
        tweets = [by_user.get(i, []) for i in df["id"]]

    y = df["_label"].to_numpy(dtype=int)
    touched = np.unique(edges) if edges.size else np.array([], np.int64)
    log.update(n=int(len(df)), n_bot=int(y.sum()), n_human=int((y == 0).sum()),
               relation_rows_read=int(raw_rows), directed_edges_among_labelled=int(edges.shape[1]),
               isolated_nodes=int(len(df) - len(touched)))
    if edges.shape[1] == 0:
        raise ValueError("Cresci-2015: no follower/friend relations between labelled accounts were found. "
                         "Check that followers.csv / friends.csv are present in each group folder.")
    return AccountData(name="cresci2015", ids=df["id"].to_numpy(), y=y, meta=meta, description=desc,
                       tweets=tweets, edges=edges, official_split=None, group=df["_group"].to_numpy(), log=log)
