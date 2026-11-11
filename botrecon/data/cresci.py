"""Cresci-2017 loader (official release, Botometer bot repository).

Expected layout (the release's own folder names, which end in ".csv"):

    <root>/genuine_accounts.csv/users.csv
    <root>/genuine_accounts.csv/tweets.csv
    <root>/social_spambots_1.csv/...
    ...
    <root>/fake_followers.csv/...

Folders without the ".csv" suffix are also accepted. Traditional spambots
#2-#4 ship without tweets in the release; that is recorded, not repaired.

Group sizes published in Cresci et al. (2017), Table 1:
genuine 3,474; social spambots 991 / 3,457 / 464; traditional spambots
1,000 / 100 / 433 / 1,128; fake followers 3,351 (total 14,398).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .common import AccountData, META_COLUMNS, clean_text, parse_twitter_time, resolve_group_dir, to_number

# Explicit label map [C]: every non-genuine group is an automated/fake account.
EXPLICIT_LABELS = {
    "genuine_accounts": 0,
    "social_spambots_1": 1,
    "social_spambots_2": 1,
    "social_spambots_3": 1,
    "traditional_spambots_1": 1,
    "traditional_spambots_2": 1,
    "traditional_spambots_3": 1,
    "traditional_spambots_4": 1,
    "fake_followers": 1,
}
GROUPS = tuple(EXPLICIT_LABELS)

PUBLISHED_GROUP_SIZES = {
    "genuine_accounts": 3474, "social_spambots_1": 991, "social_spambots_2": 3457,
    "social_spambots_3": 464, "traditional_spambots_1": 1000,
    "traditional_spambots_2": 100, "traditional_spambots_3": 433,
    "traditional_spambots_4": 1128, "fake_followers": 3351,
}


def literal_label(folder_name: str) -> int:
    """Algorithm 1, line 8, as written: bot iff the folder name contains 'bot'.
    Under this rule fake_followers is labelled human (0)."""
    return 1 if "bot" in folder_name else 0


def label_for(group: str, rule: str) -> int:
    if rule == "explicit":
        return EXPLICIT_LABELS[group]
    if rule == "literal_bot_substring":
        return literal_label(group + ".csv")
    raise ValueError(rule)


def _group_dir(root: str, group: str) -> Optional[str]:
    return resolve_group_dir(root, group)


def _read_csv(path: str, **kw) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, on_bad_lines="skip", **kw)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def _recent_tweets(path: str, user_ids: set, max_tweets: int, chunksize: int = 500_000) -> dict:
    """Return {user_id: [text, ...]} with up to max_tweets most recent tweets per user.
    Streams the file in chunks; genuine_accounts/tweets.csv has ~8.4M rows."""
    header = _read_csv(path, nrows=0).columns
    time_col = next((c for c in ("timestamp", "created_at") if c in header), None)
    usecols = ["user_id", "text"] + ([time_col] if time_col else [])
    buf = None
    for enc in ("utf-8", "latin-1"):
        try:
            reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, encoding=enc,
                                 low_memory=False, on_bad_lines="skip")
            for chunk in reader:
                chunk["user_id"] = chunk["user_id"].astype(str).str.replace(r"\.0$", "", regex=True)
                chunk = chunk[chunk["user_id"].isin(user_ids)]
                if time_col:
                    chunk["_t"] = parse_twitter_time(chunk[time_col]).values
                else:
                    chunk["_t"] = pd.NaT
                buf = chunk if buf is None else pd.concat([buf, chunk], ignore_index=True)
                buf = (buf.sort_values("_t", ascending=False, na_position="last")
                          .groupby("user_id", sort=False).head(max_tweets))
            break
        except UnicodeDecodeError:
            buf = None
            continue
    out: dict = {}
    if buf is None or buf.empty:
        return out
    buf = buf.sort_values("_t", ascending=False, na_position="last")
    for uid, grp in buf.groupby("user_id", sort=False):
        out[uid] = [clean_text(t) for t in grp["text"].tolist() if clean_text(t)]
    return out


def load_cresci2017(root: str, label_rule: str = "explicit", load_tweets: bool = False,
                    max_tweets: int = 20, groups=GROUPS) -> AccountData:
    frames, log = [], {"label_rule": label_rule, "groups": {}}
    for g in groups:
        d = _group_dir(root, g)
        if d is None or not os.path.exists(os.path.join(d, "users.csv")):
            log["groups"][g] = {"found": False}
            continue
        u = _read_csv(os.path.join(d, "users.csv"))
        u["id"] = u["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        u["_group"] = g
        u["_label"] = label_for(g, label_rule)
        log["groups"][g] = {
            "found": True, "rows": int(len(u)), "label": int(label_for(g, label_rule)),
            "published_size": PUBLISHED_GROUP_SIZES[g],
            "has_tweets_file": os.path.exists(os.path.join(d, "tweets.csv")),
        }
        frames.append(u)
    if not frames:
        raise FileNotFoundError(f"No Cresci-2017 groups found under {root}")
    df = pd.concat(frames, ignore_index=True)

    # Deduplicate on user ID [C]: identical label -> keep first; conflicting -> drop all copies.
    n_before = len(df)
    lab_per_id = df.groupby("id")["_label"].nunique()
    conflicting = set(lab_per_id[lab_per_id > 1].index)
    df = df[~df["id"].isin(conflicting)]
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    log.update(rows_before_dedup=int(n_before), conflicting_ids_dropped=len(conflicting),
               rows_after_dedup=int(len(df)))

    meta = pd.DataFrame({
        "created_at": parse_twitter_time(df.get("created_at")),
        "followers_count": to_number(df.get("followers_count")),
        "friends_count": to_number(df.get("friends_count")),
        "statuses_count": to_number(df.get("statuses_count")),
        "favourites_count": to_number(df.get("favourites_count")),
        "screen_name": df.get("screen_name").map(clean_text) if "screen_name" in df else "",
    })[list(META_COLUMNS)]
    desc = [clean_text(x) for x in df.get("description", pd.Series([""] * len(df)))]

    tweets = None
    if load_tweets:
        tweets_by_user: dict = {}
        for g in groups:
            d = _group_dir(root, g)
            if d is None:
                continue
            tp = os.path.join(d, "tweets.csv")
            if os.path.exists(tp):
                ids = set(df.loc[df["_group"] == g, "id"])
                tweets_by_user.update(_recent_tweets(tp, ids, max_tweets))
        tweets = [tweets_by_user.get(i, []) for i in df["id"]]

    y = df["_label"].to_numpy(dtype=int)
    log.update(n=int(len(df)), n_bot=int(y.sum()), n_human=int((y == 0).sum()))
    return AccountData(
        name="cresci2017", ids=df["id"].to_numpy(), y=y, meta=meta, description=desc,
        tweets=tweets, edges=None, official_split=None, group=df["_group"].to_numpy(), log=log,
    )
