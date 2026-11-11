"""Synthetic files with the STRUCTURE of the real Cresci-2017 and TwiBot-20
releases (folder names, column names, string-padded counts, neighbour lists).
Values are random. Used only to test the code; never a result."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from botrecon.data.cresci import GROUPS

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def tw_time(rng, year_lo=2008, year_hi=2016):
    y = rng.integers(year_lo, year_hi + 1); m = rng.integers(0, 12); d = rng.integers(1, 28)
    return f"{DAYS[rng.integers(0, 7)]} {MONTHS[m]} {d:02d} {rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}:00 +0000 {y}"


def _account(rng, bot: bool, uid: str):
    followers = int(rng.lognormal(3 if bot else 5, 1))
    friends = int(rng.lognormal(6 if bot else 5, 1))
    return {
        "id": uid,
        "screen_name": ("x" + "".join(rng.choice(list("abcdefghij0123456789"), 12))) if bot
        else "".join(rng.choice(list("annamariajohn"), 7)),
        "statuses_count": int(rng.lognormal(8 if bot else 6, 1)),
        "followers_count": followers, "friends_count": friends,
        "favourites_count": int(rng.lognormal(2 if bot else 5, 1)),
        "created_at": tw_time(rng, 2012 if bot else 2008, 2016 if bot else 2012),
        "description": "" if (bot and rng.random() < 0.5) else ("buy now cheap deals" if bot else "love books and music"),
    }


def make_cresci(root: str, per_group: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    info, uid = {}, 1000
    no_tweets = {"traditional_spambots_2", "traditional_spambots_3", "traditional_spambots_4"}
    first_genuine_id = None
    for g in GROUPS:
        d = os.path.join(root, g + ".csv"); os.makedirs(d, exist_ok=True)
        bot = g != "genuine_accounts"
        rows = []
        for _ in range(per_group):
            uid += 1
            rows.append(_account(rng, bot, str(uid)))
        if g == "genuine_accounts":
            first_genuine_id = rows[0]["id"]
        if g == "social_spambots_1":
            rows.append(dict(rows[0]))                       # same-label duplicate (kept once)
        if g == "fake_followers":
            dup = _account(rng, True, first_genuine_id)      # conflicting-label duplicate (dropped)
            rows.append(dup)
        pd.DataFrame(rows).to_csv(os.path.join(d, "users.csv"), index=False)
        if g not in no_tweets:
            tw = []
            for r in rows:
                for j in range(25):
                    tw.append({"user_id": r["id"], "text": f"tweet {j} of {r['id']}",
                               "timestamp": f"2016-01-{(j % 28) + 1:02d} 10:00:00"})
            pd.DataFrame(tw).to_csv(os.path.join(d, "tweets.csv"), index=False)
        info[g] = per_group
    info["conflict_id"] = first_genuine_id
    return info


def make_twibot(root: str, n_train: int = 120, n_val: int = 40, n_test: int = 55, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    total = n_train + n_val + n_test
    ids = [str(10_000 + i) for i in range(total)]
    labels = rng.integers(0, 2, total)
    recs = []
    for i, u in enumerate(ids):
        a = _account(rng, bool(labels[i]), u)
        same = [ids[j] for j in rng.choice(total, 6) if labels[j] == labels[i] and j != i]
        prof = {k: (str(v) + " ") for k, v in a.items() if k != "id"}   # padded strings, as in the release
        prof["id"] = u + " "
        recs.append({
            "ID": u, "profile": prof,
            "tweet": None if rng.random() < 0.1 else [f"t{k} {u}" for k in range(30)],
            "neighbor": None if rng.random() < 0.1 else {"following": same[:3] + ["99999999"],
                                                          "follower": same[3:]},
            "domain": ["Politics"], "label": str(int(labels[i])),
        })
    parts = {"train.json": recs[:n_train], "dev.json": recs[n_train:n_train + n_val],
             "test.json": recs[n_train + n_val:]}
    for f, r in parts.items():
        with open(os.path.join(root, f), "w") as fh:
            json.dump(r, fh)
    return {"ids": ids, "labels": labels, "records": recs}


def make_cresci2015(root: str, per_group: int = 30, seed: int = 1, shared_bot_ids=()) -> dict:
    """Cresci-2015 layout: E13, TFP (human), FSF, INT, TWT (bot), each with users/tweets/followers/friends.
    TWT uses a folder name without '.csv' (both forms occur). shared_bot_ids are re-used as FSF account IDs,
    mimicking the fake followers shared with Cresci-2017. Edges are mostly within-class."""
    rng = np.random.default_rng(seed)
    info, uid, by_label = {}, 50_000, {0: [], 1: []}
    groups = {"E13": 0, "TFP": 0, "FSF": 1, "INT": 1, "TWT": 1}
    dirs = {}
    shared = list(shared_bot_ids)
    for g, lab in groups.items():
        d = os.path.join(root, g if g == "TWT" else g + ".csv"); os.makedirs(d, exist_ok=True); dirs[g] = d
        rows = []
        for _ in range(per_group):
            if g == "FSF" and shared:
                u = shared.pop(0)
            else:
                uid += 1; u = str(uid)
            rows.append(_account(rng, bool(lab), u))
        pd.DataFrame(rows).to_csv(os.path.join(d, "users.csv"), index=False)
        pd.DataFrame([{"user_id": r["id"], "text": f"tweet {j} of {r['id']}",
                       "timestamp": f"2014-01-{j + 1:02d} 10:00:00"} for r in rows for j in range(5)]) \
            .to_csv(os.path.join(d, "tweets.csv"), index=False)
        by_label[lab] += [r["id"] for r in rows]
        info[g] = [r["id"] for r in rows]
    for g, lab in groups.items():
        rel = {"followers.csv": [], "friends.csv": []}
        for u in info[g]:
            for f in rel:
                same = rng.choice(by_label[lab], 2)
                other = rng.choice(by_label[1 - lab], 1)
                for v in list(same) + (list(other) if rng.random() < 0.15 else []):
                    if v != u:
                        rel[f].append({"source_id": v, "target_id": u} if f == "followers.csv"
                                      else {"source_id": u, "target_id": v})
                rel[f].append({"source_id": u, "target_id": "123456789"})     # unlabelled neighbour: dropped
        for f, r in rel.items():
            pd.DataFrame(r).to_csv(os.path.join(dirs[g], f), index=False)
    return info
