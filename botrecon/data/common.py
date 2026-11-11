"""Shared dataset container and parsing helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"

# Raw metadata columns every loader must provide (NaN allowed).
META_COLUMNS = (
    "created_at",        # pandas Timestamp (UTC) or NaT
    "followers_count",
    "friends_count",     # "following"
    "statuses_count",
    "favourites_count",
    "screen_name",
)


@dataclass
class AccountData:
    name: str
    ids: np.ndarray                      # str, shape (N,)
    y: np.ndarray                        # int {0 human, 1 bot}, shape (N,)
    meta: pd.DataFrame                   # META_COLUMNS, N rows, positional index
    description: list                    # N strings ("" if missing)
    tweets: Optional[list] = None        # N lists of strings, most recent first; None if not loaded
    edges: Optional[np.ndarray] = None   # (2, E) int, source -> target, positional indices
    official_split: Optional[np.ndarray] = None  # "train"/"val"/"test" per node, or None
    group: Optional[np.ndarray] = None   # source group per node (Cresci-2017)
    log: dict = field(default_factory=dict)  # counts and decisions, written to the run record

    def __post_init__(self):
        n = len(self.ids)
        assert len(self.y) == n and len(self.meta) == n and len(self.description) == n
        assert set(np.unique(self.y)).issubset({0, 1}), "labels must be 0/1"
        if self.tweets is not None:
            assert len(self.tweets) == n
        if self.edges is not None:
            assert self.edges.shape[0] == 2
            assert self.edges.size == 0 or (self.edges.min() >= 0 and self.edges.max() < n)

    @property
    def n(self) -> int:
        return len(self.ids)


def parse_twitter_time(values) -> pd.Series:
    """Parse Twitter API timestamps ('Tue Jun 11 10:53:33 +0000 2013'),
    falling back to generic parsing. Returns UTC timestamps (NaT if unparseable)."""
    s = pd.Series(values, dtype="object").astype(str).str.strip()
    s = s.where(~s.isin(["", "nan", "None", "NaN", "NaT"]), None)
    out = pd.to_datetime(s, format=TWITTER_TIME_FORMAT, errors="coerce", utc=True)
    miss = out.isna() & s.notna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(s[miss], errors="coerce", utc=True)
    return out


def to_number(values) -> pd.Series:
    """Counts arrive as ints, floats or padded strings ('123 '). Unparseable -> NaN."""
    s = pd.Series(values, dtype="object").astype(str).str.strip()
    return pd.to_numeric(s, errors="coerce")


def clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    x = str(x).strip()
    return "" if x.lower() in ("nan", "none") else x


def resolve_group_dir(root: str, group: str):
    """Folder holding <group>'s users.csv. Accepts '<group>.csv/' or '<group>/', and the doubly nested
    '<group>.csv/<group>.csv/' layout that Kaggle produces when it unpacks archives inside archives."""
    import os
    names = (group + ".csv", group, group.lower() + ".csv", group.lower())
    for a in names:
        p = os.path.join(root, a)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "users.csv")):
            return p
        for b in names:
            q = os.path.join(p, b)
            if os.path.isdir(q) and os.path.exists(os.path.join(q, "users.csv")):
                return q
    return None
