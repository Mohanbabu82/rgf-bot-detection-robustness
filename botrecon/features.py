"""Metadata features.

Names from the manuscript (Sec. 3.1, "the exact set"): account age, follower
ratio, tweet frequency, username entropy [M]. The manuscript gives no
formulas; those below are reconstruction choices [C]. The 'extended6' set adds
reputation and engagement ratios, which the manuscript names (Sec. 3.2, 5) but
never defines [C].
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

CORE4 = ("age_days", "follower_ratio", "tweet_freq", "username_entropy")
EXTENDED6 = CORE4 + ("reputation", "engagement")


def username_entropy(name: str) -> float:
    """Shannon entropy (bits) of the character distribution of the screen name."""
    if not name:
        return float("nan")
    counts = Counter(name)
    n = len(name)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def raw_features(meta: pd.DataFrame, feature_set: str = "core4") -> pd.DataFrame:
    """Untransformed feature values phi_v. NaN where inputs are missing."""
    created = meta["created_at"]
    ref = created.max()  # [C] reference time T = latest creation time in the dataset (label-free)
    age = (ref - created).dt.total_seconds() / 86400.0
    a = meta["followers_count"].astype(float)
    b = meta["friends_count"].astype(float)
    s = meta["statuses_count"].astype(float)
    f = meta["favourites_count"].astype(float)
    out = pd.DataFrame({
        "age_days": age,
        "follower_ratio": a / (b + 1.0),
        "tweet_freq": s / (age + 1.0),
        "username_entropy": meta["screen_name"].map(username_entropy).astype(float),
    })
    if feature_set == "extended6":
        out["reputation"] = a / (a + b + 1.0)
        out["engagement"] = f / (s + 1.0)
    elif feature_set != "core4":
        raise ValueError(feature_set)
    # Infinite values (e.g. from corrupt counts) are treated as missing and imputed later.
    return out.replace([np.inf, -np.inf], np.nan)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class MetadataTransformer:
    """Median imputation -> signed log1p -> z-score. Every statistic is fitted
    only on the rows passed to fit() (training rows in the corrected protocol)."""

    def fit(self, phi: np.ndarray, fit_idx: np.ndarray) -> "MetadataTransformer":
        phi = np.where(np.isfinite(phi), phi, np.nan)
        rows = phi[fit_idx]
        self.median_ = np.nanmedian(rows, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        g = signed_log1p(self._impute(rows))
        self.mean_ = g.mean(axis=0)
        sd = g.std(axis=0)
        self.std_ = np.where(sd > 0, sd, 1.0)
        self.fit_idx_size_ = int(len(fit_idx))
        return self

    def _impute(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        mask = np.isnan(x)
        x[mask] = np.take(self.median_, np.where(mask)[1])
        return x

    def transform(self, phi: np.ndarray) -> np.ndarray:
        phi = np.where(np.isfinite(phi), phi, np.nan)
        return (signed_log1p(self._impute(phi)) - self.mean_) / self.std_


def metadata_matrix(meta: pd.DataFrame, fit_idx: np.ndarray, feature_set: str = "core4"):
    phi = raw_features(meta, feature_set).to_numpy(dtype=float)
    tr = MetadataTransformer().fit(phi, fit_idx)
    return tr.transform(phi).astype(np.float32), tr
