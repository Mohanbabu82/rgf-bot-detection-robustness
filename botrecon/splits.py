"""Train / validation / test splits. Seeds make every split reproducible."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .data.common import AccountData


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    kind: str

    def check(self, n: int) -> None:
        a, b, c = set(self.train), set(self.val), set(self.test)
        assert not (a & b) and not (a & c) and not (b & c), "split sets overlap"
        assert len(a) + len(b) + len(c) <= n


def stratified_70_10_20(y: np.ndarray, seed: int) -> Split:
    """Manuscript's stratified 80/20 [M], with a validation slice (1/8 of the 80%) carved
    from the training part [C] because early stopping needs one."""
    idx = np.arange(len(y))
    trval, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    tr, va = train_test_split(trval, test_size=0.125, stratify=y[trval], random_state=seed)
    s = Split(np.sort(tr), np.sort(va), np.sort(te), "stratified_70_10_20")
    s.check(len(y))
    return s


def official(data: AccountData) -> Split:
    sp = data.official_split
    if sp is None:
        raise ValueError(f"{data.name} has no official split")
    s = Split(np.where(sp == "train")[0], np.where(sp == "val")[0], np.where(sp == "test")[0], "official")
    s.check(data.n)
    return s


def make_split(data: AccountData, cfg, seed: int) -> Split:
    if data.name == "twibot20" and cfg.twibot_split == "official":
        return official(data)
    return stratified_70_10_20(data.y, seed)


def blend_halves(train_idx: np.ndarray, y: np.ndarray, seed: int):
    """Stratified halves of the training set: A trains the GCN, B trains LightGBM."""
    a, b = train_test_split(train_idx, test_size=0.5, stratify=y[train_idx], random_state=seed)
    return np.sort(a), np.sort(b)


def kfold(train_idx: np.ndarray, y: np.ndarray, seed: int, k: int = 5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for _, held in skf.split(train_idx, y[train_idx]):
        yield np.sort(train_idx[held])
