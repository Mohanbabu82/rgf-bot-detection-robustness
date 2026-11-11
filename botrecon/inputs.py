"""Model inputs and test-time perturbations (numpy only).

An Inputs object is everything a trained model sees at prediction time. The
robustness experiments perturb a copy of it for TEST nodes only, then call the
same trained model again.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .graph import symmetrise


@dataclass
class Inputs:
    m: np.ndarray                     # (N, d_m) standardised metadata
    t: np.ndarray                     # (N, d_t) text embedding (0 where unavailable)
    text_avail: np.ndarray            # (N,) bool
    sym_edges: np.ndarray             # (2, E) symmetric, source -> target
    directed: Optional[np.ndarray] = None   # (2, E') i -> j means i follows j (TwiBot-20 only)

    @property
    def n(self) -> int:
        return self.m.shape[0]

    @property
    def x(self) -> np.ndarray:
        return np.hstack([self.m, self.t]).astype(np.float32)

    def degree(self) -> np.ndarray:
        return np.bincount(self.sym_edges[1], minlength=self.n) if self.sym_edges.size else np.zeros(self.n, int)

    def copy(self) -> "Inputs":
        return replace(self, m=self.m.copy(), t=self.t.copy(), text_avail=self.text_avail.copy(),
                       sym_edges=self.sym_edges.copy(),
                       directed=None if self.directed is None else self.directed.copy())


# ---------------- perturbations (TEST nodes only) ----------------

def missing_text(inp: Inputs, test_idx: np.ndarray, rate: float, rng) -> Inputs:
    """Remove the text of a random `rate` share of test accounts."""
    out = inp.copy()
    sel = rng.choice(test_idx, int(round(rate * len(test_idx))), replace=False)
    out.t[sel] = 0.0
    out.text_avail[sel] = False
    return out


def _incident(edges: np.ndarray, nodes_mask: np.ndarray) -> np.ndarray:
    return nodes_mask[edges[0]] | nodes_mask[edges[1]]


def drop_edges(inp: Inputs, test_idx: np.ndarray, rate: float, rng) -> Inputs:
    """Remove each undirected edge touching a test account with probability `rate`."""
    out = inp.copy()
    mask = np.zeros(inp.n, bool); mask[test_idx] = True
    s, t = inp.sym_edges
    und = inp.sym_edges[:, s < t]
    inc = _incident(und, mask)
    keep = ~inc | (rng.random(und.shape[1]) >= rate)
    out.sym_edges = symmetrise(und[:, keep])
    if inp.directed is not None:
        n = np.int64(inp.n)
        kept = und[0, keep].astype(np.int64) * n + und[1, keep]
        d = inp.directed
        key = np.minimum(d[0], d[1]).astype(np.int64) * n + np.maximum(d[0], d[1])
        out.directed = d[:, np.isin(key, kept)]
    return out


def noisy_edges(inp: Inputs, test_idx: np.ndarray, rate: float, rng) -> Inputs:
    """Add random edges from test accounts to random accounts; the number added is
    `rate` x the number of existing undirected edges that touch test accounts."""
    out = inp.copy()
    mask = np.zeros(inp.n, bool); mask[test_idx] = True
    s, t = inp.sym_edges
    und = inp.sym_edges[:, s < t]
    k = int(round(rate * _incident(und, mask).sum()))
    if k == 0:
        return out
    a = rng.choice(test_idx, k, replace=True)
    b = rng.integers(0, inp.n, k)
    keep = a != b
    new = np.vstack([a[keep], b[keep]])
    out.sym_edges = symmetrise(np.concatenate([und, new], axis=1))
    if inp.directed is not None:
        out.directed = np.unique(np.concatenate([inp.directed, new], axis=1), axis=1)
    return out


def metadata_mimicry(inp: Inputs, test_idx: np.ndarray, y_test: np.ndarray, rate: float,
                     human_reference: np.ndarray, rng) -> Inputs:
    """Evasion attack: a `rate` share of test BOTS replace their metadata with the
    median standardised metadata of human TRAINING accounts. Test labels are used
    only to choose which accounts attack, as an attacker would know it is a bot."""
    out = inp.copy()
    bots = test_idx[y_test == 1]
    sel = rng.choice(bots, int(round(rate * len(bots))), replace=False)
    out.m[sel] = human_reference
    return out


PERTURBATIONS = {
    "missing_text": [0.0, 0.25, 0.5, 0.75, 1.0],
    "drop_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
    "noisy_edges": [0.0, 0.25, 0.5, 1.0],
    "metadata_mimicry": [0.0, 0.25, 0.5, 1.0],
}
