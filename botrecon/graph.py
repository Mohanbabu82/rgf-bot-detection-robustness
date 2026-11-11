"""Graph construction and GCN normalisation (numpy / scipy only).

Edge convention: edge_index[0] = source, edge_index[1] = target; messages flow
source -> target. Normalisation follows the standard GCN rule (as in PyTorch
Geometric's gcn_norm): add self-loops, degree counted at the target,
A_hat[t, s] = deg(s)^-1/2 * deg(t)^-1/2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def symmetrise(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return edge_index.reshape(2, 0).astype(np.int64)
    both = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    both = both[:, both[0] != both[1]]
    return np.unique(both, axis=1).astype(np.int64)


def knn_edges(z: np.ndarray, k: int, base_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    """Directed edges neighbour -> query: each query node receives edges from its k most
    cosine-similar nodes among base_idx (excluding itself)."""
    if len(query_idx) == 0:
        return np.zeros((2, 0), np.int64)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(base_idx)), metric="cosine", algorithm="brute")
    nn.fit(z[base_idx])
    _, nbr = nn.kneighbors(z[query_idx])
    src, dst = [], []
    for qi, row in zip(query_idx, nbr):
        cand = [base_idx[j] for j in row if base_idx[j] != qi][:k]
        src.extend(cand)
        dst.extend([qi] * len(cand))
    return np.array([src, dst], dtype=np.int64)


@dataclass
class GraphPair:
    """Training graph (what the GCN sees while fitting) and inference graph
    (what it sees when producing outputs). Identical in the transductive setting."""
    train_edges: np.ndarray
    infer_edges: np.ndarray
    train_nodes: np.ndarray     # nodes present while fitting
    setting: str


def transductive_knn(z: np.ndarray, k: int) -> GraphPair:
    """[M] kNN over all nodes (labels are never used), symmetrised [C]."""
    allv = np.arange(len(z))
    e = symmetrise(knn_edges(z, k, allv, allv))
    return GraphPair(e, e, allv, "transductive")


def inductive_knn(z: np.ndarray, k: int, seen: np.ndarray, unseen: np.ndarray) -> GraphPair:
    """[C] Graph among seen (train+val) nodes only; unseen (test) nodes are attached at
    inference by directed edges from their k nearest SEEN nodes. No unseen-unseen edges,
    and seen nodes never receive messages from unseen nodes."""
    seen = np.sort(seen)
    e_seen = symmetrise(knn_edges(z, k, seen, seen))
    e_new = knn_edges(z, k, seen, np.sort(unseen))
    infer = np.concatenate([e_seen, e_new], axis=1)
    return GraphPair(e_seen, infer, seen, "inductive")


def transductive_follow(edges: np.ndarray, n: int) -> GraphPair:
    """[C] Follower and following merged into one undirected relation."""
    e = symmetrise(edges)
    return GraphPair(e, e, np.arange(n), "transductive")


def inductive_follow(edges: np.ndarray, n: int, seen: np.ndarray, unseen: np.ndarray) -> GraphPair:
    seen_mask = np.zeros(n, bool); seen_mask[seen] = True
    e = symmetrise(edges)
    s, t = e
    keep_train = seen_mask[s] & seen_mask[t]
    new_in = seen_mask[s] & ~seen_mask[t]          # seen -> unseen only
    infer = np.concatenate([e[:, keep_train], e[:, new_in]], axis=1)
    return GraphPair(e[:, keep_train], infer, np.sort(seen), "inductive")


def normalised_adjacency(edge_index: np.ndarray, n: int, nodes: Optional[np.ndarray] = None) -> sp.csr_matrix:
    """A_hat (n x n, rows = targets) with self-loops on `nodes` (default: all)."""
    if nodes is None:
        nodes = np.arange(n)
    s = np.concatenate([edge_index[0], nodes]).astype(np.int64)
    t = np.concatenate([edge_index[1], nodes]).astype(np.int64)
    w = np.ones(len(s), dtype=np.float64)
    deg = np.bincount(t, weights=w, minlength=n)
    dinv = np.zeros(n); nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    vals = dinv[s] * w * dinv[t]
    return sp.csr_matrix((vals, (t, s)), shape=(n, n))


def edge_homophily(edge_index: np.ndarray, y: np.ndarray, nodes: np.ndarray) -> float:
    """Share of edges joining same-label nodes, counted only among `nodes`
    (pass training nodes so no test label is read)."""
    m = np.zeros(len(y), bool); m[nodes] = True
    s, t = edge_index
    keep = m[s] & m[t]
    if keep.sum() == 0:
        return float("nan")
    return float((y[s[keep]] == y[t[keep]]).mean())
