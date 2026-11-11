"""GCN stage of the reconstruction runner (botrecon.run / botrecon.pipeline).

Architecture [M 3.1]: GCN(d -> 64) -> ReLU -> Dropout(0.5) -> GCN(64 -> 2), i.e. the SAME
model class as botrecon.nn.GCN (one implementation, used by both the reconstruction
runner and the experiment suite). Propagation uses the given sparse normalised
adjacency D^-1/2 (A + I) D^-1/2, the same computation as PyTorch Geometric's GCNConv.

Outputs are kept apart and named explicitly:
  hidden : (N, 64) node embeddings after ReLU, before dropout, eval mode
  prob   : (N, 2) class probabilities (softmax of the second layer)
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class GCNResult:
    hidden: np.ndarray      # (N, 64) after ReLU, eval mode
    prob: np.ndarray        # (N, 2) softmax
    best_epoch: int
    seconds: float
    val_loss_selected: float = float("nan")   # validation loss of the returned model
    val_loss_last: float = float("nan")       # validation loss after the final epoch


def _sparse(torch, a, device):
    a = a.tocoo()
    idx = torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long)
    val = torch.tensor(a.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, a.shape).coalesce().to(device)


class _FixedGraph:
    """Minimal stand-in for botrecon.nn.Bundle: node features plus one fixed adjacency."""

    def __init__(self, x, adj):
        self.x, self._adj = x, adj

    def get(self, kind):
        assert kind == "gcn", kind
        return self._adj


def train_gcn(x: np.ndarray, a_train, a_infer, y_view: np.ndarray, fit_idx: np.ndarray,
              val_idx: np.ndarray, cfg, seed: int, device: str | None = None) -> GCNResult:
    """y_view must hold -1 for every node whose label may not be used.
    Loss uses fit_idx only; model selection (if any) uses val_idx only.
    cfg.gcn_selection: "fixed" = model after the last epoch; "val_loss" = epoch with the lowest
    validation loss (validation labels only)."""
    import torch
    import torch.nn.functional as F
    from .nn import GCN

    assert (y_view[fit_idx] >= 0).all(), "GCN fit set contains masked labels"
    has_val = len(val_idx) > 0 and (y_view[val_idx] >= 0).all()
    if cfg.gcn_selection == "val_loss":
        assert has_val, "validation set contains masked labels"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    model = GCN(x.shape[1], cfg.gcn_hidden, cfg.gcn_dropout).to(device)   # created right after seeding
    X = torch.tensor(x, dtype=torch.float32, device=device)
    G_train = _FixedGraph(X, _sparse(torch, a_train, device))
    G_infer = _FixedGraph(X, _sparse(torch, a_infer, device))
    y = torch.tensor(np.where(y_view < 0, 0, y_view), dtype=torch.long, device=device)
    fit = torch.tensor(fit_idx, dtype=torch.long, device=device)
    val = torch.tensor(val_idx, dtype=torch.long, device=device)

    def val_loss():
        model.eval()
        with torch.no_grad():
            return F.cross_entropy(model(G_train)["logits"][val], y[val]).item()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.gcn_lr, weight_decay=cfg.gcn_weight_decay)
    t0 = time.time()
    best_state, best_loss, best_epoch = None, float("inf"), cfg.gcn_epochs
    for epoch in range(1, cfg.gcn_epochs + 1):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(G_train)["logits"][fit], y[fit])
        loss.backward(); opt.step()
        if cfg.gcn_selection == "val_loss":
            vl = val_loss()
            if vl < best_loss:
                best_loss, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
    last = val_loss() if has_val else float("nan")
    if best_state is not None:
        model.load_state_dict(best_state)
    selected = val_loss() if has_val else float("nan")
    model.eval()
    with torch.no_grad():
        out = model(G_infer)
    return GCNResult(out["hidden"].cpu().numpy(), torch.softmax(out["logits"], 1).cpu().numpy(),
                     best_epoch, time.time() - t0, selected, last)
