"""Neural models in plain PyTorch (CPU-friendly; no PyTorch Geometric needed).

  GCN        2-layer GCN on x = [m || t]                    (baseline; also the hybrid's stage 1)
  SAGE       2-layer GraphSAGE, mean aggregator             (baseline)
  GAT        2-layer GAT, 4 heads x 16                      (baseline)
  RGCNBot    BotRGCN-style relational GCN, follower/following relations (TwiBot-20 baseline;
             our re-implementation on our features, not the authors' code)
  RGF        proposed Reliability-Gated Fusion (see class docstring)

Written without being executed (no PyTorch where written). tests/test_real_backends.py
checks shapes, gradients, learning and label discipline on Kaggle before any run.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inputs import Inputs


# ---------------------------------------------------------------- graph tensors

def _sp(rows, cols, vals, n, device):
    idx = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(vals, dtype=torch.float32), (n, n)).coalesce().to(device)


def gcn_adj(edges: np.ndarray, n: int, device) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2, rows = targets."""
    s = np.concatenate([edges[0], np.arange(n)]); t = np.concatenate([edges[1], np.arange(n)])
    deg = np.bincount(t, minlength=n).astype(np.float64)
    dinv = deg ** -0.5
    return _sp(t, s, dinv[s] * dinv[t], n, device)


def mean_adj(edges: np.ndarray, n: int, device) -> torch.Tensor:
    """Row-normalised neighbour mean without self-loops; isolated rows are zero."""
    s, t = edges
    deg = np.bincount(t, minlength=n).astype(np.float64)
    w = np.zeros(len(t)) if len(t) == 0 else 1.0 / deg[t]
    return _sp(t, s, w, n, device)


class Bundle:
    """Tensors for one Inputs object (built once, reused for every forward pass)."""

    def __init__(self, inp: Inputs, device):
        self.n = inp.n
        self.device = device
        self.m = torch.as_tensor(inp.m, dtype=torch.float32, device=device)
        self.t = torch.as_tensor(inp.t, dtype=torch.float32, device=device)
        self.x = torch.cat([self.m, self.t], 1)
        self.a_t = torch.as_tensor(inp.text_avail.astype(np.float32), device=device)
        deg = inp.degree()
        self.deg = torch.as_tensor(deg.astype(np.float32), device=device)
        self.a_g = (self.deg > 0).float()
        self._edges = inp.sym_edges
        self._directed = inp.directed
        self._cache = {}

    def get(self, kind: str):
        if kind not in self._cache:
            n, e, dev = self.n, self._edges, self.device
            if kind == "gcn":
                self._cache[kind] = gcn_adj(e, n, dev)
            elif kind == "mean":
                self._cache[kind] = mean_adj(e, n, dev)
            elif kind == "gat":
                loops = np.arange(n)
                src = np.concatenate([e[0], loops]); dst = np.concatenate([e[1], loops])
                self._cache[kind] = (torch.as_tensor(src, device=dev), torch.as_tensor(dst, device=dev))
            elif kind == "rel":
                d = self._directed
                if d is None:
                    raise ValueError("relational edges unavailable")
                # relation 0: messages from accounts i follows (j -> i); relation 1: from followers (i -> j)
                self._cache[kind] = [mean_adj(d[::-1], n, dev), mean_adj(d, n, dev)]
            else:
                raise KeyError(kind)
        return self._cache[kind]


# ---------------------------------------------------------------- models

class GCN(nn.Module):
    def __init__(self, d, h=64, p=0.5):
        super().__init__()
        self.l0 = nn.Linear(d, h, bias=False); self.b0 = nn.Parameter(torch.zeros(h))
        self.l1 = nn.Linear(h, 2, bias=False); self.b1 = nn.Parameter(torch.zeros(2))
        self.p = p

    def forward(self, B: Bundle):
        A = B.get("gcn")
        h = F.relu(torch.sparse.mm(A, self.l0(B.x)) + self.b0)
        z = torch.sparse.mm(A, self.l1(F.dropout(h, self.p, self.training))) + self.b1
        return {"logits": z, "hidden": h}


class SAGE(nn.Module):
    def __init__(self, d, h=64, p=0.5):
        super().__init__()
        self.s0, self.n0 = nn.Linear(d, h), nn.Linear(d, h, bias=False)
        self.s1, self.n1 = nn.Linear(h, 2), nn.Linear(h, 2, bias=False)
        self.p = p

    def forward(self, B: Bundle):
        A = B.get("mean")
        h = F.relu(self.s0(B.x) + self.n0(torch.sparse.mm(A, B.x)))
        h = F.dropout(h, self.p, self.training)
        return {"logits": self.s1(h) + self.n1(torch.sparse.mm(A, h))}


class GATLayer(nn.Module):
    def __init__(self, d, f, heads, concat=True, p_att=0.2):
        super().__init__()
        self.lin = nn.Linear(d, heads * f, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, f)); self.a_dst = nn.Parameter(torch.empty(1, heads, f))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.bias = nn.Parameter(torch.zeros(heads * f if concat else f))
        self.h, self.f, self.concat, self.p_att = heads, f, concat, p_att

    def forward(self, x, src, dst):
        n = x.shape[0]
        wh = self.lin(x).view(n, self.h, self.f)
        e = F.leaky_relu((wh * self.a_src).sum(-1)[src] + (wh * self.a_dst).sum(-1)[dst], 0.2)   # (E, H)
        idx = dst.unsqueeze(1).expand(-1, self.h)
        emax = torch.full((n, self.h), float("-inf"), device=x.device).scatter_reduce(
            0, idx, e, reduce="amax", include_self=True)
        ex = torch.exp(e - emax[dst])
        denom = torch.zeros(n, self.h, device=x.device).index_add_(0, dst, ex)
        alpha = F.dropout(ex / denom[dst].clamp_min(1e-16), self.p_att, self.training)
        out = torch.zeros(n, self.h, self.f, device=x.device).index_add_(0, dst, alpha.unsqueeze(-1) * wh[src])
        out = out.reshape(n, self.h * self.f) if self.concat else out.mean(1)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, d, heads=4, f=16, p=0.5):
        super().__init__()
        self.g0 = GATLayer(d, f, heads, concat=True)
        self.g1 = GATLayer(heads * f, 2, 1, concat=False)
        self.p = p

    def forward(self, B: Bundle):
        src, dst = B.get("gat")
        x = F.dropout(B.x, self.p, self.training)
        h = F.elu(self.g0(x, src, dst))
        h = F.dropout(h, self.p, self.training)
        return {"logits": self.g1(h, src, dst)}


class RGCNBot(nn.Module):
    """BotRGCN-style: input projection, two relational layers (self + per-relation mean), MLP head."""

    def __init__(self, d, h=64, p=0.3):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.self0, self.rel0 = nn.Linear(h, h), nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(2)])
        self.self1, self.rel1 = nn.Linear(h, h), nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(2)])
        self.out1, self.out2 = nn.Linear(h, h), nn.Linear(h, 2)
        self.p = p

    def _layer(self, h, rels, self_lin, rel_lins):
        z = self_lin(h)
        for A, lin in zip(rels, rel_lins):
            z = z + lin(torch.sparse.mm(A, h))
        return z

    def forward(self, B: Bundle):
        rels = B.get("rel")
        h = F.leaky_relu(self.inp(B.x))
        h = F.dropout(self._layer(h, rels, self.self0, self.rel0), self.p, self.training)
        h = self._layer(h, rels, self.self1, self.rel1)
        return {"logits": self.out2(F.leaky_relu(self.out1(h)))}


class RGF(nn.Module):
    """Reliability-Gated Fusion (proposed).

    Three modality experts each produce bot/human logits:
      metadata expert  z_m = head_m(enc_m(m))
      text expert      z_t = head_t(enc_t(t))                      available iff the account has text
      graph expert     z_g = head_g(two-hop neighbour mean of [enc_m || enc_t])
                                                                    available iff degree > 0
    A gate maps reliability signals r_v = [a_t, a_g, log(1+deg)/5, H(p_m), H(p_t), H(p_g)]
    (availability, degree, expert entropies) to weights w_v over the AVAILABLE experts
    (masked softmax). Fused logits z = sum_k w_k z_k, divided by a temperature T fitted on
    validation data. Training adds per-expert auxiliary losses and random modality dropout,
    so the model learns to classify when text or graph is missing.

    Ablation switches: gate (False -> uniform mean of available experts), experts subset,
    p_drop (0 -> no modality dropout), aux (0 -> no auxiliary losses), fusion="concat"
    (single head on concatenated expert representations, a naive-fusion control).
    """

    def __init__(self, d_m, d_t, h=64, p=0.3, experts=("m", "t", "g"), gate=True,
                 p_drop=0.3, aux=0.5, fusion="gate"):
        super().__init__()
        self.experts, self.gate_on, self.p_drop, self.aux, self.fusion, self.p = experts, gate, p_drop, aux, fusion, p
        self.enc_m = nn.Sequential(nn.Linear(d_m, h), nn.LeakyReLU(), nn.Dropout(p))
        self.enc_t = nn.Sequential(nn.Linear(d_t, h), nn.LeakyReLU(), nn.Dropout(p))
        self.g1, self.g2 = nn.Linear(2 * h, h), nn.Linear(h, h)
        self.head = nn.ModuleDict({k: nn.Linear(h, 2) for k in ("m", "t", "g")})
        self.gate = nn.Sequential(nn.Linear(6, 32), nn.LeakyReLU(), nn.Linear(32, 3))
        self.concat_head = nn.Sequential(nn.Linear(3 * h, h), nn.LeakyReLU(), nn.Dropout(p), nn.Linear(h, 2))
        self.register_buffer("temperature", torch.ones(1))

    @staticmethod
    def _entropy(z):
        p = torch.softmax(z.detach(), 1)
        return -(p * torch.log(p.clamp_min(1e-12))).sum(1)

    def forward(self, B: Bundle):
        a_t, a_g = B.a_t, B.a_g
        if self.training and self.p_drop > 0:                 # modality dropout
            a_t = a_t * (torch.rand_like(a_t) >= self.p_drop).float()
            a_g = a_g * (torch.rand_like(a_g) >= self.p_drop).float()
        hm = self.enc_m(B.m)
        ht = self.enc_t(B.t) * a_t.unsqueeze(1)
        A = B.get("mean")
        s0 = torch.cat([hm, ht], 1)
        hg = F.leaky_relu(self.g1(torch.sparse.mm(A, s0)))
        hg = F.dropout(hg, self.p, self.training)
        hg = F.leaky_relu(self.g2(torch.sparse.mm(A, hg))) * a_g.unsqueeze(1)
        reps = {"m": hm, "t": ht, "g": hg}
        z = {k: self.head[k](reps[k]) for k in ("m", "t", "g")}
        avail = {"m": torch.ones_like(a_t), "t": a_t, "g": a_g}
        # on[v, k] = 1 if expert k is enabled in this variant AND its modality is available for v.
        # Accounts with no usable expert get zero logits (p = 0.5); an excluded expert is never used.
        on = torch.stack([avail[k] if k in self.experts else torch.zeros_like(a_t) for k in ("m", "t", "g")], 1)
        if self.fusion == "concat":
            parts = [reps[k] * (avail[k].unsqueeze(1) if k in self.experts else 0.0) for k in ("m", "t", "g")]
            fused = self.concat_head(torch.cat(parts, 1))
            w = on / on.sum(1, keepdim=True).clamp_min(1.0)
        else:
            if self.gate_on:
                r = torch.stack([a_t, a_g, torch.log1p(B.deg) / 5.0,
                                 self._entropy(z["m"]), self._entropy(z["t"]), self._entropy(z["g"])], 1)
                w = torch.softmax(self.gate(r).masked_fill(on == 0, -1e9), 1) * on
                w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
            else:
                w = on / on.sum(1, keepdim=True).clamp_min(1.0)
            fused = sum(w[:, i:i + 1] * z[k] for i, k in enumerate(("m", "t", "g")))
        return {"logits": fused / self.temperature, "experts": z, "avail": avail, "weights": w}

    def aux_loss(self, out, idx, y):
        if self.aux <= 0 or self.fusion == "concat":
            return 0.0
        loss = 0.0
        for k in self.experts:
            mask = out["avail"][k][idx] > 0
            if mask.any():
                loss = loss + F.cross_entropy(out["experts"][k][idx][mask], y[idx][mask])
        return self.aux * loss


# ---------------------------------------------------------------- training

def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def fit(model: nn.Module, B: Bundle, y_view: np.ndarray, fit_idx: np.ndarray, val_idx: np.ndarray,
        seed: int, lr=0.01, wd=5e-4, epochs=200, patience=30, select="val_loss") -> dict:
    """Full-batch training. Loss on fit_idx only; model selection on val_idx only.
    y_view must hold -1 for labels that may not be used (all test labels)."""
    assert (y_view[fit_idx] >= 0).all(), "fit set contains masked labels"
    assert (y_view[val_idx] >= 0).all(), "validation set contains masked labels"
    torch.manual_seed(seed); np.random.seed(seed)
    dev = B.device
    y = torch.as_tensor(np.where(y_view < 0, 0, y_view), dtype=torch.long, device=dev)
    fi = torch.as_tensor(fit_idx, dtype=torch.long, device=dev)
    vi = torch.as_tensor(val_idx, dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_loss, best_ep, t0 = None, float("inf"), 0, time.time()
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        out = model(B)
        loss = F.cross_entropy(out["logits"][fi], y[fi])
        if hasattr(model, "aux_loss"):
            loss = loss + model.aux_loss(out, fi, y)
        loss.backward(); opt.step()
        if select == "val_loss":
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(model(B)["logits"][vi], y[vi]).item()
            if vl < best_loss:
                best_loss, best_ep, best = vl, ep, copy.deepcopy(model.state_dict())
            elif patience and ep - best_ep >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return {"best_epoch": best_ep if select == "val_loss" else epochs, "seconds": time.time() - t0,
            "params": count_params(model)}


def fit_temperature(model: RGF, B: Bundle, y_view: np.ndarray, val_idx: np.ndarray) -> float:
    """Grid search for the temperature T minimising validation NLL (no test data)."""
    assert (y_view[val_idx] >= 0).all()
    model.eval()
    model.temperature.fill_(1.0)
    with torch.no_grad():
        z = model(B)["logits"][torch.as_tensor(val_idx, device=B.device)].cpu()
    y = torch.as_tensor(y_view[val_idx], dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.25), np.log(5.0), 60))
    nll = [F.cross_entropy(z / T, y).item() for T in grid]
    T = float(grid[int(np.argmin(nll))])
    model.temperature.fill_(T)
    return T


@torch.no_grad()
def predict(model: nn.Module, B: Bundle) -> dict:
    model.eval()
    out = model(B)
    res = {"p": torch.softmax(out["logits"], 1)[:, 1].cpu().numpy()}
    if "hidden" in out:
        res["hidden"] = out["hidden"].cpu().numpy()
        res["prob2"] = torch.softmax(out["logits"], 1).cpu().numpy()
    if "weights" in out:
        res["weights"] = out["weights"].cpu().numpy()
        res["expert_p"] = {k: torch.softmax(v, 1)[:, 1].cpu().numpy() for k, v in out["experts"].items()}
    return res
