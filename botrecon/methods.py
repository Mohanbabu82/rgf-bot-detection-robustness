"""Every method compared in the paper, behind one interface:

    m = make(name, dataset); info = m.fit(train_inputs, y_view, split, seed); out = m.predict(inputs)

fit() sees labels only through y_view (test labels = -1). predict() takes any Inputs,
including perturbed ones, and returns {"p": P(bot) for every account, ...}.
"""
from __future__ import annotations

import time

import numpy as np

from . import config as C
from . import provenance as P
from .inputs import Inputs
from .splits import blend_halves

THIS_WORK = P.THIS_WORK


class Method:
    name = ""
    group = ""                    # tabular | gnn | hybrid | proposed | ablation
    provenance = THIS_WORK
    needs_directed = False

    def fit(self, inp: Inputs, y_view, split, seed, device="cpu") -> dict: ...
    def predict(self, inp: Inputs) -> dict: ...

    # Checkpoints: methods that can be saved implement both; others return False from save().
    def save(self, path: str) -> bool:
        return False

    def load(self, path: str, device: str = "cpu") -> "Method":
        raise NotImplementedError(f"{self.name} has no checkpoint support")


# ---------------------------------------------------------------- tabular

class Tabular(Method):
    group = "tabular"

    def __init__(self, model: str, features: str):
        self.model_kind, self.features = model, features
        self.group = "tabular" if (model, features) == ("lgbm", "meta+text") or model != "lgbm" else "modality"
        names = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}
        self.name = f"{names[model]} ({features})"

    def _X(self, inp: Inputs):
        return {"meta": inp.m, "text": inp.t, "meta+text": inp.x}[self.features]

    def fit(self, inp, y_view, split, seed, device="cpu"):
        X, t0 = self._X(inp), time.time()
        tr, va = split.train, split.val
        if self.model_kind == "lgbm":
            import lightgbm as lgb
            self.clf = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=31, learning_rate=0.05,
                                          n_estimators=1000, objective="binary", random_state=seed, verbose=-1)
            self.clf.fit(X[tr], y_view[tr], eval_set=[(X[va], y_view[va])], eval_metric="binary_logloss",
                         callbacks=[lgb.early_stopping(50, verbose=False)])
            size = int(self.clf.best_iteration_ or 1000)
        elif self.model_kind == "xgb":
            from xgboost import XGBClassifier
            self.clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.9,
                                     colsample_bytree=0.9, eval_metric="logloss", early_stopping_rounds=50,
                                     random_state=seed, n_jobs=-1, tree_method="hist")
            self.clf.fit(X[tr], y_view[tr], eval_set=[(X[va], y_view[va])], verbose=False)
            size = int(getattr(self.clf, "best_iteration", 999)) + 1
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed,
                                              class_weight=None).fit(X[tr], y_view[tr])
            size = 500
        return {"seconds": time.time() - t0, "size": size, "best_epoch": None}

    def predict(self, inp):
        X = self._X(inp)
        if getattr(self, "booster", None) is not None:          # loaded from a LightGBM checkpoint
            return {"p": self.booster.predict(X)}
        return {"p": self.clf.predict_proba(X)[:, 1]}

    def save(self, path):
        if self.model_kind == "lgbm":
            with open(path + ".lgbm.txt", "w") as f:
                f.write(self.clf.booster_.model_to_string(num_iteration=self.clf.best_iteration_))
            return True
        if self.model_kind == "xgb":
            self.clf.save_model(path + ".xgb.json")
            return True
        return False                                             # Random Forest: not checkpointed (size)

    def load(self, path, device="cpu"):
        if self.model_kind == "lgbm":
            import lightgbm as lgb
            self.booster, self.clf = lgb.Booster(model_file=path + ".lgbm.txt"), None
        elif self.model_kind == "xgb":
            from xgboost import XGBClassifier
            self.clf = XGBClassifier(); self.clf.load_model(path + ".xgb.json")
        else:
            raise NotImplementedError("Random Forest checkpoints are not saved")
        return self

    def importances(self):
        return getattr(self.clf, "feature_importances_", None)


# ---------------------------------------------------------------- graph neural baselines

class GNN(Method):
    """Graph neural baselines. `features` selects the node features: "meta+text" (default),
    or "meta" / "text" for the modality ablations Metadata + GCN and Text + GCN."""

    def __init__(self, kind: str, features: str = "meta+text"):
        assert features in ("meta+text", "meta", "text")
        self.kind, self.features = kind, features
        base = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "rgcn": "BotRGCN-style R-GCN"}[kind]
        self.name = base if features == "meta+text" else f"{base} ({features})"
        self.group = "gnn" if features == "meta+text" else "modality"
        self.needs_directed = kind == "rgcn"

    def _view(self, inp: Inputs) -> Inputs:
        if self.features == "meta+text":
            return inp
        empty = np.zeros((inp.n, 0), np.float32)
        return Inputs(inp.m if self.features == "meta" else empty, inp.t if self.features == "text" else empty,
                      inp.text_avail, inp.sym_edges, inp.directed)

    def fit(self, inp, y_view, split, seed, device="cpu"):
        import torch
        from . import nn as N
        torch.manual_seed(seed)
        self.device = device
        inp = self._view(inp)
        d = inp.x.shape[1]
        self.model = {"gcn": lambda: N.GCN(d), "sage": lambda: N.SAGE(d), "gat": lambda: N.GAT(d),
                      "rgcn": lambda: N.RGCNBot(d)}[self.kind]().to(device)
        self.d = d
        lr = 0.005 if self.kind == "gat" else 0.01
        info = N.fit(self.model, N.Bundle(inp, device), y_view, split.train, split.val, seed, lr=lr)
        info["size"] = info.pop("params")
        return info

    def _build(self, d):
        from . import nn as N
        return {"gcn": lambda: N.GCN(d), "sage": lambda: N.SAGE(d), "gat": lambda: N.GAT(d),
                "rgcn": lambda: N.RGCNBot(d)}[self.kind]()

    def save(self, path):
        import torch
        torch.save({"kind": self.kind, "features": self.features, "d": self.d,
                    "state": self.model.state_dict()}, path + ".pt")
        return True

    def load(self, path, device="cpu"):
        import torch
        ck = torch.load(path + ".pt", map_location=device)
        assert (ck["kind"], ck["features"]) == (self.kind, self.features), "checkpoint is for another method"
        self.device, self.d = device, ck["d"]
        self.model = self._build(ck["d"]).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval()
        return self

    def predict(self, inp):
        from . import nn as N
        return {"p": N.predict(self.model, N.Bundle(self._view(inp), self.device))["p"]}


# ---------------------------------------------------------------- reconstructed original (corrected protocol)

class Hybrid(Method):
    """Manuscript architecture reconstructed: GCN (2 layers, 64 hidden, dropout 0.5, AdamW 0.01 / 5e-4,
    <= 100 epochs, best validation loss) trained on half A of the training set; LightGBM on the GCN
    output for half B (blending), early-stopped on validation."""
    group = "hybrid"
    provenance = P.REPRODUCED_CORRECTED

    def __init__(self, fusion: str, tap: str):
        self.fusion, self.tap = fusion, tap
        self.name = f"Reconstructed hybrid ({fusion}, {tap})"

    def fit(self, inp, y_view, split, seed, device="cpu"):
        import lightgbm as lgb
        import torch
        from . import nn as N
        torch.manual_seed(seed)
        self.device = device
        a, b = blend_halves(split.train, y_view, seed)
        self.gcn = N.GCN(inp.x.shape[1]).to(device)
        info = N.fit(self.gcn, N.Bundle(inp, device), y_view, a, split.val, seed, lr=0.01, epochs=100, patience=None)
        q = self._q(inp)
        self.clf = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=31, learning_rate=0.05, n_estimators=1000,
                                      objective="binary", random_state=seed, verbose=-1)
        self.clf.fit(q[b], y_view[b], eval_set=[(q[split.val], y_view[split.val])], eval_metric="binary_logloss",
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        info["size"] = info.pop("params")
        self.d = inp.x.shape[1]
        return info

    def save(self, path):
        import torch
        torch.save({"fusion": self.fusion, "tap": self.tap, "d": self.d, "gcn": self.gcn.state_dict()}, path + ".gcn.pt")
        with open(path + ".lgbm.txt", "w") as f:
            f.write(self.clf.booster_.model_to_string(num_iteration=self.clf.best_iteration_))
        return True

    def load(self, path, device="cpu"):
        import lightgbm as lgb
        import torch
        from . import nn as N
        ck = torch.load(path + ".gcn.pt", map_location=device)
        assert (ck["fusion"], ck["tap"]) == (self.fusion, self.tap), "checkpoint is for another variant"
        self.device, self.d = device, ck["d"]
        self.gcn = N.GCN(ck["d"]).to(device); self.gcn.load_state_dict(ck["gcn"]); self.gcn.eval()
        self.booster, self.clf = lgb.Booster(model_file=path + ".lgbm.txt"), None
        return self

    def _q(self, inp):
        from . import nn as N
        out = N.predict(self.gcn, N.Bundle(inp, self.device))
        r = out["hidden"] if self.tap == "hidden" else out["prob2"]
        return r if self.fusion == "P3" else np.hstack([inp.m, inp.t, r])

    def predict(self, inp):
        q = self._q(inp)
        if getattr(self, "booster", None) is not None:
            return {"p": self.booster.predict(q)}
        return {"p": self.clf.predict_proba(q)[:, 1]}


# ---------------------------------------------------------------- proposed

RGF_VARIANTS = {
    "RGF (proposed)": dict(),
    "RGF w/o gate (mean of experts)": dict(gate=False),
    "RGF w/o modality dropout": dict(p_drop=0.0),
    "RGF w/o auxiliary losses": dict(aux=0.0),
    "RGF concat fusion": dict(fusion="concat"),
    "RGF w/o graph expert": dict(experts=("m", "t")),
    "RGF w/o text expert": dict(experts=("m", "g")),
    "RGF w/o metadata expert": dict(experts=("t", "g")),
}


class RGFMethod(Method):
    def __init__(self, name: str = "RGF (proposed)"):
        self.name = name
        self.kw = RGF_VARIANTS[name]
        self.group = "proposed" if name == "RGF (proposed)" else "ablation"

    def fit(self, inp, y_view, split, seed, device="cpu"):
        import torch
        from . import nn as N
        torch.manual_seed(seed)
        self.device = device
        self.dims = (inp.m.shape[1], inp.t.shape[1])
        self.model = N.RGF(*self.dims, **self.kw).to(device)
        B = N.Bundle(inp, device)
        info = N.fit(self.model, B, y_view, split.train, split.val, seed, lr=0.005, epochs=300, patience=40)
        info["temperature"] = N.fit_temperature(self.model, B, y_view, split.val)
        info["size"] = info.pop("params")
        return info

    def predict(self, inp):
        import torch
        from . import nn as N
        B = N.Bundle(inp, self.device)
        out = N.predict(self.model, B)
        T = float(self.model.temperature.item())
        self.model.temperature.fill_(1.0)
        out["p_uncalibrated"] = N.predict(self.model, B)["p"]
        self.model.temperature.fill_(T)
        return out

    def save(self, path):
        import torch
        torch.save({"name": self.name, "dims": self.dims, "state": self.model.state_dict()}, path + ".pt")
        return True

    def load(self, path, device="cpu"):
        import torch
        from . import nn as N
        ck = torch.load(path + ".pt", map_location=device)
        assert ck["name"] == self.name, "checkpoint is for another variant"
        self.device, self.dims = device, tuple(ck["dims"])
        self.model = N.RGF(*self.dims, **self.kw).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval()   # includes the fitted temperature
        return self


# ---------------------------------------------------------------- registry

TABULAR = [("lgbm", "meta"), ("lgbm", "text"), ("lgbm", "meta+text"), ("xgb", "meta+text"), ("rf", "meta+text")]
HYBRIDS = [("P3", "hidden"), ("P3", "prob"), ("P1", "hidden"), ("P1", "prob")]


def make_all(dataset: str, setting: str) -> list:
    ms = [Tabular(m, f) for m, f in TABULAR]
    ms += [GNN(k) for k in ("gcn", "sage", "gat")]
    if setting == "transductive":
        ms += [GNN("gcn", "meta"), GNN("gcn", "text")]          # modality ablations: Metadata + GCN, Text + GCN
    if dataset in C.FOLLOW_GRAPH and setting == "transductive":
        ms.append(GNN("rgcn"))
    ms += [Hybrid(f, t) for f, t in HYBRIDS]
    ms += [RGFMethod(n) for n in RGF_VARIANTS]
    if setting == "inductive":
        keep = {"LightGBM (meta+text)", "GCN", "GraphSAGE", "GAT", "RGF (proposed)"} | {h.name for h in ms if h.group == "hybrid"}
        ms = [m for m in ms if m.name in keep]
    return ms


def by_name(dataset: str, setting: str, name: str) -> Method:
    """A fresh, unfitted method object by its reported name (used to reload checkpoints)."""
    for m in make_all(dataset, setting):
        if m.name == name:
            return m
    raise KeyError(name)


ROBUST_SET = {"LightGBM (meta+text)", "GCN", "GraphSAGE", "GAT", "BotRGCN-style R-GCN", "RGF (proposed)",
              "RGF w/o gate (mean of experts)", "RGF w/o modality dropout", "RGF concat fusion"} | \
             {f"Reconstructed hybrid ({f}, {t})" for f, t in HYBRIDS}
CROSS_SET = {"LightGBM (meta+text)", "GCN", "GraphSAGE", "RGF (proposed)"} | \
            {f"Reconstructed hybrid ({f}, {t})" for f, t in HYBRIDS}
