import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic import make_cresci, make_cresci2015, make_twibot  # noqa: E402


@pytest.fixture(scope="session")
def cresci_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("cresci"))
    info = make_cresci(root)
    return root, info


@pytest.fixture(scope="session")
def cresci2015_root(tmp_path_factory, cresci_root):
    """Shares 10 IDs with the synthetic Cresci-2017 fake_followers, as the real releases do."""
    import pandas as pd
    ff = pd.read_csv(os.path.join(cresci_root[0], "fake_followers.csv", "users.csv"), dtype=str)["id"].tolist()
    root = str(tmp_path_factory.mktemp("cresci2015"))
    info = make_cresci2015(root, shared_bot_ids=ff[:10])
    return root, info


@pytest.fixture(scope="session")
def twibot_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("twibot"))
    info = make_twibot(root)
    return root, info


class StubGCN:
    """Stand-in for train_gcn, for protocol tests only. Propagates features with
    the given adjacency and fits a logistic regression on fit_idx labels. Records
    every call so tests can check which labels were used."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, a_train, a_infer, y_view, fit_idx, val_idx, cfg, seed):
        from sklearn.linear_model import LogisticRegression
        from botrecon.gcn import GCNResult
        assert (y_view[fit_idx] >= 0).all()
        self.calls.append({"fit_idx": fit_idx.copy(), "y_fit": y_view[fit_idx].copy(),
                           "y_view": y_view.copy(), "a_train": a_train, "a_infer": a_infer})
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(x.shape[1], cfg.gcn_hidden)) / np.sqrt(x.shape[1])
        h_tr = np.maximum(a_train @ (x @ W), 0)
        h = np.maximum(a_infer @ (x @ W), 0)
        lr = LogisticRegression(max_iter=500).fit(h_tr[fit_idx], y_view[fit_idx])
        prob = lr.predict_proba(h)
        return GCNResult(h.astype(np.float32), prob.astype(np.float32), cfg.gcn_epochs, 0.0)


class StubClassifier:
    """Stand-in for LGBMStage (sklearn HistGradientBoosting). Records fit inputs."""
    log = []

    def __init__(self, cfg, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(random_state=seed, max_iter=50)

    def fit(self, X, y, X_es, y_es):
        assert (np.asarray(y) >= 0).all() and (np.asarray(y_es) >= 0).all()
        StubClassifier.log.append({"n_fit": len(y), "n_es": len(y_es), "d": X.shape[1]})
        self.m.fit(X, y)
        self.best_iteration = 50
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


@pytest.fixture
def stub_gcn():
    return StubGCN()


@pytest.fixture
def stub_clf():
    StubClassifier.log = []
    return StubClassifier
