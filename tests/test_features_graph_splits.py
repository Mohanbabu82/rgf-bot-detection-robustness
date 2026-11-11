import math

import numpy as np
import pandas as pd
import pytest

from botrecon import graph as G
from botrecon.features import MetadataTransformer, raw_features, username_entropy
from botrecon.splits import blend_halves, kfold, stratified_70_10_20


def test_username_entropy():
    assert username_entropy("aaaa") == 0
    assert username_entropy("ab") == pytest.approx(1.0)
    assert username_entropy("abcd") == pytest.approx(2.0)
    assert math.isnan(username_entropy(""))


def test_raw_feature_formulas():
    meta = pd.DataFrame({
        "created_at": pd.to_datetime(["2020-01-11", "2020-01-01"], utc=True),
        "followers_count": [10.0, 0.0], "friends_count": [4.0, 0.0],
        "statuses_count": [100.0, 5.0], "favourites_count": [3.0, 0.0],
        "screen_name": ["ab", "aaaa"],
    })
    f = raw_features(meta, "extended6")
    assert f.loc[0, "age_days"] == 0 and f.loc[1, "age_days"] == 10
    assert f.loc[0, "follower_ratio"] == 10 / 5
    assert f.loc[1, "tweet_freq"] == 5 / 11
    assert f.loc[0, "reputation"] == 10 / 15
    assert f.loc[0, "engagement"] == 3 / 101


def test_scaler_fitted_only_on_fit_rows():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(100, 3))
    phi[80:] = 1e6                                   # wild "test" rows
    phi[5, 1] = np.nan
    fit = np.arange(80)
    t = MetadataTransformer().fit(phi, fit)
    z = t.transform(phi)
    assert np.allclose(z[fit].mean(0), 0, atol=1e-9)
    assert np.allclose(z[fit].std(0), 1, atol=1e-9)
    t2 = MetadataTransformer().fit(np.vstack([phi[:80], -phi[80:]]), fit)   # test rows changed
    assert np.allclose(t2.transform(phi)[fit], z[fit])                      # train features unchanged


def test_splits_disjoint_stratified_deterministic():
    y = np.array([0] * 300 + [1] * 700)
    s = stratified_70_10_20(y, 3)
    assert len(s.train) == 700 and len(s.val) == 100 and len(s.test) == 200
    assert abs(y[s.test].mean() - 0.7) < 0.01
    s2 = stratified_70_10_20(y, 3)
    assert np.array_equal(s.test, s2.test)
    assert not np.array_equal(s.test, stratified_70_10_20(y, 4).test)
    a, b = blend_halves(s.train, y, 0)
    assert len(set(a) & set(b)) == 0 and set(a) | set(b) == set(s.train)
    folds = list(kfold(s.train, y, 0))
    assert sorted(np.concatenate(folds)) == sorted(s.train)


def test_knn_transductive():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(60, 5))
    gp = G.transductive_knn(z, 10)
    s, t = gp.train_edges
    assert (s != t).all()
    pairs = set(zip(s, t))
    assert all((b, a) in pairs for a, b in pairs)                 # symmetric
    assert np.bincount(t, minlength=60).min() >= 10              # >= k neighbours each


def test_knn_inductive_has_no_test_leakage_paths():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(80, 4))
    seen, unseen = np.arange(60), np.arange(60, 80)
    gp = G.inductive_knn(z, 10, seen, unseen)
    s, t = gp.train_edges
    assert np.isin(s, seen).all() and np.isin(t, seen).all()    # training graph excludes test nodes
    s, t = gp.infer_edges
    assert not (np.isin(s, unseen) & np.isin(t, unseen)).any()  # no test-test edges
    assert not (np.isin(s, unseen) & np.isin(t, seen)).any()    # seen nodes never hear from test
    assert np.bincount(t, minlength=80)[unseen].min() == 10


def test_follow_inductive():
    e = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    gp = G.inductive_follow(e, 5, np.array([0, 1, 2]), np.array([3, 4]))
    s, t = gp.train_edges
    assert np.isin(s, [0, 1, 2]).all() and np.isin(t, [0, 1, 2]).all()
    s, t = gp.infer_edges
    assert not (np.isin(s, [3, 4]) & np.isin(t, [0, 1, 2])).any()
    assert not (np.isin(s, [3, 4]) & np.isin(t, [3, 4])).any()


def test_normalised_adjacency_matches_dense_formula():
    e = G.symmetrise(np.array([[0, 1, 2], [1, 2, 3]]))
    A = np.zeros((4, 4)); A[e[1], e[0]] = 1
    At = A + np.eye(4)
    d = At.sum(1)
    dense = np.diag(d ** -0.5) @ At @ np.diag(d ** -0.5)
    assert np.allclose(G.normalised_adjacency(e, 4).toarray(), dense)


def test_homophily_reads_only_given_nodes():
    e = G.symmetrise(np.array([[0, 1, 2], [1, 2, 3]]))
    y = np.array([0, 0, 1, -1])
    assert G.edge_homophily(e, y, np.array([0, 1, 2])) == pytest.approx(0.5)


def test_infinite_values_are_imputed_from_training_rows():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 3)); phi[3, 0] = np.inf; phi[45, 1] = -np.inf
    t = MetadataTransformer().fit(phi, np.arange(40))
    z = t.transform(phi)
    assert np.isfinite(z).all()
    finite_train = phi[:40, 0][np.isfinite(phi[:40, 0])]
    assert z[3, 0] == pytest.approx((np.sign(np.median(finite_train)) * np.log1p(abs(np.median(finite_train))) - t.mean_[0]) / t.std_[0])


def test_raw_features_never_infinite():
    meta = pd.DataFrame({"created_at": pd.to_datetime(["2020-01-01", None], utc=True),
                         "followers_count": [np.inf, 5.0], "friends_count": [1.0, np.nan],
                         "statuses_count": [1.0, 2.0], "favourites_count": [0.0, 1.0], "screen_name": ["a", ""]})
    f = raw_features(meta, "extended6").to_numpy(float)
    assert not np.isinf(f).any()


def test_local_model_copy_is_found_and_provenance_recorded(tmp_path, monkeypatch):
    """Notebooks without internet load the text model from a Kaggle input folder."""
    import os
    from botrecon import text as T
    d = tmp_path / "input" / "sentence-transformers-222" / "all-MiniLM-L6-v2"
    (d / ".git" / "refs" / "heads").mkdir(parents=True)
    (d / "modules.json").write_text("[]"); (d / "config.json").write_text("{}")
    (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (d / ".git" / "refs" / "heads" / "main").write_text("abc123\n")
    roots = (str(tmp_path / "input"),)
    assert T.local_model_path("sentence-transformers/all-MiniLM-L6-v2", roots, "modules.json") == str(d)
    assert T.local_model_path("bert-base-uncased", roots) is None
    assert T.model_revision(str(d)) == "abc123"
    monkeypatch.setattr(T, "LOCAL_MODEL_ROOTS", roots)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    path, src = T._resolve("sentence-transformers/all-MiniLM-L6-v2", "modules.json")
    assert path == str(d) and "git abc123" in src and os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False); monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert T._resolve("bert-base-uncased", "config.json") == ("bert-base-uncased", "huggingface.co")
