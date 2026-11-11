import numpy as np
import pytest
import pandas as pd

from botrecon.data import load_cresci2015, load_cresci2017, load_twibot20
from botrecon.data.cresci import EXPLICIT_LABELS, literal_label


def test_explicit_map_counts_fake_followers_as_bots():
    assert EXPLICIT_LABELS["fake_followers"] == 1
    assert EXPLICIT_LABELS["genuine_accounts"] == 0
    assert sum(EXPLICIT_LABELS.values()) == 8


def test_literal_rule_reproduces_algorithm1_flaw():
    assert literal_label("fake_followers.csv") == 0          # the suspected error
    assert literal_label("social_spambots_1.csv") == 1
    assert literal_label("traditional_spambots_4.csv") == 1
    assert literal_label("genuine_accounts.csv") == 0


def test_cresci_dedup_and_labels(cresci_root):
    root, info = cresci_root
    d = load_cresci2017(root, "explicit")
    # 9 groups x 30 + 1 same-label duplicate + 1 conflicting duplicate
    assert d.log["rows_before_dedup"] == 272
    assert d.log["conflicting_ids_dropped"] == 1
    assert info["conflict_id"] not in set(d.ids)
    assert d.n == 269 and len(set(d.ids)) == d.n
    assert (d.y[d.group == "fake_followers"] == 1).all()
    assert (d.y[d.group == "genuine_accounts"] == 0).all()
    lit = load_cresci2017(root, "literal_bot_substring")
    assert (lit.y[lit.group == "fake_followers"] == 0).all()
    assert lit.y.sum() == d.y.sum() - (d.group == "fake_followers").sum()


def test_cresci_metadata_parsed(cresci_root):
    d = load_cresci2017(cresci_root[0])
    assert d.meta["created_at"].notna().all()
    assert str(d.meta["created_at"].dt.tz) == "UTC"
    assert d.meta["followers_count"].notna().all()


def test_cresci_tweets_recent_first_and_capped(cresci_root):
    d = load_cresci2017(cresci_root[0], load_tweets=True, max_tweets=20)
    lens = np.array([len(t) for t in d.tweets])
    assert lens.max() == 20
    # traditional spambots 2-4 ship without tweets
    for g in ("traditional_spambots_2", "traditional_spambots_3", "traditional_spambots_4"):
        assert (lens[d.group == g] == 0).all()
    # most recent first: synthetic tweet j has day (j % 28) + 1, so 'tweet 24' (day 25) is newest
    i = int(np.where(d.group == "genuine_accounts")[0][0])
    assert d.tweets[i][0].startswith("tweet 24")


def test_twibot_loader(twibot_root):
    root, info = twibot_root
    d = load_twibot20(root, load_tweets=True)
    assert d.n == 215
    assert list(d.ids) == info["ids"]
    assert np.array_equal(d.y, info["labels"])
    assert d.log["split_sizes"] == {"train": 120, "val": 40, "test": 55}
    assert d.meta["followers_count"].notna().all()      # padded strings parsed
    assert d.edges.shape[0] == 2 and d.edges.max() < d.n
    assert (d.edges[0] != d.edges[1]).all()
    # 'following' of record i gives edge i -> j; unknown id 99999999 is dropped
    r0 = info["records"][0]
    if r0["neighbor"]:
        for j in r0["neighbor"]["following"]:
            if j in info["ids"] and j != r0["ID"]:
                jj = info["ids"].index(j)
                assert ((d.edges[0] == 0) & (d.edges[1] == jj)).any()
    assert all(len(t) <= 20 for t in d.tweets)


def test_cresci2015_loader(cresci2015_root, cresci_root):
    root, info = cresci2015_root
    d = load_cresci2015(root, load_tweets=True)
    assert d.name == "cresci2015" and d.n == 5 * 30 and d.official_split is None
    humans = set(info["E13"]) | set(info["TFP"])
    assert all((y == 0) == (u in humans) for u, y in zip(d.ids, d.y))
    assert set(d.group) == {"E13", "TFP", "FSF", "INT", "TWT"}          # TWT folder without ".csv" is found
    assert d.edges.shape[0] == 2 and d.edges.shape[1] > 0
    assert d.edges.max() < d.n and not (d.edges[0] == d.edges[1]).any()  # unlabelled neighbours and self-loops dropped
    assert d.log["directed_edges_among_labelled"] == d.edges.shape[1] < d.log["relation_rows_read"]
    assert all(len(t) == 5 for t in d.tweets)
    same = (d.y[d.edges[0]] == d.y[d.edges[1]]).mean()
    assert same > 0.7                                                      # the synthetic graph is homophilous
    c17 = load_cresci2017(cresci_root[0])
    assert len(set(d.ids) & set(c17.ids)) == 10                            # shared fake followers kept visible


def test_cresci2015_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cresci2015(str(tmp_path))
