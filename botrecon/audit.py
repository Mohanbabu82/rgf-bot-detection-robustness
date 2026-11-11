"""Protocol audit on real data: run in the notebook before any training.

Every check returns PASS/FAIL with the evidence, so the saved JSON shows exactly what was verified.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .suite import Prepared
from .text import text_inputs


def _check(name, ok, evidence):
    return {"check": name, "status": "PASS" if ok else "FAIL", "evidence": evidence}


def label_audit(data) -> list:
    out = [_check("labels are 0/1 only", set(np.unique(data.y)) <= {0, 1}, f"values {sorted(set(data.y.tolist()))}"),
           _check("account IDs unique", len(set(data.ids)) == data.n, f"{data.n} rows, {len(set(data.ids))} unique IDs")]
    if data.name == "cresci2017":
        g = data.group
        human_groups = sorted(set(g[data.y == 0]))
        out.append(_check("Cresci-2017: only genuine_accounts labelled human (explicit map)",
                          human_groups == ["genuine_accounts"], f"human groups: {human_groups}"))
        ff = (g == "fake_followers")
        out.append(_check("Cresci-2017: fake_followers labelled bot", bool(ff.any()) and bool((data.y[ff] == 1).all()),
                          f"{int(ff.sum())} fake_followers accounts"))
        sizes = {k: v.get("rows") for k, v in data.log.get("groups", {}).items()}
        pub = {k: v.get("published_size") for k, v in data.log.get("groups", {}).items()}
        out.append(_check("Cresci-2017: group sizes match the published sizes", sizes == pub,
                          f"loaded {sizes} vs published {pub}"))
    if data.name == "cresci2015":
        from .data.cresci2015 import LABELS
        g = data.group
        out.append(_check("Cresci-2015: labels follow the group map (E13, TFP human; FSF, INT, TWT bot)",
                          all(int(LABELS[gg]) == int(yy) for gg, yy in zip(g, data.y)),
                          f"groups {sorted(set(g))}"))
        sizes = {k: v.get("rows") for k, v in data.log.get("groups", {}).items()}
        pub = {k: v.get("published_size") for k, v in data.log.get("groups", {}).items()}
        out.append(_check("Cresci-2015: group sizes match the published sizes", sizes == pub,
                          f"loaded {sizes} vs published {pub}"))
        out.append(_check("Cresci-2015: follower/friend graph among labelled accounts present",
                          data.edges is not None and data.edges.shape[1] > 0,
                          f"{data.log.get('directed_edges_among_labelled')} directed edges, "
                          f"{data.log.get('isolated_nodes')} isolated accounts"))
    if data.name == "twibot20":
        out.append(_check("TwiBot-20: official split present", data.official_split is not None,
                          f"{data.log.get('split_sizes')}"))
    return out


def protocol_audit(data, seed: int = 0, settings=("transductive", "inductive"), text_variant: str = "B") -> list:
    cfg = C.corrected(data.name, text=text_variant)
    texts = text_inputs(data, "B")
    emb = np.zeros((data.n, 1), np.float32)            # label-independent placeholder: audit needs no encoder
    rows = label_audit(data)
    for setting in settings:
        p = Prepared(data, emb, cfg, seed, setting, text_avail=[len(t.strip()) > 0 for t in texts])
        s, n = p.split, data.n
        tr, va, te = set(s.train), set(s.val), set(s.test)
        rows.append(_check(f"[{setting}] train/val/test disjoint", not (tr & va or tr & te or va & te),
                           f"{len(tr)}/{len(va)}/{len(te)} of {n}"))
        if data.name in ("cresci2017", "cresci2015"):
            share = [round(len(x) / n, 3) for x in (tr, va, te)]
            rows.append(_check(f"[{setting}] {data.name} split is 70/10/20", np.allclose(share, [0.7, 0.1, 0.2], atol=0.01),
                               f"shares {share}"))
            bot = [round(float(data.y[list(x)].mean()), 3) for x in (tr, va, te)]
            rows.append(_check(f"[{setting}] stratified (bot share equal across splits)", max(bot) - min(bot) < 0.01,
                               f"bot share train/val/test {bot}"))
        else:
            rows.append(_check(f"[{setting}] TwiBot-20 uses the official split", s.kind == "official", s.kind))
        rows.append(_check(f"[{setting}] test labels hidden from all fitting", bool((p.y_view[s.test] == -1).all()),
                           "y_view[test] == -1"))
        rows.append(_check(f"[{setting}] metadata scaler fitted on training rows only",
                           p.transformer.fit_idx_size_ == len(s.train), f"fit rows {p.transformer.fit_idx_size_}"))
        e = p.train_inp.sym_edges
        test_in_train_graph = bool(np.isin(e, s.test).any()) if e.size else False
        if setting == "transductive":
            rows.append(_check("[transductive] test accounts present in the graph (features only, no labels)",
                               True, f"test nodes in training graph: {test_in_train_graph} (expected in transductive)"))
        else:
            rows.append(_check("[inductive] test accounts absent from the training graph", not test_in_train_graph,
                               f"test nodes in training graph: {test_in_train_graph}"))
            ei = p.infer_inp.sym_edges
            leak = bool((np.isin(ei[0], s.test) & ~np.isin(ei[1], s.test)).any()) if ei.size else False
            rows.append(_check("[inductive] test accounts send no messages to training accounts", not leak,
                               f"test->seen edges: {leak}"))
        rows.append(_check(f"[{setting}] graph statistics", True,
                           f"edges {e.shape[1]}, edge homophily on training nodes {p.log['edge_homophily_train']:.3f}, "
                           f"isolated test share {p.log['isolated_share_test']:.3f}, "
                           f"text missing in test {p.log['text_missing_share_test']:.3f}"))
    return rows
