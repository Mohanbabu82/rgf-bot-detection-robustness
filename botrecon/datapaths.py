"""Locate (and if needed extract) the datasets. Used by the Kaggle notebook; no manual path edits.

Cresci-2017 root  = the folder containing genuine_accounts.csv/ (a FOLDER holding users.csv).
Cresci-2015 root  = the folder containing E13.csv/ (or E13/) holding users.csv, plus TFP, FSF, INT, TWT.
TwiBot-20         = library support only; not used by the plan or the notebook.
"""
from __future__ import annotations

import glob
import os
import tarfile
import zipfile

ARCHIVES = ("*.zip", "*.tar.gz", "*.tgz", "*.tar")


def _extract_one(path: str, dest: str) -> None:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    else:
        with tarfile.open(path) as t:
            members = [m for m in t.getmembers()
                       if not (os.path.isabs(m.name) or ".." in m.name.split("/")) and (m.isfile() or m.isdir())]
            try:
                t.extractall(dest, members=members, filter="data")
            except TypeError:                                   # Python without extraction filters
                t.extractall(dest, members=members)


def extract_all(path: str, dest: str) -> None:
    """Extract an archive into dest, then any archives found inside, until none are left (each once)."""
    _extract_one(path, dest)
    done = set()
    while True:
        todo = [p for pat in ARCHIVES for p in glob.glob(os.path.join(dest, "**", pat), recursive=True)
                if p not in done]
        if not todo:
            return
        for ap in todo:
            _extract_one(ap, os.path.dirname(ap))
            done.add(ap)


CRESCI2017_GROUPS = ("genuine_accounts", "social_spambots_1", "social_spambots_2", "social_spambots_3",
                     "traditional_spambots_1", "traditional_spambots_2", "traditional_spambots_3",
                     "traditional_spambots_4", "fake_followers")
CRESCI2015_GROUPS = ("E13", "TFP", "FSF", "INT", "TWT")


def _candidates(base: str, anchor: str, groups) -> list:
    """(number of groups found, root) for every folder that could be the dataset root. Handles '<g>.csv/',
    '<g>/' and the doubly nested '<g>.csv/<g>.csv/' layout Kaggle produces from archives inside archives."""
    from .data.common import resolve_group_dir
    roots = set()
    for name in (anchor + ".csv", anchor):
        for p in glob.glob(os.path.join(base, "**", name), recursive=True):
            if not os.path.isdir(p):
                continue
            parent = os.path.dirname(p)
            if os.path.basename(parent) in (anchor + ".csv", anchor):      # inner level of a nested pair
                parent = os.path.dirname(parent)
            roots.add(parent)
    out = []
    for r in roots:
        n = sum(resolve_group_dir(r, g) is not None for g in groups)
        if n:
            out.append((n, r))
    return sorted(out, key=lambda t: (-t[0], len(t[1]), t[1]))


def _best(bases, anchor, groups):
    c = [t for b in bases for t in _candidates(b, anchor, groups)]
    c.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    return c[0] if c else (0, None)


def find_cresci(base: str) -> str | None:
    return _best([base], "genuine_accounts", CRESCI2017_GROUPS)[1]


def find_cresci2015(base: str) -> str | None:
    return _best([base], "E13", CRESCI2015_GROUPS)[1]


def find_twibot(base: str) -> str | None:
    hits = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(base, "**", "dev.json"), recursive=True)
                   if all(os.path.exists(os.path.join(os.path.dirname(p), f)) for f in ("train.json", "test.json"))})
    return hits[0] if hits else None


def locate(input_dir: str, work_dir: str, cresci: str | None = None, cresci2015: str | None = None) -> dict:
    """Explicit paths (from the config) win. Otherwise the copy with the MOST groups is chosen among the
    folders under input_dir and archives extracted into work_dir/datasets (input folders are read-only on
    Kaggle). Incomplete copies are reported in notes and in out["groups_found"]."""
    out = {"cresci2017": cresci, "cresci2015": cresci2015, "notes": [], "groups_found": {}}
    spec = {"cresci2017": ("genuine_accounts", CRESCI2017_GROUPS), "cresci2015": ("E13", CRESCI2015_GROUPS)}
    ex = os.path.join(work_dir, "datasets")
    best = {k: _best([input_dir, ex], *spec[k]) for k in spec}
    if any(out[k] is None and best[k][0] < len(spec[k][1]) for k in spec):
        archives = sorted({p for pat in ARCHIVES for p in glob.glob(os.path.join(input_dir, "**", pat), recursive=True)
                           if not os.path.basename(p).startswith("botrecon")})     # code / results archives
        stamp = os.path.join(ex, ".extracted")
        seen = set(open(stamp).read().split("\n")) if os.path.exists(stamp) else set()
        archives = [a for a in archives if a not in seen]
        for a in archives:
            extract_all(a, ex)
            with open(stamp, "a") as f:
                f.write(a + "\n")
        if archives:
            out["notes"].append(f"extracted {len(archives)} archive(s) into {ex}")
            best = {k: _best([input_dir, ex], *spec[k]) for k in spec}
    for k, (anchor, groups) in spec.items():
        if out[k] is None:
            n, out[k] = best[k]
            out["groups_found"][k] = n
            if out[k] and n < len(groups):
                out["notes"].append(f"{k}: INCOMPLETE copy, only {n} of {len(groups)} groups at {out[k]}")
        if out[k] and not os.path.isdir(out[k]):
            raise FileNotFoundError(f"{k} path from the config does not exist: {out[k]}")
    return out
