"""One resumable work plan for Kaggle CPU sessions.

    python -m botrecon.plan --cresci ROOT --cresci2015 ROOT --minutes 38

Runs the steps below in priority order, each with the time that is left, and skips
anything already finished (results are read from results/). Rerun it in every new
session after restoring results/ and cache/; when everything is done it writes the
paper tables, figures and paper_facts.json to results/paper/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from . import provenance as P


def closest_literal(runs_csv: str, dataset: str):
    """Literal-family config (text B) whose mean accuracy is closest to the manuscript's number."""
    try:
        df = pd.read_csv(runs_csv)
    except FileNotFoundError:
        return None
    d = df[(df.dataset == dataset) & (df.provenance == P.REPRODUCED_LITERAL)]
    if d.empty:
        return None
    keys = ["fusion", "tap", "text", "cresci_label_rule", "cresci_knn_features", "twibot_split"]
    acc = d.groupby(keys).test_accuracy.mean() * 100
    best = (acc - P.MANUSCRIPT_RESULTS[dataset]["accuracy"]).abs().idxmin()
    cfg = dict(zip(keys, best))
    cfg.pop("twibot_split")
    return ",".join(f"{k}={v}" for k, v in cfg.items())


class _Tee:
    """Copy everything printed to a log file as well as the notebook output."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8")

    def write(self, text):
        self.stream.write(text); self.file.write(text); self.file.flush()

    def flush(self):
        self.stream.flush(); self.file.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cresci", required=True, help="Cresci-2017 root")
    ap.add_argument("--cresci2015", default=None, help="Cresci-2015 root; omit to run Cresci-2017 only")
    ap.add_argument("--part", choices=["main", "extended", "all"], default="all",
                    help="main = baselines, reconstructed hybrid, proposed model and modality ablations "
                         "(transductive); extended = RGF ablations and inductive runs")
    ap.add_argument("--minutes", type=float, default=38.0, help="session budget; report reserve included")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--robust-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--checkpoint-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--text", choices=["A", "B"], default="B")
    ap.add_argument("--with-bert-tweets", action="store_true", help="also run text variant A (slow on CPU)")
    ap.add_argument("--with-reconstruction", action="store_true",
                    help="internal audit only: literal re-implementation of the earlier manuscript (Cresci-2017). "
                         "Not part of the new paper.")
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs("results/logs", exist_ok=True)
    log_path = time.strftime("results/logs/plan_%Y%m%d_%H%M%S.log")
    old_stdout, sys.stdout = sys.stdout, _Tee(sys.stdout, log_path)
    try:
        _run(a, t0)
    finally:
        sys.stdout = old_stdout
        print(f"log written to {log_path}")


def _run(a, t0):
    def left(reserve=2.0):
        return max(0.0, a.minutes - reserve - (time.time() - t0) / 60)

    from . import run as R
    from . import suite as S
    from .methods import make_all
    from .suite_report import build

    th = ["--threads", str(a.threads)] if a.threads else []
    seeds = [str(s) for s in a.seeds]
    common = ["--robust-seeds", *map(str, a.robust_seeds), "--checkpoint-seeds", *map(str, a.checkpoint_seeds),
              "--text", a.text]
    from .config import CROSS_PARTNER
    roots = {"cresci2017": a.cresci, "cresci2015": a.cresci2015}
    other = {ds: CROSS_PARTNER[ds] for ds in roots}
    available = [ds for ds in ("cresci2017", "cresci2015") if roots[ds]]
    if len(available) < 2:
        print(f"Running with {available} only; add Cresci-2015 later and rerun (finished runs are kept).")
    print(f"part = {a.part}; datasets = {available}; seeds = {a.seeds}; text variant = {a.text}")

    def suite_step(ds, setting, names):
        cross = ["--cross-root", roots[other[ds]]] if roots.get(other[ds]) else []
        return lambda: S.main(["--dataset", ds, "--data-root", roots[ds], "--settings", setting,
                               "--seeds", *seeds, "--max-minutes", f"{left():.2f}", "--methods", *names]
                              + cross + th + common)

    steps = []
    if a.part in ("main", "all"):
        for ds in available:
            names = [m.name for m in make_all(ds, "transductive") if m.group != "ablation"]
            steps.append((f"suite {ds} transductive: baselines, hybrid, proposed, modality ablations",
                          suite_step(ds, "transductive", names)))
    if a.part in ("extended", "all"):
        for ds in available:
            names = [m.name for m in make_all(ds, "transductive") if m.group == "ablation"]
            steps.append((f"suite {ds} transductive: RGF ablations", suite_step(ds, "transductive", names)))
        for ds in available:
            names = [m.name for m in make_all(ds, "inductive")]
            steps.append((f"suite {ds} inductive", suite_step(ds, "inductive", names)))
    if a.with_reconstruction:
        ds = "cresci2017"
        steps.append(("reconstruction literal screen cresci2017 (internal audit)", lambda: R.main(
            ["--dataset", ds, "--data-root", roots[ds], "--family", "literal", "--only", "text=B", "--seeds", "0",
             "--device", "cpu", "--max-minutes", f"{left():.2f}"] + th)))

        def diag():
            only = closest_literal("results/runs.csv", ds)
            if only:
                R.main(["--dataset", ds, "--data-root", roots[ds], "--family", "literal", "--diagnostic",
                        "--only", only, "--seeds", "0", "1", "2", "--device", "cpu",
                        "--max-minutes", f"{left():.2f}"] + th)
        steps.append(("diagnostic cresci2017 (internal audit)", diag))
    if a.with_bert_tweets:          # hybrid baseline with BERT-on-tweets text (results/runs.csv; slow on CPU)
        for ds in available:
            steps.append((f"hybrid baseline, BERT on tweets, {ds}", lambda ds=ds: R.main(
                ["--dataset", ds, "--data-root", roots[ds], "--family", "corrected", "--only", "text=A",
                 "--seeds", *seeds, "--device", "cpu", "--max-minutes", f"{left():.2f}"] + th)))
    for name, fn in steps:
        if left() <= 0.2:
            print(f"\n=== session budget used; stopped before: {name}. Restore results and rerun to continue. ===")
            break
        print(f"\n=== {name} ({left():.1f} min left) ===", flush=True)
        fn()
    print("\n=== building paper tables (results/paper) ===")
    build("results", "results/paper")
    print(f"session time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
