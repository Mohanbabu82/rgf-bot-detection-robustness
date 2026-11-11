"""Build results tables R1-R6 from results/runs.csv.

Reproduced numbers come only from runs.csv. Manuscript and literature numbers
come only from botrecon.provenance and are shown in their own labelled rows.
SMOKE-TEST rows are ignored.

  python -m botrecon.report --runs results/runs.csv --out results/tables.md
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import provenance as P
from .stats import holm, mean_ci, paired_t

CFG_KEYS = ["dataset", "family", "fusion", "tap", "text", "cresci_label_rule",
            "cresci_knn_features", "twibot_split", "setting", "es_on", "stacking", "scaler_fit"]


def _fmt(ci: dict, scale: float = 100) -> str:
    if ci["n"] < 2:
        return f"{ci['mean'] * scale:.2f} (n=1)"
    return f"{ci['mean'] * scale:.2f} ± {ci['sd'] * scale:.2f} [{ci['ci_low'] * scale:.2f}, {ci['ci_high'] * scale:.2f}]"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(CFG_KEYS, dropna=False):
        r = dict(zip(CFG_KEYS, key))
        r["provenance"] = g["provenance"].iloc[0]
        r["seeds"] = len(g)
        for m in ("test_accuracy", "test_f1_bot", "test_mcc", "test_pr_auc", "test_f1_weighted", "val_f1_bot"):
            ci = mean_ci(g[m].to_numpy())
            r[m + "_mean"], r[m + "_sd"], r[m + "_fmt"] = ci["mean"], ci["sd"], _fmt(ci)
        rows.append(r)
    return pd.DataFrame(rows)


def primary(summary: pd.DataFrame, dataset: str):
    c = summary[(summary.dataset == dataset) & (summary.provenance == P.REPRODUCED_CORRECTED)
                & (summary.setting == "transductive")]
    if c.empty:
        return None
    return c.sort_values("val_f1_bot_mean", ascending=False).iloc[0]   # selected on VALIDATION only


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no runs yet_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def build(runs_csv: str) -> str:
    df = pd.read_csv(runs_csv)
    df = df[df.provenance != P.SMOKE_TEST]
    out = ["# Results tables (generated; do not edit reproduced cells by hand)\n"]
    if df.empty:
        return out[0] + "\n_No reproducible runs yet (smoke-test rows are excluded)._\n"
    s = summarise(df)
    for ds in ("twibot20", "cresci2017"):
        if ds not in set(s.dataset):
            continue
        out.append(f"\n## {ds}\n")
        # R1
        rows = []
        mr = P.MANUSCRIPT_RESULTS[ds]
        rows.append({"Model": "Proposed (original)", "Provenance": P.MANUSCRIPT_REPORTED,
                     "Accuracy": mr["accuracy"], "F1": f"{mr['f1']} ({mr['averaging']})",
                     "MCC": "not reported", "Source": mr["source"]})
        pr = primary(s, ds)
        if pr is not None:
            cfg = f"{pr.fusion}/{pr.tap}/text {pr.text}"
            rows.append({"Model": f"Reconstruction, primary ({cfg})", "Provenance": P.REPRODUCED_CORRECTED,
                         "Accuracy": pr.test_accuracy_fmt, "F1": pr.test_f1_bot_fmt + " (bot class)",
                         "MCC": pr.test_mcc_fmt, "Source": "This work"})
        if ds == "twibot20":
            for lit in P.LITERATURE_TWIBOT20:
                rows.append({"Model": lit["model"], "Provenance": P.LITERATURE, "Accuracy": lit["accuracy"],
                             "F1": lit["f1"], "MCC": lit["mcc"] if lit["mcc"] is not None else "—",
                             "Source": f"[{lit['source']}]({lit['url']})"})
        out.append("\n### R1 Main comparison (%, mean ± SD [95% CI] over seeds)\n")
        out.append(md_table(pd.DataFrame(rows)))

        # R3 literal spread
        lit = s[(s.dataset == ds) & (s.provenance == P.REPRODUCED_LITERAL)]
        out.append("\n### R3 Literal-family spread (sensitivity analysis only)\n")
        if lit.empty:
            out.append("_no literal runs yet_\n")
        else:
            acc = lit.test_accuracy_mean * 100
            closest = lit.iloc[(acc - mr["accuracy"]).abs().argsort().iloc[0]]
            out.append(md_table(pd.DataFrame([{
                "Reported accuracy (MANUSCRIPT-REPORTED)": mr["accuracy"], "Literal configs": len(lit),
                "Min": f"{acc.min():.2f}", "Median": f"{acc.median():.2f}", "Max": f"{acc.max():.2f}",
                "Closest config": f"{closest.fusion}/{closest.tap}/text {closest.text}/"
                                  f"{closest.cresci_label_rule if ds == 'cresci2017' else closest.twibot_split}",
                "Its accuracy": closest.test_accuracy_fmt, "Its MCC": closest.test_mcc_fmt}])))
            out.append("A match is *consistent with*, not confirmation of, the original pipeline.\n")

        # R4 corrected ablation + paired tests vs primary
        cor = s[(s.dataset == ds) & (s.provenance == P.REPRODUCED_CORRECTED) & (s.setting == "transductive")]
        out.append("\n### R4 Corrected-family configurations\n")
        if not cor.empty:
            t = cor.sort_values("val_f1_bot_mean", ascending=False)[
                ["fusion", "tap", "text", "seeds", "val_f1_bot_fmt", "test_f1_bot_fmt",
                 "test_mcc_fmt", "test_pr_auc_fmt"]]
            out.append(md_table(t))
            if pr is not None:
                base = df[(df.dataset == ds) & (df.provenance == P.REPRODUCED_CORRECTED)
                          & (df.setting == "transductive") & (df.fusion == pr.fusion) & (df.tap == pr.tap)
                          & (df.text == pr.text)].set_index("seed")["test_f1_bot"]
                pvals, stats_rows = {}, []
                for _, o in cor.iterrows():
                    if (o.fusion, o.tap, o.text) == (pr.fusion, pr.tap, pr.text):
                        continue
                    other = df[(df.dataset == ds) & (df.provenance == P.REPRODUCED_CORRECTED)
                               & (df.setting == "transductive") & (df.fusion == o.fusion) & (df.tap == o.tap)
                               & (df.text == o.text)].set_index("seed")["test_f1_bot"]
                    common = base.index.intersection(other.index)
                    if len(common) >= 3:
                        pt = paired_t(base[common], other[common])
                        name = f"{o.fusion}/{o.tap}/text {o.text}"
                        pvals[name] = pt["p"]
                        stats_rows.append({"vs": name, "mean ΔF1 (pp)": round(pt["mean_diff"] * 100, 2),
                                           "Cohen d_z": round(pt["cohen_dz"], 2), "p": round(pt["p"], 4)})
                if stats_rows:
                    h = holm(pvals)
                    for r in stats_rows:
                        r["p (Holm)"] = round(h[r["vs"]]["p_holm"], 4)
                    out.append("\nPaired t-tests, primary vs each other config (per-seed bot-class F1):\n")
                    out.append(md_table(pd.DataFrame(stats_rows)))

        # R5 inductive vs transductive
        ind = s[(s.dataset == ds) & (s.provenance == P.REPRODUCED_CORRECTED) & (s.setting == "inductive")]
        out.append("\n### R5 Inductive vs transductive\n")
        same = cor.merge(ind[["fusion", "tap", "text"]], on=["fusion", "tap", "text"]) if not ind.empty else cor.iloc[0:0]
        out.append(md_table(pd.concat([same, ind])[["fusion", "tap", "text", "setting", "seeds",
                                                     "test_f1_bot_fmt", "test_mcc_fmt"]]
                            if not ind.empty else pd.DataFrame()))

        # R6 diagnostic
        dia = s[(s.dataset == ds) & (s.provenance == P.DIAGNOSTIC_INVALID)]
        out.append("\n### R6 Diagnostic: effect of early stopping on the test set (INVALID, not a result)\n")
        out.append(md_table(dia[["fusion", "tap", "text", "seeds", "test_accuracy_fmt", "test_f1_bot_fmt"]]
                            if not dia.empty else pd.DataFrame()))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out", default="results/tables.md")
    a = ap.parse_args(argv)
    md = build(a.runs)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
