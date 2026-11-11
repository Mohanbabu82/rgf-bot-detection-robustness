"""Paper tables and figures from the suite results (and the reconstruction runs, if present).

  python -m botrecon.suite_report --results results --out results/paper

Writes:
  paper/tables.md          T1 datasets, T2 main comparison (+ manuscript and literature rows, kept apart),
                           T3 ablation, T4 robustness, T5 inductive, T6 calibration, T7 cross-dataset,
                           T8 explainability, T9 efficiency, T10 significance tests
  paper/*.csv              the same tables as CSV
  paper/fig_*.png          robustness curves, reliability diagram, ROC and PR curves, confusion matrix, gate weights
  paper/paper_facts.json   key numbers for drafting, each with its provenance label

Only rows from the result files are used for THIS-WORK / REPRODUCED numbers. SMOKE-TEST rows are dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import provenance as P
from .calib import reliability_curve
from .stats import holm, mean_ci, paired_t

PROPOSED = "RGF (proposed)"
# The paper is drafted from scratch on new experiments: the numbers printed in the earlier manuscript
# (implementation lost) are not shown in the tables. Set True only for the internal reconstruction audit.
INCLUDE_OLD_MANUSCRIPT = False
METRICS = [("test_accuracy", "Acc"), ("test_precision_bot", "Prec"), ("test_recall_bot", "Rec"),
           ("test_f1_bot", "F1"), ("test_mcc", "MCC"), ("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC")]
DS_NAME = {"cresci2017": "Cresci-2017", "cresci2015": "Cresci-2015", "twibot20": "TwiBot-20"}
DS_ORDER = ("cresci2017", "cresci2015", "twibot20")


def _read(path):
    from .suite import read_rows
    df = read_rows(path)
    if df.empty:
        return df
    return df[df.provenance != P.SMOKE_TEST] if "provenance" in df else df


def fmt(values, scale=100.0, digits=2):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0] * scale:.{digits}f}"
    return f"{v.mean() * scale:.{digits}f} ± {v.std(ddof=1) * scale:.{digits}f}"


def md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no results yet_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def best_hybrid(main: pd.DataFrame, ds: str):
    h = main[(main.dataset == ds) & (main.group == "hybrid") & (main.setting == "transductive")]
    if h.empty:
        return None
    return h.groupby("method")["val_f1_bot"].mean().idxmax()      # chosen on VALIDATION


def main_table(main, ds):
    d = main[(main.dataset == ds) & (main.setting == "transductive") & ~main.group.isin(["ablation", "modality"])]
    hyb = best_hybrid(main, ds)
    rows = []
    order = ["tabular", "gnn", "hybrid", "proposed"]
    for grp in order:
        for name, g in d[d.group == grp].groupby("method", sort=False):
            if grp == "hybrid" and name != hyb:
                continue
            r = {"Method": name + (" [best of 4 by validation]" if grp == "hybrid" else ""),
                 "Provenance": g.provenance.iloc[0], "Seeds": len(g)}
            for col, lab in METRICS:
                r[lab] = fmt(g[col])
            rows.append(r)
    mr = P.MANUSCRIPT_RESULTS.get(ds) if INCLUDE_OLD_MANUSCRIPT else None
    if mr: rows.append({"Method": "Original hybrid as reported (implementation lost)", "Provenance": P.MANUSCRIPT_REPORTED,
                 "Seeds": "1?", "Acc": mr["accuracy"], "Prec": f"{mr['precision']}*", "Rec": f"{mr['recall']}*",
                 "F1": f"{mr['f1']}*", "MCC": "—", "ROC-AUC": "—", "PR-AUC": "—"})
    if ds == "twibot20":
        for lit in P.LITERATURE_TWIBOT20:
            rows.append({"Method": f"{lit['model']} [{lit['source']}]", "Provenance": P.LITERATURE, "Seeds": "—",
                         "Acc": lit["accuracy"], "Prec": "—", "Rec": "—", "F1": lit["f1"],
                         "MCC": lit["mcc"] if lit["mcc"] is not None else "—", "ROC-AUC": "—", "PR-AUC": "—"})
    return pd.DataFrame(rows)


def significance(main, ds, metric="test_f1_bot"):
    d = main[(main.dataset == ds) & (main.setting == "transductive")]
    base = d[d.method == PROPOSED].set_index("seed")[metric]
    if base.empty:
        return pd.DataFrame()
    pv, rows = {}, []
    for name, g in d[d.method != PROPOSED].groupby("method"):
        o = g.set_index("seed")[metric]
        common = base.index.intersection(o.index)
        if len(common) < 3:
            continue
        t = paired_t(base[common], o[common])
        pv[name] = t["p"]
        rows.append({"Compared with": name, "Seeds": len(common), "Mean diff (pp)": round(t["mean_diff"] * 100, 2),
                     "Cohen d_z": round(t["cohen_dz"], 2), "p": f"{t['p']:.4f}"})
    if not rows:
        return pd.DataFrame()
    h = holm(pv)
    for r in rows:
        r["p (Holm)"] = f"{h[r['Compared with']]['p_holm']:.4f}"
        if r["p"] == "nan":
            r["p"] = "n/a"
        r["Significant (0.05)"] = "yes" if h[r["Compared with"]]["reject"] else "no"
    return pd.DataFrame(rows).sort_values("Mean diff (pp)")


def ablation_table(main, ds):
    d = main[(main.dataset == ds) & (main.setting == "transductive") & main.group.isin(["proposed", "ablation"])]
    rows = []
    for name, g in d.groupby("method"):
        rows.append({"Variant": name, "Seeds": len(g), "Val F1": fmt(g.val_f1_bot), "F1": fmt(g.test_f1_bot),
                     "MCC": fmt(g.test_mcc), "ROC-AUC": fmt(g.test_roc_auc), "_k": 0 if name == PROPOSED else 1})
    return pd.DataFrame(rows).sort_values(["_k", "Variant"]).drop(columns="_k") if rows else pd.DataFrame()


MODALITY_ROWS = [  # (method name, metadata, text, graph, architecture)
    ("LightGBM (meta)", "yes", "", "", "tree model"),
    ("LightGBM (text)", "", "yes", "", "tree model"),
    ("LightGBM (meta+text)", "yes", "yes", "", "tree model"),
    ("GCN (meta)", "yes", "", "yes", "GCN end-to-end"),
    ("GCN (text)", "", "yes", "yes", "GCN end-to-end"),
    ("GCN", "yes", "yes", "yes", "GCN end-to-end"),
    ("Reconstructed hybrid (P3, hidden)", "via GCN", "via GCN", "yes", "P3: GCN 64-d embedding -> LightGBM"),
    ("Reconstructed hybrid (P3, prob)", "via GCN", "via GCN", "yes", "P3: GCN 2-d probabilities -> LightGBM"),
    ("Reconstructed hybrid (P1, hidden)", "yes", "yes", "yes", "P1: meta + text + GCN 64-d -> LightGBM"),
    ("Reconstructed hybrid (P1, prob)", "yes", "yes", "yes", "P1: meta + text + GCN 2-d -> LightGBM"),
    ("RGF (proposed)", "yes", "yes", "yes", "reliability-gated experts"),
]


def modality_table(main, ds):
    d = main[(main.dataset == ds) & (main.setting == "transductive")]
    rows = []
    for name, me, te, gr, arch in MODALITY_ROWS:
        g = d[d.method == name]
        if g.empty:
            continue
        rows.append({"Method": name, "Metadata": me, "Text": te, "Graph": gr, "Architecture": arch,
                     "Seeds": len(g), "F1": fmt(g.test_f1_bot), "MCC": fmt(g.test_mcc), "ROC-AUC": fmt(g.test_roc_auc)})
    return pd.DataFrame(rows)


def robustness_table(rob, ds, metric="f1_bot"):
    d = rob[rob.dataset == ds]
    if d.empty:
        return pd.DataFrame()
    t = d.groupby(["perturbation", "level", "method"])[metric].mean().unstack("method") * 100
    return t.round(2).reset_index()


def inductive_table(main, ds):
    d = main[(main.dataset == ds)]
    rows = []
    for name, g in d.groupby("method"):
        tr, ind = g[g.setting == "transductive"], g[g.setting == "inductive"]
        if ind.empty:
            continue
        rows.append({"Method": name, "Transductive F1": fmt(tr.test_f1_bot), "Inductive F1": fmt(ind.test_f1_bot),
                     "Transductive MCC": fmt(tr.test_mcc), "Inductive MCC": fmt(ind.test_mcc)})
    return pd.DataFrame(rows)


def calibration_table(main, ds):
    d = main[(main.dataset == ds) & (main.setting == "transductive") & (main.group != "ablation")]
    rows = []
    for name, g in d.groupby("method"):
        r = {"Method": name, "ECE": fmt(g.test_ece, 100, 2), "Brier": fmt(g.test_brier, 1, 4),
             "Error @80% coverage": fmt(g.test_risk_at_80), "Error @90% coverage": fmt(g.test_risk_at_90)}
        if "test_uncal_ece" in g and g.test_uncal_ece.notna().any():
            r["ECE before temperature scaling"] = fmt(g.test_uncal_ece, 100, 2)
        rows.append(r)
    return pd.DataFrame(rows).fillna("—")


def cross_table(cross):
    if cross.empty:
        return pd.DataFrame()
    rows = []
    for (s, t, name), g in cross.groupby(["source", "target", "method"]):
        spec = g.tn / (g.tn + g.fp).where((g.tn + g.fp) > 0)
        r = {"Train on": DS_NAME[s], "Test on": DS_NAME[t], "Method": name, "Seeds": len(g)}
        if "overlap_removed" in g:
            r.update({"Shared accounts removed": int(g.overlap_removed.iloc[0]),
                      "Test bots": int(g.target_bots.iloc[0]), "Test humans": int(g.target_humans.iloc[0])})
        # a target test set with no bots left (all shared) has no bot-class metrics: report specificity only
        two = "target_bots" not in g or (g.target_bots.iloc[0] > 0 and g.target_humans.iloc[0] > 0)
        r.update({"Acc": fmt(g.accuracy), "Specificity": fmt(spec),
                  "F1": fmt(g.f1_bot) if two else "— (no bots)", "MCC": fmt(g.mcc) if two else "—",
                  "ROC-AUC": fmt(g.roc_auc) if two else "—"})
        rows.append(r)
    return pd.DataFrame(rows)


def explain_tables(ex, ds):
    d = ex[ex.dataset == ds]
    out = {}
    if d.empty:
        return out
    gw = d[d.kind == "gate_weight"]
    if not gw.empty:
        out["gate"] = gw.groupby("group")[["n", "w_meta", "w_text", "w_graph"]].mean().round(3).reset_index()
    ea = d[d.kind == "expert_alone"]
    if not ea.empty:
        out["experts"] = ea.groupby("group")[["n", "f1_bot", "mcc"]].mean().round(4).reset_index()
    pi = d[d.kind == "permutation_importance"]
    if not pi.empty:
        out["perm"] = (pi.groupby("group")[["delta_f1", "delta_mcc"]].mean() * 100).round(2) \
            .sort_values("delta_f1", ascending=False).reset_index()
    lg = d[d.kind == "lgbm_split_importance"]
    if not lg.empty:
        out["lgbm"] = lg.groupby("group")[["share"]].mean().round(4).sort_values("share", ascending=False).reset_index()
    return out


def efficiency_table(main, ds):
    d = main[(main.dataset == ds) & (main.setting == "transductive") & (main.group != "ablation")]
    rows = []
    for name, g in d.groupby("method"):
        rows.append({"Method": name, "Fit time (s)": f"{g.fit_seconds.mean():.1f}",
                     "Inference (ms / 1k accounts)": f"{g.predict_ms_per_1k.mean():.1f}",
                     "Size (params or trees)": int(g["size"].mean()) if "size" in g and g["size"].notna().any() else "—"})
    return pd.DataFrame(rows)


def reconstruction_summary(results_dir):
    path = os.path.join(results_dir, "runs.csv")
    if not os.path.exists(path):
        return None
    from .report import build
    return build(path)


# ---------------------------------------------------------------- figures

def figures(results_dir, out_dir, main, rob):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
    made = []
    show = [PROPOSED, "RGF w/o gate (mean of experts)", "GCN", "GraphSAGE", "GAT", "LightGBM (meta+text)"]
    for ds in sorted(rob.dataset.unique()) if not rob.empty else []:
        kinds = [k for k in ["missing_text", "drop_edges", "noisy_edges", "metadata_mimicry"]
                 if k in set(rob[rob.dataset == ds].perturbation)]
        hyb = best_hybrid(main, ds)
        fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.4), squeeze=False)
        for ax, k in zip(axes[0], kinds):
            d = rob[(rob.dataset == ds) & (rob.perturbation == k)]
            for name in show + ([hyb] if hyb else []):
                g = d[d.method == name].groupby("level").f1_bot.agg(["mean", "std"])
                if g.empty:
                    continue
                ax.errorbar(g.index, g["mean"] * 100, yerr=g["std"].fillna(0) * 100, marker="o", ms=3, capsize=2,
                            lw=2.2 if name == PROPOSED else 1.1, label=name)
            ax.set_title(k.replace("_", " ")); ax.set_xlabel("share of test accounts / edges affected")
            ax.set_ylabel("bot-class F1 (%)"); ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{DS_NAME[ds]}: robustness to test-time perturbation (mean ± SD over seeds)")
        fig.tight_layout()
        p = os.path.join(out_dir, f"fig_robustness_{ds}.png"); fig.savefig(p, dpi=200); plt.close(fig); made.append(p)
    preds = glob.glob(os.path.join(results_dir, "preds", "*transductive*_0.npz"))
    by = {}
    for f in preds:
        base = os.path.basename(f)
        ds = next((d for d in DS_ORDER if base.startswith(d + "_")), None)
        if ds is None:
            continue
        by.setdefault(ds, {})[base] = f
    for ds, files in by.items():
        sel = {}
        for name in [PROPOSED, "GCN", "GraphSAGE", "LightGBM (meta+text)"] + ([best_hybrid(main, ds)] if best_hybrid(main, ds) else []):
            key = f"{ds}_transductive_{name}_0".replace(" ", "_").replace("/", "-") + ".npz"
            if key in files:
                sel[name] = np.load(files[key])
        if not sel:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for name, z in sel.items():
            fpr, tpr, _ = roc_curve(z["y"], z["p"]); axes[0].plot(fpr, tpr, label=name, lw=2 if name == PROPOSED else 1)
            pr, rc, _ = precision_recall_curve(z["y"], z["p"]); axes[1].plot(rc, pr, label=name, lw=2 if name == PROPOSED else 1)
            rel = reliability_curve(z["y"], z["p"])
            if rel:
                axes[2].plot([r[0] for r in rel], [r[1] for r in rel], marker="o", ms=3, label=name)
        axes[0].plot([0, 1], [0, 1], "k:", lw=0.8); axes[2].plot([0.5, 1], [0.5, 1], "k:", lw=0.8)
        for ax, (xl, yl, t) in zip(axes, [("FPR", "TPR", "ROC"), ("Recall", "Precision", "Precision-recall"),
                                          ("Confidence", "Accuracy", "Reliability (seed 0)")]):
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{DS_NAME[ds]}, test set, seed 0"); fig.tight_layout()
        p = os.path.join(out_dir, f"fig_curves_{ds}.png"); fig.savefig(p, dpi=200); plt.close(fig); made.append(p)
        if PROPOSED in sel:
            z = sel[PROPOSED]
            cm = confusion_matrix(z["y"], (z["p"] >= 0.5).astype(int), labels=[0, 1])
            fig, ax = plt.subplots(figsize=(3.4, 3.2))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["human", "bot"]); ax.set_yticks([0, 1], ["human", "bot"])
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{DS_NAME[ds]}: RGF, seed 0")
            fig.tight_layout()
            p = os.path.join(out_dir, f"fig_confusion_{ds}.png"); fig.savefig(p, dpi=200); plt.close(fig); made.append(p)
    return made


# ---------------------------------------------------------------- facts for drafting

def facts(main, rob, cross, results_dir):
    out = {"_note": "Every number below comes from result files; provenance given per entry. "
                    "MANUSCRIPT-REPORTED and LITERATURE values are copied from their sources, not produced here."}
    for ds in sorted(main.dataset.unique()):
        d = main[(main.dataset == ds) & (main.setting == "transductive")]
        f = ({"manuscript_reported": {**P.MANUSCRIPT_RESULTS[ds], "provenance": P.MANUSCRIPT_REPORTED}}
             if INCLUDE_OLD_MANUSCRIPT and ds in P.MANUSCRIPT_RESULTS else {})
        for name, g in d.groupby("method"):
            f[name] = {"provenance": g.provenance.iloc[0], "seeds": int(len(g)),
                       **{lab: mean_ci(g[col].to_numpy() * 100) for col, lab in METRICS if g[col].notna().any()}}
        hyb = best_hybrid(main, ds)
        f["_best_reconstructed_hybrid_by_validation"] = hyb
        if not d.empty:
            f["_data"] = {k: float(d[k].iloc[0]) for k in ("edge_homophily_train", "isolated_share_test",
                                                           "text_missing_share_test", "empty_text_rate_bot",
                                                           "empty_text_rate_human") if k in d}
        out[ds] = f
    return out


def build(results_dir: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    main = _read(os.path.join(results_dir, "suite_main.jsonl"))
    rob = _read(os.path.join(results_dir, "suite_robust.jsonl"))
    cross = _read(os.path.join(results_dir, "suite_cross.jsonl"))
    ex = _read(os.path.join(results_dir, "suite_explain.jsonl"))
    lines = ["# Paper tables (generated from result files; do not edit numbers by hand)\n",
             "Provenance: THIS-WORK = new experiments; REPRODUCED-CORRECTED = BERT/SBERT + GCN + LightGBM hybrid "
             "baseline implemented here under the leakage-free protocol. All numbers come from this run. "
             "Metrics are bot-class unless stated; mean ± SD over seeds.\n"]
    if main.empty:
        lines.append("\n_No suite results yet._\n")
    for ds in [d for d in DS_ORDER if not main.empty and d in set(main.dataset)]:
        n = DS_NAME[ds]
        d0 = main[main.dataset == ds].iloc[0]
        lines.append(f"\n## {n}\n")
        lines.append(f"\n### T1 Data ({n})\n")
        lines.append(md(pd.DataFrame([{"Train": int(d0.n_train), "Validation": int(d0.n_val), "Test": int(d0.n_test),
                                       "Edge homophily (train)": round(d0.edge_homophily_train, 3),
                                       "Test accounts without graph links": f"{d0.isolated_share_test * 100:.1f}%",
                                       "Test accounts without text": f"{d0.text_missing_share_test * 100:.1f}%"}])))
        tabs = {
            "T2 Main comparison": main_table(main, ds),
            "T3 Ablation of the proposed model": ablation_table(main, ds),
            "T3b Modality ablation (same split and protocol for every row)": modality_table(main, ds),
            "T4 Robustness, bot-class F1 (%)": robustness_table(rob, ds),
            "T5 Unseen accounts: transductive vs inductive": inductive_table(main, ds),
            "T6 Calibration and selective prediction": calibration_table(main, ds),
            "T9 Efficiency on CPU": efficiency_table(main, ds),
            "T10 Paired t-tests, RGF vs each method (bot-class F1, Holm-corrected)": significance(main, ds),
            "T10b Paired t-tests, RGF vs each method (MCC, Holm-corrected)": significance(main, ds, "test_mcc"),
        }
        for title, t in tabs.items():
            lines.append(f"\n### {title} ({n})\n")
            lines.append(md(t))
            if not t.empty:
                t.to_csv(os.path.join(out_dir, f"{title.split()[0]}_{ds}.csv"), index=False)
        for k, t in explain_tables(ex, ds).items():
            title = {"gate": "T8a Mean gate weights of RGF by account group",
                     "experts": "T8b Each RGF expert alone (accounts where its modality exists)",
                     "perm": "T8c Permutation importance for RGF (drop in F1 / MCC, percentage points)",
                     "lgbm": "T8d LightGBM (meta+text) split-importance share"}[k]
            lines.append(f"\n### {title} ({n})\n")
            lines.append(md(t))
            t.to_csv(os.path.join(out_dir, f"T8{k}_{ds}.csv"), index=False)
    ct = cross_table(cross)
    lines.append("\n## T7 Cross-dataset generalisation (train on one, test on the other)\n")
    lines.append(md(ct))
    rec = reconstruction_summary(results_dir)
    if rec:
        lines.append("\n## Reconstruction study (manuscript-faithful vs corrected)\n")
        lines.append(rec)
    figs = figures(results_dir, out_dir, main, rob) if not main.empty else []
    lines.append("\n## Figures\n" + "\n".join(f"- {os.path.basename(f)}" for f in figs) + "\n")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    for name, df in (("suite_main", main), ("suite_robust", rob), ("suite_cross", cross), ("suite_explain", ex)):
        if not df.empty:
            df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)      # flat copies for spreadsheets
    if not main.empty:
        with open(os.path.join(out_dir, "paper_facts.json"), "w") as f:
            json.dump(facts(main, rob, cross, results_dir), f, indent=1, default=float)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/paper")
    a = ap.parse_args(argv)
    print(build(a.results, a.out))


if __name__ == "__main__":
    main()
