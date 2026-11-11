"""Figures 5-12: experimental graphs, computed directly from data/raw/suite_*.csv.
Each function also writes the exact numbers it plots to data/derived/figXX_*.csv.
Run:  python plots.py
Aggregation rules (identical to the run's own report, botrecon/suite_report.py):
  mean and sample SD over seeds; clean results 5 seeds (0-4); robustness and cross-dataset 3 seeds (0-2);
  cost = mean over the 5 transductive seeds. Metrics are stored as fractions and shown in percent.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import (style, save, load, DER, DATASETS, SHORT, MAIN, ROBUST, ABLATIONS, STYLE, TWO_COL, ONE_COL)

DS_COL = {"cresci2017": "#0072B2", "cresci2015": "#D55E00"}
DS_MK = {"cresci2017": "o", "cresci2015": "s"}


def label_rgf(ax, names):
    for t in ax.get_yticklabels():
        if t.get_text().startswith("RGF"):
            t.set_fontweight("bold")


def dotplot(ax, df, methods, col, sdcol, xlabel, dodge=0.16, hollow=None):
    """Horizontal dot plot: methods on y, one marker per dataset with SD error bars."""
    ys = np.arange(len(methods))[::-1]
    for j, ds in enumerate(DATASETS):
        d = df[df.dataset == ds].set_index("method")
        for y, m in zip(ys, methods):
            if m not in d.index:
                continue
            ax.errorbar(d.loc[m, col], y + (dodge if j == 0 else -dodge), xerr=d.loc[m, sdcol], fmt=DS_MK[ds],
                        color=DS_COL[ds], ms=3.6, elinewidth=0.7, capsize=1.5, mfc=DS_COL[ds], zorder=3)
    ax.set_yticks(ys, [SHORT[m] for m in methods])
    ax.set_xlabel(xlabel)
    ax.axhspan(ys[methods.index("RGF (proposed)")] - 0.45, ys[methods.index("RGF (proposed)")] + 0.45,
               color="#eeeeee", zorder=0) if "RGF (proposed)" in methods else None
    ax.grid(axis="y", visible=False)
    label_rgf(ax, methods)


def ds_legend(fig, extra=(), y=1.0, ncol=None):
    h = [Line2D([], [], color=DS_COL[d], marker=DS_MK[d], ls="", ms=4, label=DATASETS[d]) for d in DATASETS]
    h += list(extra)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(h), handletextpad=0.3,
               columnspacing=1.2)


def agg(df, keys, cols):
    g = df.groupby(keys)[cols].agg(["mean", "std", "count"])
    g.columns = [f"{c}_{s}" if s != "count" else f"{c}_n" for c, s in g.columns]
    return g.reset_index()


# ------------------------------------------------------------------ Fig. 5 clean performance
def fig5():
    m = load("main")
    m = m[(m.setting == "transductive") & m.method.isin(MAIN)].copy()
    for c in ("test_f1_bot", "test_mcc", "test_roc_auc"):
        m[c] *= 100
    a = agg(m, ["dataset", "method"], ["test_f1_bot", "test_mcc", "test_roc_auc"])
    a.to_csv(DER / "fig05_clean_performance.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(TWO_COL, 2.55), sharey=True)
    for ax, (c, lab) in zip(axs, [("test_f1_bot", "Bot-class F1 (%)"), ("test_mcc", "MCC (%)"),
                                  ("test_roc_auc", "ROC-AUC (%)")]):
        dotplot(ax, a, MAIN, f"{c}_mean", f"{c}_std", lab)
    for ax, t in zip(axs, "abc"):
        ax.set_title(f"({t})", loc="left")
    ds_legend(fig, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig05_clean_performance")


# ------------------------------------------------------------------ robustness helper
def robust(pert, methods=ROBUST):
    r = load("robust")
    r = r[(r.perturbation == pert) & r.method.isin(methods)].copy()
    r["f1"] = 100 * r.f1_bot
    return agg(r, ["dataset", "method", "level"], ["f1"])


def lines(ax, a, ds, methods, ylab=True, ref_flat=None):
    d = a[a.dataset == ds]
    for m in methods:
        g = d[d.method == m].sort_values("level")
        if g.empty:
            continue
        c, mk = STYLE[m]
        ls = "--" if m == ref_flat else "-"
        lw, z = (1.6, 5) if m.startswith("RGF") else (1.0, 3)
        ax.plot(100 * g.level, g.f1_mean, ls=ls, marker=mk, color=c, lw=lw, ms=4.5 if mk == "*" else 3.2, zorder=z,
                label=SHORT[m])
        ax.fill_between(100 * g.level, g.f1_mean - g.f1_std, g.f1_mean + g.f1_std, color=c, alpha=0.12, lw=0, zorder=1)
    if ylab:
        ax.set_ylabel("Bot-class F1 (%)")


def method_legend(fig, methods, y, ncol=7):
    h = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], lw=1.6 if m.startswith("RGF") else 1.0,
                ms=5 if STYLE[m][1] == "*" else 3.5, label=SHORT[m]) for m in methods]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.8,
               columnspacing=1.0, handletextpad=0.4)


# ------------------------------------------------------------------ Fig. 6 accuracy vs robustness
def fig6():
    a = robust("metadata_mimicry")
    a.to_csv(DER / "fig06_mimicry_f1.csv", index=False)
    rows = []
    for ds in DATASETS:
        d = a[a.dataset == ds]
        for lev in (0.0, 1.0):
            e = d[d.level == lev].sort_values("f1_mean", ascending=False).reset_index(drop=True)
            for i, r in e.iterrows():
                rows.append({"dataset": ds, "level": lev, "method": r.method, "f1_mean": r.f1_mean, "rank": i + 1})
    rk = pd.DataFrame(rows)
    rk.to_csv(DER / "fig06_mimicry_ranks.csv", index=False)
    fig = plt.figure(figsize=(TWO_COL, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.28)
    for j, ds in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, j])
        lines(ax, a, ds, ROBUST, ylab=(j == 0))
        ax.set_xlabel(r"Test bots given human-median metadata, $\rho$ (%)")
        ax.set_xticks([0, 25, 50, 100]); ax.set_ylim(-3, 101)
        ax.set_title(f"({'ab'[j]}) {DATASETS[ds]}", loc="left")
        bx = fig.add_subplot(gs[1, j])
        d = rk[rk.dataset == ds]
        n = d.method.nunique()
        for m in d.method.unique():
            r0 = d[(d.method == m) & (d.level == 0.0)].iloc[0]; r1 = d[(d.method == m) & (d.level == 1.0)].iloc[0]
            c, mk = STYLE[m]
            lw = 1.8 if m.startswith("RGF") else 1.0
            bx.plot([0, 1], [r0["rank"], r1["rank"]], color=c, lw=lw, marker=mk, ms=5 if mk == "*" else 3.5, zorder=4 if lw > 1 else 2)
            w = "bold" if m.startswith("RGF") else "normal"
            bx.text(-0.04, r0["rank"], f"{SHORT[m]}  {r0.f1_mean:.1f}", ha="right", va="center", fontsize=6.2, weight=w)
            bx.text(1.04, r1["rank"], f"{r1.f1_mean:.1f}  {SHORT[m]}", ha="left", va="center", fontsize=6.2, weight=w)
        bx.set_xlim(-1.05, 2.05); bx.set_ylim(n + 0.5, 0.5); bx.axis("off")
        bx.text(0, n + 1.0, "clean", ha="center", fontsize=6.4, weight="bold")
        bx.text(1, n + 1.0, "ρ = 100%", ha="center", fontsize=6.4, weight="bold")
        bx.set_title(f"({'cd'[j]}) {DATASETS[ds]}: rank by F1 (%), clean → full mimicry", loc="left", fontsize=7.5)
    method_legend(fig, ROBUST, 0.965)
    save(fig, "fig06_accuracy_vs_robustness")


# ------------------------------------------------------------------ Fig. 7 missing text
def fig7():
    a = robust("missing_text")
    a.to_csv(DER / "fig07_missing_text_f1.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.45))
    for j, (ax, ds) in enumerate(zip(axs, DATASETS)):
        lines(ax, a, ds, ROBUST, ylab=(j == 0))
        ax.set_xlabel(r"Test accounts with text removed, $\rho$ (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"({'ab'[j]}) {DATASETS[ds]}", loc="left")
    method_legend(fig, ROBUST, 0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig07_missing_text")


# ------------------------------------------------------------------ Fig. 8 graph perturbation
def fig8():
    frames = []
    for p in ("drop_edges", "noisy_edges"):
        x = robust(p); x["perturbation"] = p; frames.append(x)
    a = pd.concat(frames)
    a.to_csv(DER / "fig08_graph_perturbation_f1.csv", index=False)
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 4.1), sharex="col")
    lab = {"drop_edges": r"Test-node edges deleted, $\rho$ (%)", "noisy_edges": r"Random edges injected per test node, $\rho$ (%)"}
    for i, ds in enumerate(DATASETS):
        for j, p in enumerate(("drop_edges", "noisy_edges")):
            ax = axs[i, j]
            lines(ax, a[a.perturbation == p], ds, ROBUST, ylab=(j == 0), ref_flat="LightGBM (meta+text)")
            ax.set_title(f"({'abcd'[2 * i + j]}) {DATASETS[ds]}, {'edge deletion' if j == 0 else 'edge injection'}", loc="left")
            if i == 1:
                ax.set_xlabel(lab[p])
            ax.set_xticks([0, 25, 50, 75, 100] if j == 0 else [0, 25, 50, 100])
    method_legend(fig, ROBUST, 0.975)
    fig.text(0.5, -0.01, "LightGBM (dashed) uses no graph and is a constant reference. Cresci-2017 graph = kNN on metadata; "
             "R-GCN is run on Cresci-2015 only.", ha="center", fontsize=6.2, color="#555555")
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    save(fig, "fig08_graph_perturbation")


# ------------------------------------------------------------------ Fig. 9 cross-dataset
CROSS = ["LightGBM (meta+text)", "GCN", "GraphSAGE", "Reconstructed hybrid (P1, hidden)", "RGF (proposed)"]


def fig9():
    c = load("cross")
    c = c[c.method.isin(CROSS)].copy()
    c["fpr_human"] = 100 * c.fp / (c.fp + c.tn)
    for k in ("f1_bot", "mcc", "roc_auc"):
        c[k] = 100 * c[k]
    a = agg(c, ["source", "target", "method"], ["f1_bot", "mcc", "roc_auc", "fpr_human"])
    a.to_csv(DER / "fig09_cross_dataset.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    d = a[a.source == "cresci2015"].set_index("method").loc[CROSS]
    x = np.arange(3); w = 0.16
    for i, m in enumerate(CROSS):
        vals = [d.loc[m, f"{k}_mean"] for k in ("f1_bot", "mcc", "roc_auc")]
        sds = [d.loc[m, f"{k}_std"] for k in ("f1_bot", "mcc", "roc_auc")]
        ax.bar(x + (i - 2) * w, vals, w, yerr=sds, color=STYLE[m][0], edgecolor="black", lw=0.4,
               error_kw=dict(lw=0.6, capsize=1.2), label=SHORT[m], hatch="///" if m.startswith("RGF") else None)
    ax.set_xticks(x, ["Bot-class F1", "MCC", "ROC-AUC"]); ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
    ax.set_title("(a) Train Cresci-2015 → test Cresci-2017", loc="left"); ax.grid(axis="x", visible=False)
    ax = axs[1]
    x = np.arange(2)
    for i, m in enumerate(CROSS):
        vals, sds = [], []
        for src in ("cresci2015", "cresci2017"):
            r = a[(a.source == src) & (a.method == m)].iloc[0]
            vals.append(r.fpr_human_mean); sds.append(r.fpr_human_std)
        ax.bar(x + (i - 2) * w, vals, w, yerr=sds, color=STYLE[m][0], edgecolor="black", lw=0.4,
               error_kw=dict(lw=0.6, capsize=1.2), hatch="///" if m.startswith("RGF") else None)
    ax.set_xticks(x, ["C-15 → C-17\n(695 humans)", "C-17 → C-15\n(390 humans)"])
    ax.set_ylabel("Humans flagged as bots (%)"); ax.set_ylim(0, 100)
    ax.set_title("(b) Human false-positive rate", loc="left"); ax.grid(axis="x", visible=False)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.83, 1.1), ncol=5, columnspacing=1.0, handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig09_cross_dataset")


# ------------------------------------------------------------------ Fig. 10 ablations
def fig10():
    m = load("main")
    m = m[(m.setting == "transductive") & m.method.isin(ABLATIONS)].copy()
    m["f1"] = 100 * m.test_f1_bot; m["mcc"] = 100 * m.test_mcc
    a = agg(m, ["dataset", "method"], ["f1", "mcc"])
    a.to_csv(DER / "fig10_ablation_clean.csv", index=False)
    rb = load("robust")
    rb = rb[rb.method.isin(ABLATIONS) & (rb.level == 1.0) & rb.perturbation.isin(["metadata_mimicry", "missing_text"])].copy()
    rb["f1"] = 100 * rb.f1_bot
    b = agg(rb, ["dataset", "perturbation", "method"], ["f1"])
    b.to_csv(DER / "fig10_ablation_robust.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(TWO_COL, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    dotplot(axs[0], a, ABLATIONS, "f1_mean", "f1_std", "Bot-class F1 (%)")
    dotplot(axs[1], a, ABLATIONS, "mcc_mean", "mcc_std", "MCC (%)")
    axs[1].set_yticklabels([])
    axs[0].set_title("(a) Clean F1", loc="left"); axs[1].set_title("(b) Clean MCC", loc="left")
    ax = axs[2]
    have = [v for v in ABLATIONS if v in set(b.method)]
    groups = [("cresci2017", "metadata_mimicry"), ("cresci2015", "metadata_mimicry"),
              ("cresci2017", "missing_text"), ("cresci2015", "missing_text")]
    x = np.arange(len(groups)); w = 0.2
    shade = {"RGF (proposed)": "#000000", "RGF concat fusion": "#8c8c8c", "RGF w/o gate (mean of experts)": "#56B4E9",
             "RGF w/o modality dropout": "#E69F00"}
    for i, v in enumerate(have):
        vals = [b[(b.dataset == d) & (b.perturbation == p) & (b.method == v)].f1_mean.iloc[0] for d, p in groups]
        sds = [b[(b.dataset == d) & (b.perturbation == p) & (b.method == v)].f1_std.iloc[0] for d, p in groups]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=sds, color=shade[v], edgecolor="black", lw=0.4,
               error_kw=dict(lw=0.6, capsize=1.2), label=SHORT[v])
    ax.set_xticks(x, ["C-17\nmimicry", "C-15\nmimicry", "C-17\nno text", "C-15\nno text"])
    ax.set_ylabel("Bot-class F1 (%)"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    ax.set_title("(c) At ρ = 100% (3 seeds)", loc="left")
    from matplotlib.patches import Patch
    bars = [Patch(fc=shade[v], ec="black", lw=0.4, label=SHORT[v]) for v in have]
    ds_legend(fig, extra=bars, y=0.95, ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig10_ablation")


# ------------------------------------------------------------------ Fig. 11 calibration, selective prediction
def fig11():
    m = load("main")
    m = m[(m.setting == "transductive") & m.method.isin(MAIN)].copy()
    for k in ("test_ece", "test_risk_at_80", "test_risk_at_90", "test_uncal_ece"):
        m[k] = 100 * m[k]
    a = agg(m, ["dataset", "method"], ["test_ece", "test_brier", "test_risk_at_80", "test_risk_at_90", "test_uncal_ece"])
    a.to_csv(DER / "fig11_calibration_selective.csv", index=False)
    fig, axs = plt.subplots(1, 4, figsize=(TWO_COL, 2.6), sharey=True)
    spec = [("test_ece", "ECE (%)"), ("test_brier", "Brier score"), ("test_risk_at_80", "Error at 80% coverage (%)"),
            ("test_risk_at_90", "Error at 90% coverage (%)")]
    for ax, (c, lab), t in zip(axs, spec, "abcd"):
        dotplot(ax, a, MAIN, f"{c}_mean", f"{c}_std", lab)
        ax.set_title(f"({t})", loc="left")
    # RGF before temperature scaling (hollow markers)
    ys = np.arange(len(MAIN))[::-1]; yr = ys[MAIN.index("RGF (proposed)")]
    for j, ds in enumerate(DATASETS):
        r = a[(a.dataset == ds) & (a.method == "RGF (proposed)")].iloc[0]
        axs[0].plot(r.test_uncal_ece_mean, yr + (0.16 if j == 0 else -0.16), DS_MK[ds], mfc="white", mec=DS_COL[ds],
                    ms=3.6, zorder=4)
    axs[0].set_xscale("log"); axs[0].set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    hollow = Line2D([], [], marker="o", mfc="white", mec="#555555", ls="", ms=4, label="RGF before temperature scaling")
    ds_legend(fig, extra=[hollow], y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig11_calibration_selective")


# ------------------------------------------------------------------ Fig. 12 cost
def fig12():
    m = load("main")
    m = m[(m.setting == "transductive") & m.method.isin(MAIN)]
    a = agg(m, ["dataset", "method"], ["fit_seconds", "predict_ms_per_1k", "size"])
    a.to_csv(DER / "fig12_cost.csv", index=False)
    trees = {"LightGBM (meta+text)", "XGBoost (meta+text)", "Random Forest (meta+text)"}
    fig, axs = plt.subplots(1, 3, figsize=(TWO_COL, 2.55), sharey=True)
    spec = [("fit_seconds", "Training time (s)"), ("predict_ms_per_1k", "Inference time (ms per 1,000 accounts)"),
            ("size", "Model size (parameters or trees)")]
    for ax, (c, lab), t in zip(axs, spec, "abc"):
        dotplot(ax, a, MAIN, f"{c}_mean", f"{c}_std", lab)
        ax.set_xscale("log"); ax.set_title(f"({t})", loc="left")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    from matplotlib.ticker import FixedLocator
    for ax, tk in zip(axs, ([1, 3, 10, 30, 100], [1, 3, 10, 30], [100, 1000, 10000, 100000])):
        ax.set_xticks(tk, [f"{t:,}" for t in tk])
    ys = np.arange(len(MAIN))[::-1]
    for tr in trees:
        axs[2].text(1.5e5, ys[MAIN.index(tr)], "trees", fontsize=5.8, va="center", color="#555555")
    yr = ys[MAIN.index("RGF (proposed)")]
    axs[2].plot(38089, yr, "|", color="black", ms=7, mew=1.2)
    axs[2].annotate("38,089 used\n(50,571 registered)", (38089, yr), xytext=(120, yr + 0.62), fontsize=5.8,
                    arrowprops=dict(arrowstyle="-", lw=0.5))
    axs[2].set_xlim(30, 4e5)
    ds_legend(fig, y=0.97)
    fig.text(0.5, -0.02, "CPU (Kaggle), mean ± SD over 5 seeds, transductive setting. Tree models count trees, neural models "
             "count trainable parameters; the two are not comparable units.", ha="center", fontsize=6.2, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig12_computational_cost")


if __name__ == "__main__":
    style()
    for f in (fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12):
        f()
