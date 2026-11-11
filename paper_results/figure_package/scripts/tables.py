"""Tables 1-15. Every numeric cell is computed here from data/raw (result CSVs and data_audit.json);
configuration tables (2, 3) and the related-work table (15) are transcribed from the code and from the
verified reference library, as noted in each table's footnote. Outputs per table: CSV, Markdown, LaTeX.
Run:  python tables.py
"""
import json
import numpy as np
import pandas as pd
from scipy import stats as st
from common import RAW, TAB, DATASETS, SHORT, MAIN, ROBUST, ABLATIONS, load

MD, TEX = [], []
PM = " ± "


def f(m, s, d=2):
    return "—" if pd.isna(m) else (f"{m:.{d}f}" if pd.isna(s) else f"{m:.{d}f}{PM}{s:.{d}f}")


def esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("×10⁻⁴", r"$\times10^{-4}$"), ("α", r"$\alpha$"), ("Δ", r"$\Delta$"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("→", r"$\rightarrow$"), ("ρ", r"$\rho$"), ("—", "--"), ("≤", r"$\le$"),
                 ("×", r"$\times$"), ("†", r"$^\dagger$"), ("‡", r"$^\ddagger$"), ("λ", r"$\lambda$"),
                 ("∈", r"$\in$"), ("≥", r"$\ge$"), ("−", "-")):
        s = s.replace(a, b)
    return s


def emit(no, slug, df, caption, notes, wide=False, size=r"\footnotesize", spec=None):
    name = f"table{no:02d}_{slug}"
    df.to_csv(TAB / f"{name}.csv", index=False)
    md = [f"### TABLE {no}. {caption}", "", "| " + " | ".join(map(str, df.columns)) + " |",
          "|" + "---|" * len(df.columns)]
    for r in df.astype(str).values:
        r = list(r)
        if r[0].startswith("§"):
            r[0] = f"**{r[0][1:]}**"
        md.append("| " + " | ".join(r) + " |")
    md += ["", *[f"*{n}*" for n in notes], ""]
    (TAB / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    MD.extend(md)
    env = "table*" if wide else "table"
    cols = spec or ("l" + "c" * (len(df.columns) - 1))
    tex = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{slug}}}", size,
           r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{cols}}}", r"\hline",
           " & ".join(rf"\textbf{{{esc(c)}}}" for c in df.columns) + r" \\", r"\hline"]
    for r in df.astype(str).values:
        cells = [esc(c) for c in r]
        if str(r[0]).startswith("RGF (proposed)") or str(r[0]).startswith("**"):
            cells = [rf"\textbf{{{c.replace('**', '')}}}" for c in cells]
        if str(r[0]).startswith("§"):           # section row inside a table
            tex.append(rf"\multicolumn{{{len(cells)}}}{{l}}{{\textit{{{esc(r[0][1:])}}}}} \\")
            continue
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    if notes:
        tex.append(r"\par\smallskip\parbox{\linewidth}{\scriptsize " + " ".join(esc(n) for n in notes) + "}")
    tex.append(rf"\end{{{env}}}")
    (TAB / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
    TEX.append("\n".join(tex))


def agg(df, keys, cols):
    g = df.groupby(keys)[cols].agg(["mean", "std"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g.reset_index()


main = load("main"); rob = load("robust"); cross = load("cross"); expl = load("explain")
audit = json.loads((RAW / "data_audit.json").read_text())
tr = main[main.setting == "transductive"].copy()

# ---------------------------------------------------------------- T1 dataset statistics
d17, d15 = audit["cresci2017"], audit["cresci2015"]
first = lambda ds, c: tr[tr.dataset == ds][c].iloc[0]
rows = [
    ("Accounts (after deduplication)", f"{d17['n']:,}", f"{d15['n']:,}"),
    ("Bots / humans", f"{d17['n_bot']:,} / {d17['n_human']:,}", f"{d15['n_bot']:,} / {d15['n_human']:,}"),
    ("Bot share (%)", f"{100*d17['bot_share']:.1f}", f"{100*d15['bot_share']:.1f}"),
    ("Human groups", "genuine accounts", "E13, TFP"),
    ("Bot groups", "social spambots 1-3, traditional spambots 1-4, fake followers", "FSF, INT, TWT"),
    ("Empty profile description, bots / humans (%)",
     f"{100*d17['empty_bio']['empty_text_rate_bot']:.1f} / {100*d17['empty_bio']['empty_text_rate_human']:.1f}",
     f"{100*d15['empty_bio']['empty_text_rate_bot']:.1f} / {100*d15['empty_bio']['empty_text_rate_human']:.1f}"),
    ("Graph", "none released; kNN (k = 10, cosine) on metadata", "follower + friend relations"),
    ("Directed edges among labelled accounts", "—", f"{d15['directed_edges_among_labelled']:,}"),
    ("Isolated accounts (all / test, %)", "0 / 0.0", f"{d15['isolated_nodes']:,} / {100*first('cresci2015','isolated_share_test'):.1f}"),
    ("Test accounts without text (%)", f"{100*first('cresci2017','text_missing_share_test'):.1f}",
     f"{100*first('cresci2015','text_missing_share_test'):.1f}"),
    ("Edge homophily, training nodes", f"{first('cresci2017','edge_homophily_train'):.3f}",
     f"{first('cresci2015','edge_homophily_train'):.3f}"),
]
emit(1, "dataset_statistics", pd.DataFrame(rows, columns=["Property", "Cresci-2017", "Cresci-2015"]),
     "Dataset statistics",
     ["Counts from the run's data audit (data_audit.json). Cresci-2017 group traditional spambots 3 has 403 of the "
      "433 published accounts. The Cresci-2017 kNN graph is built from the same metadata the models use."], wide=True,
     spec="p{0.32\\textwidth}p{0.33\\textwidth}p{0.25\\textwidth}")

# ---------------------------------------------------------------- T2 preprocessing and splits
sp = tr.groupby("dataset")[["n_train", "n_val", "n_test"]].first()
rows = [
    ("Split", "stratified 70 / 10 / 20 per seed; all methods share the split of a seed"),
    ("Seeds", "0-4 (clean, inductive, ablation); 0-2 (robustness, cross-dataset, explanation)"),
    ("Train / validation / test, Cresci-2017", " / ".join(f"{int(v):,}" for v in sp.loc["cresci2017"])),
    ("Train / validation / test, Cresci-2015", " / ".join(f"{int(v):,}" for v in sp.loc["cresci2015"])),
    ("Metadata features (d = 4)", "account age (days), followers/(friends+1), statuses/(age+1), screen-name entropy (bits)"),
    ("Metadata transform", "±inf → missing; training-median imputation; sign(x)·log(1+|x|); training z-score"),
    ("Text features (d = 384)", "profile description; all-MiniLM-L6-v2, frozen; empty → zero vector, availability 0"),
    ("Graph, transductive", "test accounts in the graph, labels hidden"),
    ("Graph, inductive", "test accounts absent from training graph; no test-to-seen messages"),
    ("Label leakage controls", "test labels stored as -1 and asserted absent from fit/validation sets; scaler fitted on training rows"),
    ("Model selection", "validation loss (neural), validation log-loss early stopping (trees); no hyper-parameter search"),
]
emit(2, "preprocessing_splits", pd.DataFrame(rows, columns=["Item", "Setting"]),
     "Preprocessing and split configuration",
     ["Transcribed from botrecon/features.py, splits.py, suite.py and the run's protocol audit (39 of 40 checks pass; "
      "the failure is the Cresci-2017 group-size check)."], wide=True,
     spec="p{0.27\\textwidth}p{0.66\\textwidth}")

# ---------------------------------------------------------------- T3 architectures
rows = [
    ("LightGBM", "meta+text", "gbdt, 31 leaves, lr 0.05, ≤ 1000 trees, early stop 50 (val log-loss)"),
    ("XGBoost", "meta+text", "≤ 1000 trees, lr 0.05, depth 6, subsample 0.9, colsample 0.9, hist, early stop 50"),
    ("Random forest", "meta+text", "500 trees, scikit-learn defaults, no class weights"),
    ("GCN", "meta+text, graph", "2 layers d→64→2, ReLU, dropout 0.5, sym.-normalised A + I"),
    ("GraphSAGE", "meta+text, graph", "2 layers, mean aggregator, 64 hidden, ReLU, dropout 0.5"),
    ("GAT", "meta+text, graph", "4 heads × 16, ELU; output 1 head; dropout 0.5; self-loops"),
    ("R-GCN (BotRGCN-style)", "meta+text, 2 relations", "proj. 64, 2 relational layers, MLP 64→64→2, dropout 0.3 (Cresci-2015 only)"),
    ("Hybrid P1 / P3", "meta+text, graph", "GCN (half A of train, 100 epochs) → LightGBM (half B); P1 = [meta, text, GCN], P3 = GCN only"),
    ("Neural training", "—", "full batch, AdamW, wd 5×10⁻⁴, lr 0.01 (GAT 0.005), ≤ 200 epochs, patience 30"),
    ("RGF (proposed)", "meta, text, graph", "experts h = 64, dropout 0.3; gate 6→32→3; λ = 0.5; modality dropout 0.3; "
     "AdamW lr 0.005, wd 5×10⁻⁴, ≤ 300 epochs, patience 40; temperature grid 60 values in [0.25, 5]"),
]
emit(3, "architectures_hyperparameters", pd.DataFrame(rows, columns=["Model", "Inputs", "Configuration"]),
     "Model architectures and hyper-parameters",
     ["All values are fixed in code (methods.py, nn.py); none was tuned. The hybrid reported in the main tables is "
      "P1 with the GCN's 64-d hidden layer, the best of four variants by mean validation F1 on both datasets."], wide=True,
     spec="p{0.16\\textwidth}p{0.17\\textwidth}p{0.6\\textwidth}")

# ---------------------------------------------------------------- significance (replicates T10/T10b of the run)
def holm(p):
    items = sorted(p.items(), key=lambda kv: (1.0 if kv[1] != kv[1] else kv[1]))
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        v = 1.0 if v != v else v
        run = min(1.0, max(run, (m - i) * v)); out[k] = run
    return out


SIG = {}
for ds in DATASETS:
    d = tr[tr.dataset == ds]
    rg = d[d.method == "RGF (proposed)"].set_index("seed")
    for metric in ("test_f1_bot", "test_mcc"):
        ps, diffs = {}, {}
        for m in d.method.unique():
            if m == "RGF (proposed)":
                continue
            o = d[d.method == m].set_index("seed").loc[rg.index]
            a, b = rg[metric].values, o[metric].values
            ps[m] = st.ttest_rel(a, b).pvalue if np.std(a - b) > 0 else 1.0
            diffs[m] = 100 * (a - b).mean()
        for m, ph in holm(ps).items():
            SIG[(ds, metric, m)] = (diffs[m], ps[m], ph)
sig_rows = [{"dataset": ds, "metric": mt, "method": m, "mean_diff_RGF_minus_method_pp": v[0], "p": v[1], "p_holm": v[2]}
            for (ds, mt, m), v in SIG.items()]
pd.DataFrame(sig_rows).to_csv(TAB / "table04_significance_detail.csv", index=False)
# check against the run's own table: XGBoost on Cresci-2017, F1: diff -1.11, Holm p 0.0044
_x = SIG[("cresci2017", "test_f1_bot", "XGBoost (meta+text)")]
assert abs(_x[0] + 1.11) < 0.01 and abs(_x[2] - 0.0044) < 0.0006, _x


def mark(ds, metric, m):
    if m == "RGF (proposed)":
        return ""
    diff, _, ph = SIG[(ds, metric, m)]
    return "" if ph >= 0.05 else ("†" if diff < 0 else "‡")


# ---------------------------------------------------------------- T4 clean performance
cols = [("test_accuracy", "Acc"), ("test_precision_bot", "Prec"), ("test_recall_bot", "Rec"), ("test_f1_bot", "F1"),
        ("test_mcc", "MCC"), ("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC")]
t = tr[tr.method.isin(MAIN)].copy()
for c, _ in cols:
    t[c] = 100 * t[c]
a = agg(t, ["dataset", "method"], [c for c, _ in cols])
rows = []
for ds in DATASETS:
    rows.append([f"§{DATASETS[ds]} (5 seeds)"] + [""] * len(cols))
    for m in MAIN:
        r = a[(a.dataset == ds) & (a.method == m)]
        if r.empty:
            continue
        r = r.iloc[0]
        cells = [f(r[f"{c}_mean"], r[f"{c}_std"]) for c, _ in cols]
        cells[3] += mark(ds, "test_f1_bot", m); cells[4] += mark(ds, "test_mcc", m)
        rows.append([SHORT[m]] + cells)
emit(4, "clean_performance", pd.DataFrame(rows, columns=["Method"] + [n for _, n in cols]),
     "Clean test performance (%, mean ± SD over 5 seeds, transductive)",
     ["Bot-class precision, recall and F1. † RGF is significantly lower than this method, ‡ significantly higher "
      "(paired t-test over seeds, Holm correction over all comparisons with RGF on that dataset and metric, α = 0.05; "
      "with 5 seeds these tests are indicative). Under the Nadeau-Bengio corrected resampled t-test, RGF vs GraphSAGE on Cresci-2017 "
      "is no longer significant (audit_stats/clean_tests.csv). Full test results: table04_significance_detail.csv."], wide=True)

# ---------------------------------------------------------------- T5 ablations
t = tr[tr.method.isin(ABLATIONS)].copy()
for c in ("val_f1_bot", "test_f1_bot", "test_mcc", "test_roc_auc"):
    t[c] = 100 * t[c]
a = agg(t, ["dataset", "method"], ["val_f1_bot", "test_f1_bot", "test_mcc", "test_roc_auc"])
rows = []
for m in ABLATIONS:
    row = [SHORT[m] if m != "RGF (proposed)" else "RGF (proposed)"]
    for ds in DATASETS:
        r = a[(a.dataset == ds) & (a.method == m)].iloc[0]
        base = a[(a.dataset == ds) & (a.method == "RGF (proposed)")].iloc[0]
        row += [f(r.test_f1_bot_mean, r.test_f1_bot_std), f(r.test_mcc_mean, r.test_mcc_std),
                "—" if m == "RGF (proposed)" else f"{r.test_mcc_mean - base.test_mcc_mean:+.2f}{mark(ds, 'test_mcc', m)}"]
    rows.append(row)
emit(5, "ablation", pd.DataFrame(rows, columns=["Variant", "F1 C-17", "MCC C-17", "ΔMCC C-17", "F1 C-15", "MCC C-15", "ΔMCC C-15"]),
     "Ablation of RGF (%, mean ± SD over 5 seeds)",
     ["ΔMCC = variant − RGF in percentage points; † / ‡ as in Table 4 (here † = variant significantly better than RGF). "
      "Concat fusion also removes the auxiliary losses, so it changes two components."], wide=True)

# ---------------------------------------------------------------- robustness tables 6, 7, 8
def robust_table(pert, methods, extra_methods=()):
    r = rob[(rob.perturbation == pert) & rob.method.isin(list(methods) + list(extra_methods))].copy()
    r["f1"] = 100 * r.f1_bot
    a = agg(r, ["dataset", "method", "level"], ["f1"])
    levels = sorted(a.level.unique())
    rows = []
    for ds in DATASETS:
        rows.append([f"§{DATASETS[ds]} (3 seeds)"] + [""] * (len(levels) + 1))
        for m in list(methods) + list(extra_methods):
            g = a[(a.dataset == ds) & (a.method == m)].set_index("level")
            if g.empty:
                continue
            vals = [f(g.loc[l, "f1_mean"], g.loc[l, "f1_std"]) for l in levels]
            drop = g.loc[levels[-1], "f1_mean"] - g.loc[levels[0], "f1_mean"]
            rows.append([SHORT[m]] + vals + [f"{drop:+.2f}"])
    cols = ["Method"] + [f"ρ = {int(100*l)}%" for l in levels] + ["Δ (100% − 0%)"]
    return pd.DataFrame(rows, columns=cols)


ABL3 = ["RGF concat fusion", "RGF w/o gate (mean of experts)", "RGF w/o modality dropout"]
emit(6, "missing_text", robust_table("missing_text", ROBUST, ABL3),
     "Bot-class F1 (%) when the profile text of a fraction ρ of test accounts is removed",
     ["Mean ± SD over seeds 0-2; models trained on clean data. Δ in percentage points. ρ = 0 differs slightly from "
      "Table 4, which averages seeds 0-4."], wide=True)
t7 = []
for pert, lab in (("drop_edges", "edge deletion"), ("noisy_edges", "edge injection")):
    x = robust_table(pert, ROBUST)
    x.insert(1, "Perturbation", lab)
    t7.append(x)
t7 = pd.concat(t7, ignore_index=True).fillna("—")
t7.loc[t7.Method.str.startswith("§"), "Perturbation"] = ""
emit(7, "graph_perturbation", t7,
     "Bot-class F1 (%) under graph perturbation of test accounts",
     ["Edge deletion removes each edge incident to a test account with probability ρ; edge injection adds "
      "round(ρ × degree) random edges per test account (ρ ∈ {0, 25, 50, 100}%). LightGBM uses no graph and is "
      "unaffected. Cresci-2017 edges come from the metadata kNN graph. Mean ± SD over seeds 0-2."], wide=True,
     size=r"\scriptsize")
t8 = robust_table("metadata_mimicry", ROBUST, ABL3)
r = rob[(rob.perturbation == "metadata_mimicry") & (rob.level == 1.0)].copy(); r["mcc"] = 100 * r.mcc
mm = agg(r, ["dataset", "method"], ["mcc"])
mcc_col = []
cur = None
for m in t8.Method:
    if m.startswith("§"):
        cur = [k for k, v in DATASETS.items() if v in m][0]; mcc_col.append(""); continue
    full = [k for k, v in SHORT.items() if v == m][0]
    g = mm[(mm.dataset == cur) & (mm.method == full)].iloc[0]
    mcc_col.append(f(g.mcc_mean, g.mcc_std))
t8["MCC at ρ = 100%"] = mcc_col
emit(8, "metadata_mimicry", t8,
     "Bot-class F1 (%) when a fraction ρ of test bots receive the median metadata of training humans",
     ["The attacker knows which test accounts are bots and the human median (white-box on the feature, not the "
      "model). The graph is not rebuilt; on Cresci-2017 this leaves kNN neighbours computed from the original "
      "metadata. Mean ± SD over seeds 0-2."], wide=True, size=r"\scriptsize")

# ---------------------------------------------------------------- T9 inductive
ind = main[main.method.isin(MAIN)].copy()
ind["f1"] = 100 * ind.test_f1_bot; ind["mcc"] = 100 * ind.test_mcc
a = agg(ind, ["dataset", "setting", "method"], ["f1", "mcc"])
rows = []
for ds in DATASETS:
    rows.append([f"§{DATASETS[ds]} (5 seeds)"] + [""] * 5)
    for m in MAIN:
        g = a[(a.dataset == ds) & (a.method == m)].set_index("setting")
        if "inductive" not in g.index:
            continue
        tr_, in_ = g.loc["transductive"], g.loc["inductive"]
        rows.append([SHORT[m], f(tr_.f1_mean, tr_.f1_std), f(in_.f1_mean, in_.f1_std),
                     f(tr_.mcc_mean, tr_.mcc_std), f(in_.mcc_mean, in_.mcc_std), f"{in_.mcc_mean - tr_.mcc_mean:+.2f}"])
emit(9, "inductive", pd.DataFrame(rows, columns=["Method", "F1 trans.", "F1 induct.", "MCC trans.", "MCC induct.", "ΔMCC"]),
     "Transductive versus inductive evaluation (%, mean ± SD over 5 seeds)",
     ["Inductive: test accounts are absent from the training graph and send no messages to seen accounts. "
      "Tree models do not use the graph, so both columns are identical. R-GCN, XGBoost and random forest were not "
      "run inductively."], wide=True)

# ---------------------------------------------------------------- T10 cross-dataset
c = cross.copy()
c["spec"] = 100 * c.tn / (c.tn + c.fp); c["fpr"] = 100 - c.spec
for k in ("accuracy", "f1_bot", "mcc", "roc_auc"):
    c[k] = 100 * c[k]
a = agg(c, ["source", "method"], ["accuracy", "spec", "fpr", "f1_bot", "mcc", "roc_auc"])
order = ["LightGBM (meta+text)", "GCN", "GraphSAGE", "Reconstructed hybrid (P1, hidden)", "Reconstructed hybrid (P1, prob)",
         "Reconstructed hybrid (P3, hidden)", "Reconstructed hybrid (P3, prob)", "RGF (proposed)"]
rows = []
for src, tgt in (("cresci2015", "cresci2017"), ("cresci2017", "cresci2015")):
    s = cross[cross.source == src].iloc[0]
    rows.append([f"§Train {DATASETS[src]} → test {DATASETS[tgt]} ({int(s.target_bots)} bots, {int(s.target_humans)} humans, "
                 f"{int(s.overlap_removed)} shared accounts removed)"] + [""] * 5)
    for m in order:
        r = a[(a.source == src) & (a.method == m)].iloc[0]
        nobots = int(s.target_bots) == 0
        rows.append([SHORT[m], f(r.accuracy_mean, r.accuracy_std), f(r.fpr_mean, r.fpr_std),
                     "—" if nobots else f(r.f1_bot_mean, r.f1_bot_std), "—" if nobots else f(r.mcc_mean, r.mcc_std),
                     "—" if nobots else f(r.roc_auc_mean, r.roc_auc_std)])
emit(10, "cross_dataset", pd.DataFrame(rows, columns=["Method", "Accuracy", "Human FPR", "F1", "MCC", "ROC-AUC"]),
     "Cross-dataset evaluation (%, mean ± SD over seeds 0-2)",
     ["Human FPR = humans classified as bots / all humans. Removing accounts shared by the two datasets (by user ID) "
      "removes every Cresci-2015 test bot, so the Cresci-2017 → Cresci-2015 direction measures only false positives "
      "on 390 humans; F1, MCC and ROC-AUC are undefined there."], wide=True)

# ---------------------------------------------------------------- T11 calibration, T12 selective
t = tr[tr.method.isin(MAIN)].copy()
for k in ("test_ece", "test_uncal_ece", "test_risk_at_80", "test_risk_at_90", "test_accuracy"):
    t[k] = 100 * t[k]
t["err_full"] = 100 - t.test_accuracy
a = agg(t, ["dataset", "method"], ["test_ece", "test_uncal_ece", "test_brier", "test_uncal_brier", "temperature",
                                    "test_risk_at_80", "test_risk_at_90", "err_full"])
rows, rows12 = [], []
for m in MAIN:
    row, row12 = [SHORT[m]], [SHORT[m]]
    for ds in DATASETS:
        r = a[(a.dataset == ds) & (a.method == m)]
        if r.empty:
            row += ["—"] * 2; row12 += ["—"] * 3; continue
        r = r.iloc[0]
        row += [f(r.test_ece_mean, r.test_ece_std), f(r.test_brier_mean, r.test_brier_std, 4)]
        row12 += [f(r.err_full_mean, r.err_full_std), f(r.test_risk_at_90_mean, r.test_risk_at_90_std),
                  f(r.test_risk_at_80_mean, r.test_risk_at_80_std)]
    rows.append(row); rows12.append(row12)
r17, r15 = [a[(a.dataset == ds) & (a.method == "RGF (proposed)")].iloc[0] for ds in DATASETS]
rows.append(["RGF before temperature scaling"] + sum([[f(r.test_uncal_ece_mean, r.test_uncal_ece_std),
                                                       f(r.test_uncal_brier_mean, r.test_uncal_brier_std, 4)] for r in (r17, r15)], []))
rows.append(["RGF temperature T"] + sum([[f(r.temperature_mean, r.temperature_std), ""] for r in (r17, r15)], []))
emit(11, "calibration", pd.DataFrame(rows, columns=["Method", "ECE C-17 (%)", "Brier C-17", "ECE C-15 (%)", "Brier C-15"]),
     "Calibration on the test split (mean ± SD over 5 seeds)",
     ["ECE: 15 equal-width bins of predicted-class confidence. Only RGF applies temperature scaling, with T chosen on the "
      "same validation split used for early stopping; on both datasets ECE and Brier score are slightly higher after scaling, a change within seed variation (paired t, p = 0.38 and 0.27)."], wide=True)
emit(12, "selective_prediction",
     pd.DataFrame(rows12, columns=["Method", "Err 100% C-17", "Err 90% C-17", "Err 80% C-17",
                                   "Err 100% C-15", "Err 90% C-15", "Err 80% C-15"]),
     "Selective prediction: error (%) on the most confident fraction of test accounts",
     ["Accounts ranked by |p − 0.5|; Err 100% = 100 − accuracy. Temperature scaling does not change this ranking. "
      "Mean ± SD over 5 seeds."], wide=True)

# ---------------------------------------------------------------- T13 gate weights
gw = expl[expl.kind == "gate_weight"]
ea = expl[expl.kind == "expert_alone"]
rows = []
labels = {"all": "All test accounts", "bots": "Bots", "humans": "Humans", "text missing": "Text missing",
          "isolated (no graph)": "No graph link"}
for ds in DATASETS:
    rows.append([f"§{DATASETS[ds]} — mean gate weight (seeds 0-2)"] + [""] * 4)
    for g in ("all", "bots", "humans", "text missing", "isolated (no graph)"):
        x = gw[(gw.dataset == ds) & (gw.group == g)]
        if x.empty:
            continue
        rows.append([labels[g], f"{x.n.mean():.0f}", f(x.w_meta.mean(), x.w_meta.std(), 3),
                     f(x.w_text.mean(), x.w_text.std(), 3), f(x.w_graph.mean(), x.w_graph.std(), 3)])
    rows.append([f"§{DATASETS[ds]} — each expert alone, accounts where its modality exists: F1 / MCC (%)"] + [""] * 4)
    vals = {}
    for k in ("metadata", "text", "graph"):
        x = ea[(ea.dataset == ds) & (ea.group == k)]
        vals[k] = f"{100*x.f1_bot.mean():.2f} / {100*x.mcc.mean():.2f} (n = {x.n.mean():.0f})"
    rows.append(["Expert alone", "", vals["metadata"], vals["text"], vals["graph"]])
emit(13, "gate_weights", pd.DataFrame(rows, columns=["Group", "n", "w meta", "w text", "w graph"]),
     "Reliability-gate weights of RGF and accuracy of each expert alone",
     ["Weights are averaged over accounts, then over seeds. Unavailable experts receive weight 0 by construction "
      "(text-missing and isolated rows). Gate weights describe what the model relies on, not which modality is correct."],
     wide=True)

# ---------------------------------------------------------------- T14 cost
a = agg(tr[tr.method.isin(MAIN)], ["dataset", "method"], ["fit_seconds", "predict_ms_per_1k", "size"])
trees = {"LightGBM (meta+text)", "XGBoost (meta+text)", "Random Forest (meta+text)"}
rows = []
for m in MAIN:
    row = [SHORT[m]]
    for ds in DATASETS:
        r = a[(a.dataset == ds) & (a.method == m)]
        if r.empty:
            row += ["—"] * 3; continue
        r = r.iloc[0]
        size = f"{r.size_mean:,.1f} trees" if m in trees else f"{int(r.size_mean):,}"
        if m == "RGF (proposed)":
            size = "38,089*"
        row += [f(r.fit_seconds_mean, r.fit_seconds_std, 1), f(r.predict_ms_per_1k_mean, r.predict_ms_per_1k_std, 1), size]
    rows.append(row)
emit(14, "computational_cost", pd.DataFrame(rows, columns=["Method", "Fit (s) C-17", "Infer. (ms/1k) C-17", "Size C-17",
                                                           "Fit (s) C-15", "Infer. (ms/1k) C-15", "Size C-15"]),
     "Computational cost on CPU (mean ± SD over 5 seeds, transductive)",
     ["Kaggle CPU session; fit time includes early stopping; inference in milliseconds per 1,000 accounts. Size = trainable "
      "parameters for neural models and number of trees for tree models (not comparable units). *Parameters used by RGF; "
      "the checkpoint registers 50,571 because it also holds a 12,482-parameter head used only by the concatenation ablation."], wide=True, size=r"\scriptsize")

# ---------------------------------------------------------------- T15 related work (qualitative, verified)
rows = [
    ("BotRGCN [Feng et al., 2021]", "metadata, text, follow graph", "relational GCN", "TwiBot-20 acc 0.8462, F1 0.8707 (reported)",
     "R-GCN baseline re-implemented here"),
    ("RGT [Feng et al., 2022]", "metadata, text, heterogeneous graph", "relation-aware graph transformer", "outperforms prior methods on TwiBot-20",
     "not re-run"),
    ("BotMoE [Liu et al., 2023]", "metadata, text, graph", "community-aware modality experts", "most robust to feature manipulation in its study",
     "closest architecture; not re-run"),
    ("BotBuster [Ng & Carley, 2023]", "per-platform account fields", "mixture of experts", "handles incomplete account information",
     "missing-input motivation"),
    ("BotCHF [Wang et al., 2026]", "multi-modal", "camouflage-aware fusion", "fusion weights differ across datasets",
     "per-account weighting is not new"),
    ("LMBot [Cai et al., 2024]", "text, graph (distilled)", "graph-to-LM distillation", "graph-less deployment at state-of-the-art accuracy",
     "alternative to graph availability"),
    ("Hays et al., 2023", "simple features", "shallow decision trees", "near state-of-the-art; datasets do not generalise",
     "consistent with our strong tree baselines"),
    ("Echeverría et al., 2018", "account features", "leave-one-botnet-out", ">97% accuracy does not transfer to unseen bot classes",
     "consistent with our cross-dataset result"),
    ("Mukherjee et al., 2026", "graph", "optimal-transport edge attacks", "attacks bypass five defences",
     "edge injection is a real threat"),
    ("This work", "metadata, text, graph", "availability-masked gated experts", "see Tables 4-14",
     "same protocol for all baselines"),
]
emit(15, "related_work", pd.DataFrame(rows, columns=["Study", "Modalities", "Approach", "Reported finding", "Relation to this work"]),
     "Positioning against related work",
     ["Findings are summarised from the verified reference library (VERIFIED_REFERENCES.csv); numbers from other papers "
      "use different datasets, splits and protocols and are not comparable with Tables 4-14, so no numeric ranking is given."],
     wide=True, size=r"\scriptsize", spec="p{0.17\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}p{0.24\\textwidth}p{0.18\\textwidth}")

(TAB / "ALL_TABLES.md").write_text("# Tables 1-15\n\n" + "\n".join(MD), encoding="utf-8")
(TAB / "all_tables.tex").write_text("\n\n".join(TEX), encoding="utf-8")
print("tables written:", len(TEX))
