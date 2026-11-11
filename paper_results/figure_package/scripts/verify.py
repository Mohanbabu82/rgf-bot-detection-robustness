"""Cross-check: every number plotted or tabulated here must equal the run's own report
(data/raw/tables.md, written by botrecon/suite_report.py on Kaggle). Exits non-zero on any mismatch.
Run:  python verify.py
"""
import re
import sys
import pandas as pd
from common import RAW, DER, TAB, DATASETS

md = (RAW / "tables.md").read_text(encoding="utf-8")
NAME = {"Cresci-2017": "cresci2017", "Cresci-2015": "cresci2015"}


def run_table(title, ds=None):
    head = f"### {title} ({ds})" if ds else f"## {title}"
    i = md.index(head)
    lines = md[i:].splitlines()[2:]
    rows = []
    for ln in lines:
        if not ln.startswith("|"):
            if rows:
                break
            continue
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return pd.DataFrame(rows[2:], columns=rows[0])


def num(s):
    return float(re.match(r"-?[\d.]+", s).group())


fails, checks = [], 0


def check(label, ours, theirs, tol):
    global checks
    checks += 1
    if abs(ours - theirs) > tol:
        fails.append(f"{label}: ours {ours:.4f} vs run {theirs:.4f}")


# Fig. 5 / Table 4: clean metrics
f5 = pd.read_csv(DER / "fig05_clean_performance.csv")
for name, ds in NAME.items():
    t = run_table("T2 Main comparison", name)
    for _, r in t.iterrows():
        m = r["Method"].replace(" [best of 4 by validation]", "")
        g = f5[(f5.dataset == ds) & (f5.method == m)]
        if g.empty:
            continue
        for col, ours in (("F1", "test_f1_bot_mean"), ("MCC", "test_mcc_mean"), ("ROC-AUC", "test_roc_auc_mean")):
            check(f"clean {name} {m} {col}", g[ours].iloc[0], num(r[col]), 0.006)

# Fig. 10: ablation clean
f10 = pd.read_csv(DER / "fig10_ablation_clean.csv")
for name, ds in NAME.items():
    for _, r in run_table("T3 Ablation of the proposed model", name).iterrows():
        g = f10[(f10.dataset == ds) & (f10.method == r["Variant"])].iloc[0]
        check(f"ablation {name} {r['Variant']} F1", g.f1_mean, num(r["F1"]), 0.006)
        check(f"ablation {name} {r['Variant']} MCC", g.mcc_mean, num(r["MCC"]), 0.006)

# Figs. 6-8: robustness (run table rounds to 2 decimals)
rob = pd.concat([pd.read_csv(DER / p).assign(perturbation=k) for p, k in
                 (("fig06_mimicry_f1.csv", "metadata_mimicry"), ("fig07_missing_text_f1.csv", "missing_text"))]
                + [pd.read_csv(DER / "fig08_graph_perturbation_f1.csv")])
for name, ds in NAME.items():
    t = run_table("T4 Robustness, bot-class F1 (%)", name)
    for _, r in t.iterrows():
        for m in t.columns[2:]:
            g = rob[(rob.dataset == ds) & (rob.method == m) & (rob.perturbation == r["perturbation"]) &
                    (rob.level == float(r["level"]))]
            if not g.empty:
                check(f"robust {name} {m} {r['perturbation']} {r['level']}", g.f1_mean.iloc[0], float(r[m]), 0.006)

# Fig. 9: cross-dataset (specificity -> human FPR)
f9 = pd.read_csv(DER / "fig09_cross_dataset.csv")
for _, r in run_table("T7 Cross-dataset generalisation (train on one, test on the other)").iterrows():
    g = f9[(f9.source == NAME[r["Train on"]]) & (f9.method == r["Method"])]
    if not g.empty:
        check(f"cross {r['Train on']} {r['Method']} FPR", g.fpr_human_mean.iloc[0], 100 - num(r["Specificity"]), 0.006)
        if r["F1"].startswith("—") is False:
            check(f"cross {r['Train on']} {r['Method']} F1", g.f1_bot_mean.iloc[0], num(r["F1"]), 0.006)

# Fig. 11: calibration, selective prediction
f11 = pd.read_csv(DER / "fig11_calibration_selective.csv")
for name, ds in NAME.items():
    for _, r in run_table("T6 Calibration and selective prediction", name).iterrows():
        g = f11[(f11.dataset == ds) & (f11.method == r["Method"])]
        if g.empty:
            continue
        g = g.iloc[0]
        check(f"ECE {name} {r['Method']}", g.test_ece_mean, num(r["ECE"]), 0.006)
        check(f"Brier {name} {r['Method']}", g.test_brier_mean, num(r["Brier"]), 0.00006)
        check(f"err80 {name} {r['Method']}", g.test_risk_at_80_mean, num(r["Error @80% coverage"]), 0.006)
        check(f"err90 {name} {r['Method']}", g.test_risk_at_90_mean, num(r["Error @90% coverage"]), 0.006)

# Fig. 12: cost
f12 = pd.read_csv(DER / "fig12_cost.csv")
for name, ds in NAME.items():
    for _, r in run_table("T9 Efficiency on CPU", name).iterrows():
        g = f12[(f12.dataset == ds) & (f12.method == r["Method"])]
        if g.empty:
            continue
        g = g.iloc[0]
        check(f"fit {name} {r['Method']}", g.fit_seconds_mean, num(r["Fit time (s)"]), 0.051)
        check(f"infer {name} {r['Method']}", g.predict_ms_per_1k_mean, num(r["Inference (ms / 1k accounts)"]), 0.051)

# Table 13: gate weights
t13 = pd.read_csv(TAB / "table13_gate_weights.csv")
for name in NAME:
    for _, r in run_table("T8a Mean gate weights of RGF by account group", name).iterrows():
        checks_label = {"all": "All test accounts", "bots": "Bots", "humans": "Humans", "text missing": "Text missing",
                        "isolated (no graph)": "No graph link"}[r["group"]]
        block = t13.iloc[t13.index[t13.Group.str.contains(f"§{name} — mean")][0]:]
        ours = block[block.Group == checks_label].iloc[0]
        for col, c2 in (("w_meta", "w meta"), ("w_text", "w text"), ("w_graph", "w graph")):
            check(f"gate {name} {r['group']} {col}", num(ours[c2]), float(r[col]), 0.0006)

print(f"{checks} checks, {len(fails)} mismatches")
for x in fails:
    print("  MISMATCH", x)
sys.exit(1 if fails else 0)
