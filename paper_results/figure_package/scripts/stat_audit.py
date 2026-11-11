"""Independent statistical audit of the result files. Writes CSVs to ../audit_stats/ and prints a summary.
Run:  python stat_audit.py
Tests: paired t over seeds (as in the paper) and the Nadeau-Bengio corrected resampled t-test, which accounts for
overlap between the random train/test splits of different seeds (Nadeau & Bengio, Mach. Learn. 52:239-281, 2003).
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats as st
from common import ROOT, load, MAIN, ROBUST, ABLATIONS, SHORT, DATASETS

OUT = ROOT / "audit_stats"; OUT.mkdir(exist_ok=True)
main, rob, cross, expl = load("main"), load("robust"), load("cross"), load("explain")
tr = main[main.setting == "transductive"]
RATIO = {ds: tr[tr.dataset == ds].n_test.iloc[0] / tr[tr.dataset == ds].n_train.iloc[0] for ds in DATASETS}


def holm(p):
    k = sorted(p, key=lambda x: p[x]); m = len(k); out = {}; run = 0
    for i, key in enumerate(k):
        run = min(1, max(run, (m - i) * p[key])); out[key] = run
    return out


def tests(a, b, ratio):
    d = np.asarray(a) - np.asarray(b); n = len(d); sd = d.std(ddof=1)
    if sd == 0:
        return dict(diff=d.mean(), t=np.nan, p=1.0, p_nb=1.0, dz=np.nan, n=n, wins=int((d > 0).sum()))
    t = d.mean() / (sd / np.sqrt(n)); p = 2 * st.t.sf(abs(t), n - 1)
    t_nb = d.mean() / np.sqrt((1 / n + ratio) * sd ** 2); p_nb = 2 * st.t.sf(abs(t_nb), n - 1)
    return dict(diff=d.mean(), t=t, p=p, p_nb=p_nb, dz=d.mean() / sd, n=n, wins=int((d > 0).sum()),
                ci_lo=d.mean() - st.t.ppf(.975, n - 1) * sd / np.sqrt(n), ci_hi=d.mean() + st.t.ppf(.975, n - 1) * sd / np.sqrt(n))


# 1. clean: RGF vs every method (run's family) and all pairs among MAIN -------------------------------
rows = []
for ds in DATASETS:
    d = tr[tr.dataset == ds]
    for metric in ("test_f1_bot", "test_mcc"):
        piv = d.pivot(index="seed", columns="method", values=metric) * 100
        res = {m: tests(piv["RGF (proposed)"], piv[m], RATIO[ds]) for m in piv.columns if m != "RGF (proposed)"}
        hp = holm({m: r["p"] for m, r in res.items()}); hn = holm({m: r["p_nb"] for m, r in res.items()})
        for m, r in res.items():
            rows.append(dict(dataset=ds, metric=metric, family="RGF vs all (run)", a="RGF (proposed)", b=m, **r,
                             p_holm=hp[m], p_nb_holm=hn[m]))
        meths = [m for m in MAIN if m in piv.columns]
        res = {(a, b): tests(piv[a], piv[b], RATIO[ds]) for a, b in combinations(meths, 2)}
        hp = holm({k: r["p"] for k, r in res.items()}); hn = holm({k: r["p_nb"] for k, r in res.items()})
        for (a, b), r in res.items():
            rows.append(dict(dataset=ds, metric=metric, family="all pairs MAIN", a=a, b=b, **r, p_holm=hp[(a, b)],
                             p_nb_holm=hn[(a, b)]))
clean = pd.DataFrame(rows); clean.to_csv(OUT / "clean_tests.csv", index=False)

# per-seed ranks among MAIN
rk = []
for ds in DATASETS:
    piv = tr[(tr.dataset == ds) & tr.method.isin(MAIN)].pivot(index="seed", columns="method", values="test_f1_bot")
    r = piv.rank(axis=1, ascending=False)
    for m in piv.columns:
        rk.append(dict(dataset=ds, method=m, mean_rank=r[m].mean(), min_rank=r[m].min(), max_rank=r[m].max()))
    fr = st.friedmanchisquare(*[piv[c] for c in piv.columns])
    rk.append(dict(dataset=ds, method="Friedman test", mean_rank=fr.statistic, min_rank=fr.pvalue, max_rank=np.nan))
pd.DataFrame(rk).to_csv(OUT / "clean_ranks.csv", index=False)

# 2. robustness: RGF vs others at full strength and on the mean over levels (area under degradation) -------
rrows = []
for ds in DATASETS:
    for pert in rob.perturbation.unique():
        d = rob[(rob.dataset == ds) & (rob.perturbation == pert)]
        full = d[d.level == 1.0].pivot(index="seed", columns="method", values="f1_bot") * 100
        area = d.groupby(["seed", "method"]).f1_bot.mean().unstack() * 100
        drop = (d[d.level == 0.0].pivot(index="seed", columns="method", values="f1_bot") * 100) - full
        for name, piv in (("F1 at rho=1", full), ("mean F1 over levels", area), ("F1 loss 0->1", drop)):
            res = {m: tests(piv["RGF (proposed)"], piv[m], RATIO[ds]) for m in piv.columns if m != "RGF (proposed)"}
            hp = holm({m: r["p"] for m, r in res.items()})
            for m, r in res.items():
                rrows.append(dict(dataset=ds, perturbation=pert, quantity=name, other=m, rgf=piv["RGF (proposed)"].mean(),
                                  other_mean=piv[m].mean(), **r, p_holm=hp[m]))
robt = pd.DataFrame(rrows); robt.to_csv(OUT / "robust_tests.csv", index=False)

# 3. rank agreement clean (5-seed) vs full mimicry (3-seed), all methods in the robustness file -------------
cor = []
for ds in DATASETS:
    meths = sorted(set(rob[rob.dataset == ds].method))
    c5 = tr[(tr.dataset == ds) & tr.method.isin(meths)].groupby("method").test_f1_bot.mean()
    c3 = rob[(rob.dataset == ds) & (rob.perturbation == "metadata_mimicry") & (rob.level == 0)].groupby("method").f1_bot.mean()
    mm = rob[(rob.dataset == ds) & (rob.perturbation == "metadata_mimicry") & (rob.level == 1)].groupby("method").f1_bot.mean()
    mt = rob[(rob.dataset == ds) & (rob.perturbation == "missing_text") & (rob.level == 1)].groupby("method").f1_bot.mean()
    for lab, x in (("clean5 vs mimicry", c5), ("clean3 vs mimicry", c3)):
        s = st.spearmanr(x.loc[mm.index], mm); k = st.kendalltau(x.loc[mm.index], mm)
        cor.append(dict(dataset=ds, pair=lab, n=len(mm), spearman=s.statistic, p=s.pvalue, kendall=k.statistic, p_k=k.pvalue))
    s = st.spearmanr(c5.loc[mt.index], mt)
    cor.append(dict(dataset=ds, pair="clean5 vs missing text", n=len(mt), spearman=s.statistic, p=s.pvalue))
pd.DataFrame(cor).to_csv(OUT / "rank_correlation.csv", index=False)

# 4. cross-dataset human false-positive rate ------------------------------------------------------------------
c = cross.copy(); c["humans"] = c.tn + c.fp; c["bots"] = c.tp + c.fn; c["fpr"] = 100 * c.fp / c.humans
c["fpr_from_spec"] = 100 - 100 * c.tn / c.humans
chk = c.groupby(["source", "method"]).agg(humans_min=("humans", "min"), humans_max=("humans", "max"),
                                          target_humans=("target_humans", "first"), bots=("bots", "max"),
                                          fpr_mean=("fpr", "mean"), fpr_sd=("fpr", "std"),
                                          fp_sum=("fp", "sum"), n_sum=("humans", "sum")).reset_index()
chk["fpr_pooled"] = 100 * chk.fp_sum / chk.n_sum
chk.to_csv(OUT / "cross_fpr_check.csv", index=False)
# accuracy of a constant 'bot' classifier, for context
# 5. calibration: RGF after vs before temperature; ECE vs other methods -------------------------------------
cal = []
for ds in DATASETS:
    r = tr[(tr.dataset == ds) & (tr.method == "RGF (proposed)")].sort_values("seed")
    for a, b in (("test_ece", "test_uncal_ece"), ("test_brier", "test_uncal_brier")):
        cal.append(dict(dataset=ds, comparison=f"RGF {a} after - before T", **tests(100 * r[a], 100 * r[b], RATIO[ds])))
    piv = tr[(tr.dataset == ds) & tr.method.isin(MAIN)].pivot(index="seed", columns="method", values="test_ece") * 100
    for m in piv.columns:
        if m != "RGF (proposed)":
            cal.append(dict(dataset=ds, comparison=f"ECE RGF - {m}", **tests(piv["RGF (proposed)"], piv[m], RATIO[ds])))
    piv = tr[(tr.dataset == ds) & tr.method.isin(MAIN)].pivot(index="seed", columns="method", values="test_risk_at_90") * 100
    for m in piv.columns:
        if m != "RGF (proposed)":
            cal.append(dict(dataset=ds, comparison=f"Err90 RGF - {m}", **tests(piv["RGF (proposed)"], piv[m], RATIO[ds])))
pd.DataFrame(cal).to_csv(OUT / "calibration_tests.csv", index=False)

# 6. per-seed mimicry values for RGF and key comparators (precision check) -------------------------------------
ps = rob[(rob.perturbation == "metadata_mimicry") & rob.method.isin(
    ["RGF (proposed)", "LightGBM (meta+text)", "GAT", "RGF concat fusion", "RGF w/o gate (mean of experts)",
     "RGF w/o modality dropout", "Reconstructed hybrid (P1, hidden)"])]
ps.pivot_table(index=["dataset", "method", "level"], columns="seed", values="f1_bot").mul(100).round(2).to_csv(OUT / "mimicry_per_seed.csv")
print("done")
