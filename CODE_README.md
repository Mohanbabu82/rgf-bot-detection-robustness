# botrecon: multi-modal fake-account detection on Cresci-2017 and Cresci-2015

**Start here:** `notebooks/reproduce_kaggle_cpu.ipynb` (19 steps, Kaggle CPU). Documents:
`EXPERIMENT_PROTOCOL.md` (datasets, labels, splits, leakage controls, metrics), `ARCHITECTURE.md` (models and
dimensions), `TEST_REPORT.md` (every test and its status), `REPRODUCIBILITY_REPORT.md` (environment, seeds,
commands, limitations), `CODE_AUDIT_REPORT.md` (defects found in the uploaded code and their fixes).
Configuration: `config/kaggle_cpu.json`. Version history: `git log`.

**Scope.** A fresh set of experiments for a paper drafted from scratch. Datasets: **Cresci-2017** (no graph released,
so a kNN graph is built from metadata) and **Cresci-2015** (real follower/friend graph). **TwiBot-20 is not used.**
No number from the earlier manuscript appears in the result tables; every table cell is produced by this code.

## Provenance labels

| Label | Meaning |
| --- | --- |
| `THIS-WORK` | Proposed RGF, its ablations and all baselines. |
| `REPRODUCED-CORRECTED` | The metadata + text + GCN → LightGBM hybrid baseline (from the earlier design), run under the leakage-free protocol. |
| `WALKTHROUGH` | Single-seed notebook walkthrough. Never a result. |
| `SMOKE-TEST-NOT-A-RESULT` | Hashing text encoder / installation check. Excluded from reports. |
| `REPRODUCED-LITERAL`, `DIAGNOSTIC-INVALID`, `MANUSCRIPT-REPORTED` | Only in the optional internal audit (`--with-reconstruction`); never in the paper tables. |

## What has and has not been executed

| Part | Status where written |
| --- | --- |
| Loaders (incl. Cresci-2015), labels, features, graphs, splits, overlap removal, perturbations, calibration, suite, resume, time limits, reports, plan, protocol audit, dataset locator (zip and tar.gz), config, notebook helpers | Tested (72 tests pass) on synthetic files with the real releases' structure; models replaced by numpy stand-ins |
| All PyTorch models (GCN, SAGE, GAT, R-GCN, RGF), LightGBM, XGBoost, BERT/SBERT | **Not executed** where written (no PyTorch, LightGBM, XGBoost, weights or network). 15 tests in `tests/test_real_backends.py` train every model on synthetic data; they run first in the Kaggle notebook (step 2) |
| Any result on Cresci-2017 or Cresci-2015 | **None yet** |

## Run everything on Kaggle CPU

Inputs (Add Input): this code (`botrecon_repo.zip`, private), Cresci-2017 and Cresci-2015 (both downloaded from
https://botometer.osome.iu.edu/bot-repository/datasets.html and uploaded as **private** datasets; zip or tar.gz is fine;
the Kaggle copy `hashemalsalmi/cresci-2017` is incomplete, 2 of 9 groups),
and from the second session on the previous `botrecon_results.zip`.

Open `notebooks/reproduce_kaggle_cpu.ipynb` (Accelerator: None, Internet: On) and run all cells. It executes
`python -m botrecon.plan --cresci <root> --cresci2015 <root>`, which runs, in priority order and resumably:

1. `botrecon.suite`, transductive, both datasets, 5 seeds: LightGBM (meta / text / meta+text), XGBoost, Random Forest,
   GCN, GraphSAGE, GAT, BotRGCN-style R-GCN (Cresci-2015), the hybrid baseline (4 versions, best chosen on
   validation), the proposed **RGF**, modality ablations; plus robustness (missing text, dropped edges, noisy edges,
   metadata mimicry), calibration (ECE, Brier, selective risk), cross-dataset transfer (shared accounts removed),
   explainability (gate weights, expert accuracy, permutation importance) and CPU efficiency.
2. RGF ablations, and the whole suite in the inductive (unseen-account) setting.
3. `botrecon.suite_report`: tables T1-T10, figures and `paper_facts.json` in `results/paper/`.

Without Cresci-2015 the plan runs Cresci-2017 alone; add Cresci-2015 later and rerun (finished runs are kept).
Run times are not yet measured on the real data: either *Save & Run All* unattended, or several 40-minute sessions
(download `botrecon_results.zip` at the end of each, add it as an input next time).

### The proposed model: RGF (Reliability-Gated Fusion), `botrecon/nn.py`

Three modality experts (metadata MLP; text MLP on frozen sentence embeddings; graph expert on two-hop neighbour means)
each output bot/human logits. A gate maps reliability signals (text available, graph available, degree, expert
entropies) to weights over the experts that are actually available for that account (masked softmax). Training uses
per-expert auxiliary losses and random modality dropout; a temperature fitted on validation data calibrates the output.
It is designed for accounts with missing text or no graph links, which the data audit quantifies.

## Data roots

- **Cresci-2017**: the folder that contains `genuine_accounts.csv/`, `social_spambots_1.csv/`, …, `fake_followers.csv/` (each with `users.csv`, and `tweets.csv` where the release has one).
- **Cresci-2015**: the folder that contains `E13.csv/`, `TFP.csv/`, `FSF.csv/`, `INT.csv/`, `TWT.csv/` (folder names with or without `.csv`), each with `users.csv`, `followers.csv`, `friends.csv` (and `tweets.csv`).
- Both are found and extracted automatically in the notebook.

**Overlap:** the Cresci-2017 fake followers come from the 2015 study. Cross-dataset tests remove accounts present in
both datasets by user ID and report how many were removed.

## Layout

```
botrecon/
  config.py       run configuration, variant grids, provenance per config
  provenance.py   provenance labels (old-manuscript numbers kept only for the internal audit)
  data/           Cresci-2017 and Cresci-2015 loaders (explicit label maps, dedup log); TwiBot-20 loader kept, unused
  features.py     4 core metadata features (+2 optional), train-fitted scaling
  text.py         BERT [CLS] on tweets (A), SBERT mean on bios (B), cache, empty-text rates
  graph.py        kNN (Cresci-2017) / follow (Cresci-2015) graphs, inductive variants, GCN normalisation, homophily
  splits.py       stratified 70-10-20, blending halves, folds
  gcn.py          2-layer GCN (plain PyTorch, same computation as PyG GCNConv)
  classifier.py   LightGBM stage
  pipeline.py     one run; test labels masked from all fitting
  metrics.py      bot-class, macro, weighted, MCC, ROC-AUC, PR-AUC, confusion
  stats.py        mean ± SD, t-CI, paired t, McNemar, paired bootstrap, Holm
  run.py          CLI; appends labelled rows to results/runs.csv, saves predictions
  report.py       tables R1–R6 from runs.csv
  inputs.py       model inputs and test-time perturbations
  nn.py           GCN, GraphSAGE, GAT, BotRGCN-style R-GCN, proposed RGF; trainer; temperature scaling
  methods.py      every compared method behind one fit/predict interface
  calib.py        ECE, Brier, selective risk, reliability curves
  suite.py        full experiment suite (resumable, time-limited); JSON Lines results
  suite_report.py paper tables T1–T10, figures, paper_facts.json
  plan.py         priority-ordered, resumable plan for Kaggle sessions (--part main / extended), logs
  audit.py        label and leakage-protocol audit on the real data
  env.py          dependency and hardware check
  datapaths.py    find / extract datasets (no manual path edits)
  settings.py     config/kaggle_cpu.json loader
  notebook.py     restart-safe notebook context and session budget
  walkthrough.py  notebook steps 8-14 (single seed, labelled WALKTHROUGH)
  repro_report.py results/REPRODUCIBILITY_RUN.md after each session
tests/            synthetic fixtures, protocol tests, real-backend tests
notebooks/        reproduce_kaggle_cpu.ipynb (use this); reproduce_colab.ipynb (legacy)
```

## Rules for using the output in the paper

1. Every number in the paper comes from `results/paper/` (tables, `paper_facts.json`); none is typed by hand.
2. The best hybrid version and any configuration choice are made on mean **validation** F1, never test.
3. Report the actually loaded dataset sizes and the number of shared accounts removed in cross-dataset tests.
4. Published numbers from other papers (different splits/features) are discussed in the text with citations, never placed in result tables.
