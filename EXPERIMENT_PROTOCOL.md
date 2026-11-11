# Experiment protocol

Applies to every `THIS-WORK` and `REPRODUCED-CORRECTED` result. Tags: [M] taken from the earlier manuscript's design,
[R] co-author recollection (unverified), [C] choice made here.

## Datasets and labels

The paper is drafted from scratch on new experiments. **TwiBot-20 is not used** (access is by request only and no
legitimate Kaggle copy exists); its loader stays in the library but no plan step, notebook cell or table uses it.

| | Cresci-2017 | Cresci-2015 |
| --- | --- | --- |
| Source | Official release, Botometer bot repository, uploaded as a private Kaggle dataset. The Kaggle copy `hashemalsalmi/cresci-2017` was found to hold only 2 of the 9 groups and must not be used; the notebook stops on any copy with missing groups | Official release, Botometer bot repository (https://botometer.osome.iu.edu/bot-repository/datasets.html), uploaded as a **private** Kaggle dataset |
| Accounts used | All nine groups after deduplication | All five groups after deduplication |
| Labels | **Explicit map [C]**: `genuine_accounts` = 0; `social_spambots_1-3`, `traditional_spambots_1-4`, `fake_followers` = 1 | E13, TFP = 0 (verified humans); FSF, INT, TWT = 1 (purchased fake followers) |
| Duplicates | Same ID in several groups: same label → keep one; conflicting labels → drop all copies; counts logged | Same rule |
| Graph | None released → kNN graph built from metadata | Follower and friend relations among the labelled accounts (`followers.csv`, `friends.csv`) |
| Split | Stratified 70 / 10 / 20 train / validation / test, re-drawn per seed; every method compared under one seed shares the split | Same |

Published group sizes. Cresci-2017 (Cresci et al., 2017): genuine 3,474; social spambots 991 / 3,457 / 464;
traditional spambots 1,000 / 100 / 433 / 1,128; fake followers 3,351 (total 14,398). Cresci-2015 (Cresci et al.,
2015): TFP 469, E13 1,481, FSF 1,169, INT 1,337, TWT 845 (total 5,301). The notebook reports the counts actually
loaded; no count is changed to match these.

**Overlap between the two datasets.** The Cresci-2017 fake-follower group (3,351 accounts) comes from the 2015 study
and has the same size as the Cresci-2015 FSF + INT + TWT groups; the run measures the actual overlap by user ID. Within-dataset results are unaffected. In cross-dataset tests every target test account whose user ID
also occurs in the source dataset is removed before scoring, and the number removed is reported (table T7). Because
of this, Cresci-2017 → Cresci-2015 may leave no bots to score (only humans: specificity is reported, bot-class
metrics are marked "—"), and Cresci-2015 → Cresci-2017 measures whether a fake-follower detector transfers to
spambots it never saw.

## Preprocessing

- **Metadata [M names, C formulas]:** account age (days before the latest creation time in the dataset), follower ratio
  followers / (following + 1), tweet frequency statuses / (age + 1), Shannon entropy of the screen-name characters.
  Non-finite values are treated as missing → median imputation → signed log1p → z-score. **Imputer and scaler are
  fitted on training rows only.** No identifiers are used as features.
- **Text:** variant B (default) `sentence-transformers/all-MiniLM-L6-v2` [C] on the profile description, mean pooling,
  384-d. Variant A `bert-base-uncased` [M], frozen, last-layer [CLS], up to 20 tweets joined, ≤ 128 tokens, 768-d.
  Empty text → zero vector, flagged as unavailable. Encoders run in eval mode (deterministic); embeddings are cached.
  Variants are never mixed within a run.
- **Graph:** Cresci-2015 follower/friend edges among labelled accounts, both directions merged, undirected [C]; the
  direction column convention is not relied on (only R-GCN uses direction). Cresci-2017 kNN,
  k = 10, cosine, on standardised metadata [M], symmetrised [C]. No self-edges in the edge list; the GCN adds self-loops
  in its normalisation. Isolated accounts are allowed and counted.

## Evaluation settings

| Setting | What the model sees while training | Use |
| --- | --- | --- |
| Transductive | All accounts' features and edges; labels of training (and, for selection, validation) accounts only | Main results; same setting as BotRGCN and RGT |
| Inductive | Training and validation accounts only. Test accounts are attached at inference: kNN to training/validation accounts (Cresci-2017) or their existing edges to them (Cresci-2015); they send no messages to seen accounts; no test-test edges | Unseen-account generalisation (table T5) |

## Leakage controls (all enforced in code and tested)

1. `y_view` copies the labels with every test label set to −1; all fitting reads only `y_view`.
2. GCN loss uses training (or blending-half) indices only; validation loss is used only for model selection.
3. LightGBM / XGBoost early stopping uses the validation set. The single exception is the `DIAGNOSTIC-INVALID` family,
   which deliberately early-stops on test data to measure how much that flaw inflates scores; it is never a result.
4. Stacking (hybrid): GCN trained on training half A, LightGBM on half B (blending). The manuscript's in-sample stacking
   exists only in the manuscript-faithful family.
5. Temperature scaling (RGF) uses validation data.
6. Primary configurations are chosen by mean **validation** F1.
7. Notebook step 7 re-checks 1, 7 and test isolation on the real data and stops on failure.

## Methods compared (same split and protocol for every row)

Tree models: LightGBM (metadata), LightGBM (text), LightGBM (metadata + text), XGBoost, Random Forest.
Graph models: GCN, GraphSAGE, GAT, BotRGCN-style R-GCN (Cresci-2015 only, which has a real follow graph; our re-implementation on our features),
GCN (metadata) = Metadata + GCN, GCN (text) = Text + GCN.
Hybrid baseline (metadata + text + GCN → LightGBM): P1 and P3, each with the 64-d or 2-d tap; the best of the four is
chosen on validation F1.
Proposed: RGF and 7 ablations.
Published numbers from other papers are not placed in the result tables (different splits and features); they may be
discussed in the text with citations.

## Metrics (`botrecon/metrics.py`)

Bot class is positive. Reported for every run:

- accuracy
- bot-class precision, recall and F1
- macro and weighted precision, recall and F1
- MCC, ROC-AUC, PR-AUC (average precision)
- confusion counts
- calibration: ECE (15 bins), Brier score, error at 80% and 90% coverage

Averaging conventions:

- **Bot class:** F1 = 2PR/(P+R) exactly.
- **Macro:** the mean of the per-class scores, so macro-F1 is not the harmonic mean of macro P and macro R.
- **Weighted:** the support-weighted mean; weighted recall always equals accuracy.

`check_consistency` verifies these relations against the confusion counts.

## Statistics (`botrecon/stats.py`)

- Seeds 0–4. Each result is reported as mean ± SD with a 95% t-interval (t = 2.776, 4 degrees of freedom); per-seed rows are kept.
- RGF is compared with each method by a paired t-test on per-seed differences, with Cohen's d_z as the effect size and Holm correction across comparisons.
- A comparison with zero variance gets p = 1 and is never called significant.
- The Wilcoxon signed-rank test is not used: with 5 seeds its smallest possible two-sided p is 0.0625.
- McNemar's test and a paired bootstrap are available for comparisons within one test set.

## Robustness, transfer, explainability (seeds 0–2, transductive)

**Test-time perturbations** apply to test accounts only:

- missing text: 0–100% of test accounts
- dropped edges: 0–100% of edges touching test accounts
- noisy edges: random edges added, 0–100% of the existing count
- metadata mimicry: 0–100% of test bots copy the median human training metadata (attack simulation)

**Cross-dataset:** train on one Cresci dataset and test on the other, using the source's scaler; accounts shared by
both datasets are removed from the target test set (see "Overlap" above).

**Explainability** (RGF):

- gate weights by account group
- each expert's accuracy on its own
- permutation importance of each metadata feature, plus the drop when all text or all graph links are removed

LightGBM split importance is also recorded.

## Provenance labels

In the paper tables: `THIS-WORK` and `REPRODUCED-CORRECTED` (the hybrid baseline). Never results: `WALKTHROUGH`
(single seed) and `SMOKE-TEST-NOT-A-RESULT`. `REPRODUCED-LITERAL`, `DIAGNOSTIC-INVALID` and `MANUSCRIPT-REPORTED`
belong only to the optional internal audit of the earlier manuscript (`--with-reconstruction`) and do not appear in
the paper tables.
