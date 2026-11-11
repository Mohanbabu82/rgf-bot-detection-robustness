# Figures and tables: captions, sources, placement and interpretation checks

Paper: *Accuracy Is Not Robustness: Reliability-Gated Multi-Modal Fusion for Social Bot Detection under Missing and Adversarial Modalities*

**Data provenance.** Every number comes from the Kaggle run's result files in `data/raw/` (copied unchanged; checksums in `data/raw/SHA256SUMS`):
- `suite_main.csv` — clean, ablation, inductive and cost results, 5 seeds.
- `suite_robust.csv` — perturbation results, seeds 0–2.
- `suite_cross.csv` — cross-dataset results, seeds 0–2.
- `suite_explain.csv` — gate weights and expert-alone results, seeds 0–2.
- `data_audit.json` — dataset counts.

No value was estimated, imputed or synthesised; cells with no experiment show "—". `scripts/verify.py` re-checks 461 plotted and tabulated values against the run's own report (`data/raw/tables.md`). Result: **0 mismatches.**

**Reproduce.** `cd scripts && python diagrams.py && python plots.py && python tables.py && python verify.py` (Python 3, pandas, matplotlib ≥ 3.8, scipy).

**Formats.** Figures are in `figures/` as vector PDF and SVG (fonts embedded as TrueType, Liberation Sans ≈ Arial) and as 600-dpi PNG. Widths are the IEEE double-column width (7.16 in). Each table is in `tables/` as `.tex` (IEEE-style `table*`, `\hline` rules), `.csv` and `.md`. `tables/all_tables.tex` holds all fifteen.

**Notation used throughout.**
- Datasets: C-17 = Cresci-2017, C-15 = Cresci-2015.
- ρ = perturbation level.
- F1 = bot-class F1. All metrics are in % except the Brier score.
- Results are reported as mean ± SD over seeds.
- "Hybrid P1-h" = the reconstructed BERT/SBERT + GCN + LightGBM hybrid, variant P1 with the 64-d hidden GCN layer. It is the best of four variants by validation F1 on both datasets.

**Section placement.** This follows the planned manuscript structure:
- II Related Work
- III Datasets and Problem Formulation
- IV Methodology
- V Experimental Setup
- VI Results (A clean, B ablation, C mimicry, D missing text, E graph, F inductive, G cross-dataset, H calibration/selective, I gate weights, J cost)
- VII Discussion
- VIII Limitations

Robustness results labelled "VII-x" below now sit in Section VI (C–I) of the revised draft.

Adjust if the outline changes.

> **Renumbering needed:** the drafted Section I refers to the clean-vs-mimicry scatter as "Fig. 1". In this package Fig. 1 is the framework and the accuracy-versus-robustness evidence is **Fig. 6**. Update Section I's reference when it is revised.

---

## Part A — Diagrams

### Fig. 1 — Research framework
**File:** `fig01_research_framework` · **Script:** `diagrams.py::fig1` · **Section:** III (overview), referenced from I

**Caption.** Fig. 1. Research framework. The framework covers the following, left to right:
- **Datasets:** two Cresci benchmarks.
- **Leakage-controlled preprocessing:** stratified 70/10/20 splits over five seeds; transformations fitted on training rows; test labels masked.
- **Features:** metadata (4-d), profile text (384-d, frozen MiniLM) and graph (metadata kNN for Cresci-2017; follower/friend relations for Cresci-2015).
- **Models:** tree, GNN and hybrid baselines and the proposed RGF.
- **Test-time perturbations:** applied to a model trained on clean data.
- **Evaluation:** discrimination, probability quality, inductive and cross-dataset transfer, and Holm-corrected paired tests.

**Source:**
- `data_audit.json`: 14,368 and 5,301 accounts; 75.8% and 63.2% bots.
- `suite_main.csv`: 64.3% of Cresci-2015 test accounts isolated.
- Code: `features.py`, `splits.py`, `suite.py`.

**Check:** every box corresponds to a step that the run executed. The tweet-level BERT encoder exists in the code but was not run (`run_bert_tweets: false`), so it is not shown.

### Fig. 2 — RGF architecture
**File:** `fig02_rgf_architecture` · **Script:** `diagrams.py::fig2` · **Section:** IV-A to IV-D

**Caption.** Fig. 2. Reliability-gated fusion (RGF):
- **Experts:** three experts produce class logits $z^m_v$, $z^t_v$, $z^g_v$. The text expert's output is multiplied by text availability $a^t_v$. The graph expert aggregates the neighbours' metadata and text encodings over two mean-aggregation layers without self-loops, and is multiplied by graph availability $a^g_v$.
- **Gate:** a gate MLP maps six reliability signals to expert scores: the two availability flags, scaled log-degree, and the stop-gradient entropy of each expert's prediction.
- **Mask:** a softmax restricted by the mask $o_v=[1,a^t_v,a^g_v]$ gives weights that are zero for unavailable experts.
- **Output:** the fused logits are the weighted sum of expert logits, divided by a temperature $T$ fitted on validation data.

**Source:** `nn.py` (class RGF) and Section IV, Eqs. (1)–(11). Layer sizes come from the code: 4→64, 384→64, 128→64→64, heads 64→2, gate 6→32→3.

**Check:**
- No scalar "reliability score" is drawn, because none exists in the code.
- The figure states that the gate inputs are availability, connectivity and confidence signals.

### Fig. 3 — Training workflow
**File:** `fig03_training_workflow` · **Script:** `diagrams.py::fig3` · **Section:** IV-E (with Algorithm 1)

**Caption.** Fig. 3. Training of RGF.
- **Modality dropout:** at each full-batch epoch, text and graph availability are dropped independently per account with probability 0.3; metadata is never dropped.
- **Loss:** the fused cross-entropy plus 0.5 × the cross-entropy of each expert on training accounts where that expert is available. AdamW updates the parameters.
- **Early stopping:** validation cross-entropy, computed without any dropout and with T = 1, selects the stored parameters (patience 40, at most 300 epochs).
- **Temperature:** T is then chosen on the same validation split by grid search over 60 log-spaced values in [0.25, 5].

**Source:** `nn.py::fit`, `fit_temperature`; `methods.py::RGFMethod.fit`.

**Check:** the figure shows the two weaknesses found in the technical audit:
- the degree term is not dropped together with $a^g$ ("deg kept");
- early stopping and temperature use the same validation split.

### Fig. 4 — Evaluation framework
**File:** `fig04_evaluation_framework` · **Script:** `diagrams.py::fig4` · **Section:** V

**Caption.** Fig. 4. Evaluation protocol. Each model is trained once per dataset and seed on clean data, then evaluated on seven axes:
- clean test performance;
- missing modality (text removal, edge deletion);
- adversarial perturbation (metadata mimicry, random edge injection);
- inductive evaluation on accounts unseen by the training graph;
- cross-dataset transfer with shared accounts removed;
- calibration;
- selective prediction.

Perturbations affect test accounts only, and the graph is not rebuilt afterwards.

**Source:** `suite.py` and `suite_robust.csv` (levels: drop and missing-text 0–100% in steps of 25; mimicry and injection {0, 25, 50, 100}%).

**Check:** the seed counts match the files: 5 for clean and inductive, 3 for robustness and cross-dataset.

---

## Part B — Experimental graphs

### Fig. 5 — Clean performance
**File:** `fig05_clean_performance` · **Data:** `data/derived/fig05_clean_performance.csv` · **Script:** `plots.py::fig5` · **Section:** VI-A (with Table 4)

**Caption.** Fig. 5. Clean test performance on Cresci-2017 (circles) and Cresci-2015 (squares): (a) bot-class F1, (b) MCC, (c) ROC-AUC. Markers show the mean and bars ± 1 SD over five seeds (transductive setting). The R-GCN baseline needs typed relations and is run on Cresci-2015 only.

**Interpretation check (what the data support):**
- **Cresci-2017:** RGF is *not* the most accurate model.
  - F1 is 97.28 vs LightGBM 98.34, XGBoost 98.39 and GraphSAGE 98.22.
  - MCC is 88.59 vs 93.34, 93.51 and 92.61.
  - These differences are significant after Holm correction (Table 4, †). Under the Nadeau–Bengio corrected resampled test (`audit_stats/clean_tests.csv`), XGBoost and LightGBM remain significant but GraphSAGE does not (p = 0.11).
  - RGF is significantly better than GCN and GAT.
- **Cresci-2015:** the clean differences are small.
  - RGF F1 97.98 vs Hybrid P1-h 98.39 and LightGBM 98.08 (not significant).
  - RGF's ROC-AUC, 99.54, equals the highest (GraphSAGE 99.54).
- Do not describe RGF as state of the art on clean data.

### Fig. 6 — Accuracy versus robustness
**File:** `fig06_accuracy_vs_robustness` · **Data:** `fig06_mimicry_f1.csv`, `fig06_mimicry_ranks.csv` · **Script:** `plots.py::fig6` · **Section:** VII-A (central result; cite in I and the abstract)

**Caption.** Fig. 6. Accuracy is not robustness.
- **(a, b)** Bot-class F1 when a fraction ρ of test bots receive the median metadata of training humans. Mean and ± 1 SD band over three seeds; models trained on clean data.
- **(c, d)** Rank of each model by F1 on clean data and under full mimicry (ρ = 100%), with F1 values (%).

The models that are most accurate on clean data are among the least robust.

**Interpretation check:**
- The ranking inversion is real on both datasets.
  - LightGBM falls from 98.24 to 40.17 (C-17) and from 98.02 to 0.00 (C-15).
  - Hybrid P1-h falls to 68.88 and 0.49.
- RGF degrades much less: 97.28 → 94.47 on C-17 and 98.07 → 55.25 on C-15.
- RGF is **not** the most robust model:
  - on C-17, GAT (95.99) and GCN (94.82) are ahead of it;
  - on C-15, GAT (60.71) is ahead of it.
- **C-17 caveat:** the kNN graph is computed from the *original* metadata and is not rebuilt after the attack. Graph models therefore still aggregate over neighbours chosen by the unperturbed features, which favours them. Say this in the text.
- The attack is white-box on the feature: the attacker knows the test bots and the human median.

### Fig. 7 — Missing-text robustness
**File:** `fig07_missing_text` · **Data:** `fig07_missing_text_f1.csv` · **Script:** `plots.py::fig7` · **Section:** VII-B (with Table 6)

**Caption.** Fig. 7. Bot-class F1 when the profile description of a fraction ρ of test accounts is removed (text set to zero, availability flag 0). Mean and ± 1 SD band over three seeds; models trained on clean data.

**Interpretation check:**
- **Cresci-2017:** RGF has the highest F1 at full text removal: 95.04 vs GAT 94.71, GCN 94.62, Hybrid P1-h 93.62, GraphSAGE 90.28 and LightGBM 89.75. It loses 2.24 points, against 8.49 for LightGBM.
  - RGF's margin over GAT and GCN is smaller than one SD.
  - RGF is below LightGBM and GraphSAGE only at ρ = 0; from ρ = 25% onwards it is above them.
- **Cresci-2015:** all methods except GAT stay within about 1 point of each other; at ρ = 100%, R-GCN 97.76 and Hybrid P1-h 97.73 are above RGF (97.47).
- The claim "RGF degrades least when text is missing" holds on Cresci-2017 only.

### Fig. 8 — Graph perturbation
**File:** `fig08_graph_perturbation` · **Data:** `fig08_graph_perturbation_f1.csv` · **Script:** `plots.py::fig8` · **Section:** VII-C (with Table 7)

**Caption.** Fig. 8. Bot-class F1 under (a, c) deletion of a fraction ρ of the edges incident to test accounts and (b, d) injection of round(ρ × degree) random edges per test account.
- Mean and ± 1 SD band over three seeds.
- LightGBM (dashed) uses no graph and serves as a constant reference.
- On Cresci-2017 the edges come from the metadata kNN graph.

**Interpretation check:**
- **Edge deletion, C-17, ρ = 100%:** RGF (96.49) beats every graph model (GraphSAGE 95.02, GAT 92.12, GCN 92.37).
- **Edge injection, C-15:** RGF (93.61) ≈ R-GCN (93.65) ≈ GraphSAGE (93.03), well above GCN and GAT (≈ 86.7) but below Hybrid P1-h (95.89).
- The graph-free LightGBM is at or above RGF at every level, apart from differences of at most 0.05 points at ρ ≤ 25% on C-15. A graph attack cannot hurt a model that ignores the graph, so this panel measures how much each graph model loses, not which model is most accurate.
- Do not claim RGF is robust "in absolute terms" to graph attacks: random injection is not an optimised attack.

### Fig. 9 — Cross-dataset generalisation
**File:** `fig09_cross_dataset` · **Data:** `fig09_cross_dataset.csv` · **Script:** `plots.py::fig9` · **Section:** VII-E (with Table 10)

**Caption.** Fig. 9. Cross-dataset transfer, mean ± SD over three seeds.
- **(a)** Trained on Cresci-2015 and tested on Cresci-2017 (1,507 bots, 695 humans after removing 672 shared accounts).
- **(b)** Share of humans classified as bots in both directions. Removing shared accounts leaves no bots in the Cresci-2015 test set, so the Cresci-2017 → Cresci-2015 direction measures only false positives on 390 humans.

**Interpretation check:**
- **C-15 → C-17:** RGF F1 is 67.72 and ROC-AUC 81.40. GCN is slightly higher on F1 and MCC (69.78, 47.72), with the same ROC-AUC (81.38). LightGBM and Hybrid P1-h transfer worst (F1 54.91 and 53.24).
- **C-17 → C-15:** every model flags most humans as bots. RGF flags 82.5%, GCN 85.2%, GraphSAGE 78.7%, Hybrid P1-h 70.3%; LightGBM is lowest at 53.5%.
- This is a strong **negative** result for all methods, RGF included. Report it, and do not describe RGF as generalising across datasets.

### Fig. 10 — Ablation
**File:** `fig10_ablation` · **Data:** `fig10_ablation_clean.csv`, `fig10_ablation_robust.csv` · **Script:** `plots.py::fig10` · **Section:** VI-B (clean) and VII-D (robust); with Tables 5 and 8

**Caption.** Fig. 10. Ablation of RGF.
- **(a, b)** Clean F1 and MCC of the full model and its seven ablations (five seeds).
- **(c)** F1 at full metadata mimicry and full text removal for the three ablations evaluated under perturbation (three seeds).

**Interpretation check:**
- **Clean data:** removing the text expert costs 5.96 MCC on C-17, and removing the metadata expert costs 11.17 on C-15 (both significant).
- **Concat fusion is better on clean data:** +2.68 MCC on C-17 and +1.17 on C-15 (Holm p = 0.049 and 0.037; not significant under the corrected test). It collapses under mimicry: 59.54 (C-17) and 26.68 (C-15), against 94.47 and 55.25 for RGF.
  - Concat also removes the auxiliary losses, so this gap is not due to the fusion rule alone.
- **Mean of experts (no gate)** lies between them: 88.89 and 48.83 under mimicry.
- **Modality dropout does not explain the robustness:** without it, F1 under mimicry is 95.96 / 54.70 (vs 94.47 / 55.25), and the clean differences are insignificant.
- Only three of the seven ablations were run under perturbation; panel (c) shows those three, without any estimates for the other four.

### Fig. 11 — Calibration and selective prediction
**File:** `fig11_calibration_selective` · **Data:** `fig11_calibration_selective.csv` · **Script:** `plots.py::fig11` · **Section:** VI-C (with Tables 11–12)

**Caption.** Fig. 11. Probability quality on the test split (mean ± SD over five seeds):
- **(a)** expected calibration error (15 bins, log scale);
- **(b)** Brier score;
- **(c, d)** error on the 80% and 90% most confident test accounts.

Hollow markers in (a) show RGF before temperature scaling.

**Interpretation check:**
- **Calibration:** RGF's ECE is mid-pack: 1.17 on C-17 (XGBoost 0.87 is lowest) and 1.71 on C-15 (Hybrid P1-h 1.00 is lowest). Temperature scaling did not improve ECE (1.05 → 1.17; 1.54 → 1.71); the changes are within seed variation (p = 0.38 and 0.27) and T stayed between 0.69 and 1.15.
- **Selective prediction, C-15:** RGF has the lowest error at 80% and 90% coverage (0.31, 0.42), but the SDs (0.27, 0.25) overlap with the trees.
- **Selective prediction, C-17:** RGF's error at 90% coverage (1.86) is two to three times that of the trees (0.63–0.87).
- The random forest's high ECE (8.7–10.4) comes from its uncalibrated vote fractions.

### Fig. 12 — Computational cost
**File:** `fig12_computational_cost` · **Data:** `fig12_cost.csv` · **Script:** `plots.py::fig12` · **Section:** VI-D (with Table 14)

**Caption.** Fig. 12. Cost on a Kaggle CPU session (mean ± SD over five seeds, transductive):
- **(a)** training time including early stopping;
- **(b)** inference time per 1,000 accounts;
- **(c)** model size: trainable parameters for neural models, number of trees for tree models. The two units are not comparable.

RGF uses 38,089 of its 50,571 registered parameters.

**Interpretation check:**
- RGF trains in 43.2 s (C-17) and 9.4 s (C-15): 2–4 times as long as LightGBM, GCN or GraphSAGE, but less than half as long as GAT on C-17 (107.5 s).
- Inference (10.2 and 4.9 ms per 1,000 accounts) is comparable to LightGBM; the random forest is slowest (41–46 ms).
- All timings are from one shared CPU environment. Treat them as relative, not as hardware benchmarks.

---

## Part C — Tables (`tables/tableNN_*.{tex,csv,md}`; script `tables.py`)

| No. | Title | Source | Section | Interpretation check / limitation |
|---|---|---|---|---|
| I | Dataset statistics | `data_audit.json`, `suite_main.csv` | V-A | Cresci-2017 traditional spambots 3: 403 of 433 published accounts (protocol audit FAIL, disclosed). |
| II | Preprocessing and split configuration | code, protocol audit | V-B | No hyper-parameter search was run; stated. |
| III | Architectures and hyper-parameters | `methods.py`, `nn.py` | V-C | Values fixed in code; GAT uses lr 0.005, the other GNNs 0.01. |
| IV | Clean performance (7 metrics, both datasets) | `suite_main.csv` | VI-A | † / ‡ = Holm-significant paired t-test vs RGF, recomputed here; it reproduces the run's own test table (checked: XGBoost C-17 Δ −1.11, Holm p 0.0044). Full results in `table04_significance_detail.csv`. With 5 seeds the tests are indicative. |
| V | Ablation | `suite_main.csv` | VI-B | Concat fusion also drops the auxiliary losses (two changes at once). |
| VI | Missing text | `suite_robust.csv` | VII-B | ρ = 0 uses 3 seeds, so it differs slightly from Table IV (5 seeds). |
| VII | Graph perturbation | `suite_robust.csv` | VII-C | Random injection, not an optimised attack. |
| VIII | Metadata mimicry (F1 at all ρ, MCC at 100%) | `suite_robust.csv` | VII-A | Cresci-2017 kNN graph not rebuilt (favours graph models); white-box feature attack. |
| IX | Inductive evaluation | `suite_main.csv` | VII-E | Trees identical by construction; XGBoost, random forest and R-GCN were not run inductively. |
| X | Cross-dataset evaluation | `suite_cross.csv` | VII-E | C-17 → C-15 has no test bots; only the human FPR is defined. |
| XI | Calibration (+ RGF before T, and T) | `suite_main.csv` | VI-H | ECE and Brier slightly higher after temperature scaling; the change is not significant (p = 0.38, 0.27). |
| XII | Selective prediction (error at 100/90/80% coverage) | `suite_main.csv` | VI-C | The ranking is unaffected by T. |
| XIII | Reliability-gate weights + expert-alone accuracy | `suite_explain.csv` | VII-F | Weights show reliance, not correctness. Cresci-2015 graph expert alone: n = 385 connected accounts only. |
| XIV | Computational cost | `suite_main.csv` | VI-D | Size units differ (parameters vs trees); RGF 38,089 used / 50,571 registered. |
| XV | Positioning against related work | `references/VERIFIED_REFERENCES.csv` | II | Qualitative only. Numbers from other papers use different data and protocols, so no numeric ranking is given. BotCHF and BoCloak have partly unverified bibliographic details (see `UNVERIFIED_REFERENCES.md`). |

## Items requested but not produced as asked

- **Fig. 10 robustness of all seven ablations.** Only three ablations (concat, mean of experts, w/o modality dropout) were evaluated under perturbation in the run. The other four have clean results only and are not extrapolated.
- **Table XV numeric comparison with prior work.** No verified Cresci-2015/2017 results under a comparable protocol exist in the reference library, so the table is qualitative.
- **Statistical tests for robustness, cross-dataset and calibration results.** None were run, so no significance is claimed for Figs. 6–11 or Tables VI–XIII.
