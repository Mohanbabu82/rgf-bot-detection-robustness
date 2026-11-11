# Tables 1-15

### TABLE 1. Dataset statistics

| Property | Cresci-2017 | Cresci-2015 |
|---|---|---|
| Accounts (after deduplication) | 14,368 | 5,301 |
| Bots / humans | 10,894 / 3,474 | 3,351 / 1,950 |
| Bot share (%) | 75.8 | 63.2 |
| Human groups | genuine accounts | E13, TFP |
| Bot groups | social spambots 1-3, traditional spambots 1-4, fake followers | FSF, INT, TWT |
| Empty profile description, bots / humans (%) | 49.2 / 10.9 | 32.0 / 7.4 |
| Graph | none released; kNN (k = 10, cosine) on metadata | follower + friend relations |
| Directed edges among labelled accounts | — | 7,108 |
| Isolated accounts (all / test, %) | 0 / 0.0 | 3,381 / 64.3 |
| Test accounts without text (%) | 40.6 | 21.0 |
| Edge homophily, training nodes | 0.894 | 0.973 |

*Counts from the run's data audit (data_audit.json). Cresci-2017 group traditional spambots 3 has 403 of the 433 published accounts. The Cresci-2017 kNN graph is built from the same metadata the models use.*

### TABLE 2. Preprocessing and split configuration

| Item | Setting |
|---|---|
| Split | stratified 70 / 10 / 20 per seed; all methods share the split of a seed |
| Seeds | 0-4 (clean, inductive, ablation); 0-2 (robustness, cross-dataset, explanation) |
| Train / validation / test, Cresci-2017 | 10,057 / 1,437 / 2,874 |
| Train / validation / test, Cresci-2015 | 3,710 / 530 / 1,061 |
| Metadata features (d = 4) | account age (days), followers/(friends+1), statuses/(age+1), screen-name entropy (bits) |
| Metadata transform | ±inf → missing; training-median imputation; sign(x)·log(1+|x|); training z-score |
| Text features (d = 384) | profile description; all-MiniLM-L6-v2, frozen; empty → zero vector, availability 0 |
| Graph, transductive | test accounts in the graph, labels hidden |
| Graph, inductive | test accounts absent from training graph; no test-to-seen messages |
| Label leakage controls | test labels stored as -1 and asserted absent from fit/validation sets; scaler fitted on training rows |
| Model selection | validation loss (neural), validation log-loss early stopping (trees); no hyper-parameter search |

*Transcribed from botrecon/features.py, splits.py, suite.py and the run's protocol audit (39 of 40 checks pass; the failure is the Cresci-2017 group-size check).*

### TABLE 3. Model architectures and hyper-parameters

| Model | Inputs | Configuration |
|---|---|---|
| LightGBM | meta+text | gbdt, 31 leaves, lr 0.05, ≤ 1000 trees, early stop 50 (val log-loss) |
| XGBoost | meta+text | ≤ 1000 trees, lr 0.05, depth 6, subsample 0.9, colsample 0.9, hist, early stop 50 |
| Random forest | meta+text | 500 trees, scikit-learn defaults, no class weights |
| GCN | meta+text, graph | 2 layers d→64→2, ReLU, dropout 0.5, sym.-normalised A + I |
| GraphSAGE | meta+text, graph | 2 layers, mean aggregator, 64 hidden, ReLU, dropout 0.5 |
| GAT | meta+text, graph | 4 heads × 16, ELU; output 1 head; dropout 0.5; self-loops |
| R-GCN (BotRGCN-style) | meta+text, 2 relations | proj. 64, 2 relational layers, MLP 64→64→2, dropout 0.3 (Cresci-2015 only) |
| Hybrid P1 / P3 | meta+text, graph | GCN (half A of train, 100 epochs) → LightGBM (half B); P1 = [meta, text, GCN], P3 = GCN only |
| Neural training | — | full batch, AdamW, wd 5×10⁻⁴, lr 0.01 (GAT 0.005), ≤ 200 epochs, patience 30 |
| RGF (proposed) | meta, text, graph | experts h = 64, dropout 0.3; gate 6→32→3; λ = 0.5; modality dropout 0.3; AdamW lr 0.005, wd 5×10⁻⁴, ≤ 300 epochs, patience 40; temperature grid 60 values in [0.25, 5] |

*All values are fixed in code (methods.py, nn.py); none was tuned. The hybrid reported in the main tables is P1 with the GCN's 64-d hidden layer, the best of four variants by mean validation F1 on both datasets.*

### TABLE 4. Clean test performance (%, mean ± SD over 5 seeds, transductive)

| Method | Acc | Prec | Rec | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Cresci-2017 (5 seeds)** |  |  |  |  |  |  |  |
| LightGBM | 97.50 ± 0.26 | 99.09 ± 0.14 | 97.60 ± 0.30 | 98.34 ± 0.18† | 93.34 ± 0.69† | 99.41 ± 0.07 | 99.82 ± 0.03 |
| XGBoost | 97.57 ± 0.22 | 99.09 ± 0.13 | 97.70 ± 0.24 | 98.39 ± 0.15† | 93.51 ± 0.57† | 99.45 ± 0.06 | 99.82 ± 0.03 |
| Random forest | 96.87 ± 0.24 | 98.23 ± 0.34 | 97.63 ± 0.23 | 97.93 ± 0.15 | 91.52 ± 0.67† | 99.14 ± 0.11 | 99.71 ± 0.06 |
| GCN | 93.21 ± 0.21 | 95.79 ± 0.26 | 95.23 ± 0.48 | 95.51 ± 0.15‡ | 81.61 ± 0.47‡ | 97.10 ± 0.25 | 98.98 ± 0.16 |
| GraphSAGE | 97.29 ± 0.14 | 98.13 ± 0.22 | 98.30 ± 0.12 | 98.22 ± 0.09† | 92.61 ± 0.40† | 99.08 ± 0.12 | 99.69 ± 0.05 |
| GAT | 93.58 ± 0.27 | 95.96 ± 0.40 | 95.56 ± 0.10 | 95.76 ± 0.17‡ | 82.58 ± 0.84‡ | 97.00 ± 0.25 | 98.87 ± 0.14 |
| Hybrid P1-h | 96.55 ± 0.26 | 98.27 ± 0.21 | 97.15 ± 0.27 | 97.71 ± 0.18 | 90.73 ± 0.70 | 99.00 ± 0.23 | 99.64 ± 0.14 |
| RGF (proposed) | 95.85 ± 0.43 | 96.82 ± 0.37 | 97.74 ± 0.27 | 97.28 ± 0.28 | 88.59 ± 1.20 | 98.48 ± 0.13 | 99.47 ± 0.06 |
| **Cresci-2015 (5 seeds)** |  |  |  |  |  |  |  |
| LightGBM | 97.57 ± 0.37 | 97.98 ± 0.66 | 98.18 ± 0.45 | 98.08 ± 0.28 | 94.77 ± 0.80 | 99.36 ± 0.35 | 99.53 ± 0.25 |
| XGBoost | 97.59 ± 0.31 | 98.04 ± 0.51 | 98.15 ± 0.29 | 98.09 ± 0.24 | 94.81 ± 0.67 | 99.53 ± 0.25 | 99.71 ± 0.17 |
| Random forest | 97.55 ± 0.19 | 98.09 ± 0.45 | 98.03 ± 0.22 | 98.06 ± 0.14 | 94.73 ± 0.41 | 99.17 ± 0.34 | 99.29 ± 0.38 |
| GCN | 96.57 ± 0.56 | 96.52 ± 0.90 | 98.12 ± 0.34 | 97.31 ± 0.43 | 92.61 ± 1.21 | 99.11 ± 0.46 | 99.44 ± 0.33 |
| GraphSAGE | 97.30 ± 0.33 | 96.96 ± 0.46 | 98.84 ± 0.63 | 97.89 ± 0.26 | 94.20 ± 0.72 | 99.54 ± 0.11 | 99.72 ± 0.07 |
| GAT | 96.51 ± 0.44 | 95.79 ± 0.81 | 98.84 ± 0.34 | 97.29 ± 0.33 | 92.51 ± 0.92 | 99.39 ± 0.16 | 99.63 ± 0.08 |
| R-GCN (BotRGCN-style) | 97.42 ± 0.57 | 97.58 ± 0.68 | 98.36 ± 0.35 | 97.97 ± 0.45 | 94.44 ± 1.24 | 99.47 ± 0.21 | 99.67 ± 0.13 |
| Hybrid P1-h | 97.96 ± 0.21 | 98.54 ± 0.45 | 98.24 ± 0.32 | 98.39 ± 0.16 | 95.63 ± 0.45 | 99.46 ± 0.35 | 99.61 ± 0.29 |
| RGF (proposed) | 97.44 ± 0.18 | 97.75 ± 0.46 | 98.21 ± 0.57 | 97.98 ± 0.15 | 94.49 ± 0.39 | 99.54 ± 0.24 | 99.71 ± 0.14 |

*Bot-class precision, recall and F1. † RGF is significantly lower than this method, ‡ significantly higher (paired t-test over seeds, Holm correction over all comparisons with RGF on that dataset and metric, α = 0.05; with 5 seeds these tests are indicative). Under the Nadeau-Bengio corrected resampled t-test, RGF vs GraphSAGE on Cresci-2017 is no longer significant (audit_stats/clean_tests.csv). Full test results: table04_significance_detail.csv.*

### TABLE 5. Ablation of RGF (%, mean ± SD over 5 seeds)

| Variant | F1 C-17 | MCC C-17 | ΔMCC C-17 | F1 C-15 | MCC C-15 | ΔMCC C-15 |
|---|---|---|---|---|---|---|
| RGF (proposed) | 97.28 ± 0.28 | 88.59 ± 1.20 | — | 97.98 ± 0.15 | 94.49 ± 0.39 | — |
| Concat fusion | 97.89 ± 0.08 | 91.27 ± 0.26 | +2.68† | 98.41 ± 0.11 | 95.66 ± 0.31 | +1.17† |
| w/o gate (mean) | 97.49 ± 0.24 | 89.47 ± 1.03 | +0.87 | 98.24 ± 0.16 | 95.22 ± 0.44 | +0.73† |
| w/o auxiliary losses | 97.83 ± 0.22 | 91.04 ± 0.97 | +2.44 | 98.06 ± 0.19 | 94.68 ± 0.53 | +0.20 |
| w/o modality dropout | 97.31 ± 0.30 | 88.68 ± 1.29 | +0.09 | 97.89 ± 0.23 | 94.24 ± 0.65 | -0.25 |
| w/o metadata expert | 97.15 ± 0.31 | 88.10 ± 1.30 | -0.49 | 93.98 ± 0.34 | 83.32 ± 0.90 | -11.17‡ |
| w/o text expert | 95.76 ± 0.18 | 82.63 ± 0.62 | -5.96‡ | 98.07 ± 0.40 | 94.72 ± 1.12 | +0.24 |
| w/o graph expert | 96.62 ± 0.19 | 85.50 ± 0.82 | -3.09‡ | 97.41 ± 0.23 | 92.90 ± 0.64 | -1.59 |

*ΔMCC = variant − RGF in percentage points; † / ‡ as in Table 4 (here † = variant significantly better than RGF). Concat fusion also removes the auxiliary losses, so it changes two components.*

### TABLE 6. Bot-class F1 (%) when the profile text of a fraction ρ of test accounts is removed

| Method | ρ = 0% | ρ = 25% | ρ = 50% | ρ = 75% | ρ = 100% | Δ (100% − 0%) |
|---|---|---|---|---|---|---|
| **Cresci-2017 (3 seeds)** |  |  |  |  |  |  |
| LightGBM | 98.24 ± 0.16 | 96.40 ± 0.14 | 94.25 ± 0.54 | 92.24 ± 0.71 | 89.75 ± 1.07 | -8.49 |
| GCN | 95.48 ± 0.20 | 95.34 ± 0.22 | 95.01 ± 0.30 | 94.93 ± 0.26 | 94.62 ± 0.34 | -0.86 |
| GraphSAGE | 98.17 ± 0.09 | 96.44 ± 0.33 | 94.62 ± 0.27 | 92.48 ± 0.55 | 90.28 ± 0.72 | -7.90 |
| GAT | 95.84 ± 0.11 | 95.63 ± 0.13 | 95.34 ± 0.19 | 95.07 ± 0.16 | 94.71 ± 0.27 | -1.13 |
| Hybrid P1-h | 97.64 ± 0.05 | 96.76 ± 0.04 | 95.86 ± 0.26 | 94.84 ± 0.18 | 93.62 ± 0.12 | -4.03 |
| RGF (proposed) | 97.28 ± 0.32 | 96.75 ± 0.37 | 96.26 ± 0.34 | 95.69 ± 0.04 | 95.04 ± 0.30 | -2.24 |
| Concat fusion | 97.87 ± 0.09 | 97.26 ± 0.15 | 96.64 ± 0.02 | 95.93 ± 0.16 | 95.22 ± 0.12 | -2.65 |
| w/o gate (mean) | 97.48 ± 0.34 | 97.07 ± 0.35 | 96.35 ± 0.24 | 95.75 ± 0.18 | 95.10 ± 0.13 | -2.38 |
| w/o modality dropout | 97.24 ± 0.39 | 96.67 ± 0.48 | 96.10 ± 0.34 | 95.46 ± 0.15 | 94.76 ± 0.40 | -2.48 |
| **Cresci-2015 (3 seeds)** |  |  |  |  |  |  |
| LightGBM | 98.02 ± 0.36 | 97.82 ± 0.44 | 97.70 ± 0.31 | 97.73 ± 0.60 | 97.47 ± 0.56 | -0.54 |
| GCN | 97.46 ± 0.36 | 97.54 ± 0.18 | 97.10 ± 0.12 | 96.90 ± 0.53 | 96.68 ± 0.59 | -0.78 |
| GraphSAGE | 97.96 ± 0.33 | 97.92 ± 0.27 | 97.51 ± 0.22 | 97.21 ± 0.57 | 96.90 ± 0.62 | -1.07 |
| GAT | 97.43 ± 0.24 | 96.96 ± 0.29 | 95.86 ± 0.28 | 95.02 ± 0.35 | 94.42 ± 0.64 | -3.02 |
| R-GCN (BotRGCN-style) | 97.85 ± 0.46 | 98.02 ± 0.48 | 97.97 ± 0.62 | 97.74 ± 0.76 | 97.76 ± 0.84 | -0.09 |
| Hybrid P1-h | 98.33 ± 0.16 | 98.26 ± 0.23 | 98.04 ± 0.29 | 97.90 ± 0.44 | 97.73 ± 0.51 | -0.60 |
| RGF (proposed) | 98.07 ± 0.07 | 98.15 ± 0.20 | 97.83 ± 0.10 | 97.61 ± 0.11 | 97.47 ± 0.34 | -0.60 |
| Concat fusion | 98.42 ± 0.15 | 98.39 ± 0.11 | 98.12 ± 0.15 | 98.05 ± 0.09 | 97.93 ± 0.07 | -0.49 |
| w/o gate (mean) | 98.34 ± 0.08 | 98.47 ± 0.09 | 98.22 ± 0.23 | 98.07 ± 0.52 | 97.98 ± 0.52 | -0.37 |
| w/o modality dropout | 97.95 ± 0.25 | 98.08 ± 0.07 | 97.61 ± 0.17 | 97.28 ± 0.12 | 97.21 ± 0.33 | -0.74 |

*Mean ± SD over seeds 0-2; models trained on clean data. Δ in percentage points. ρ = 0 differs slightly from Table 4, which averages seeds 0-4.*

### TABLE 7. Bot-class F1 (%) under graph perturbation of test accounts

| Method | Perturbation | ρ = 0% | ρ = 25% | ρ = 50% | ρ = 75% | ρ = 100% | Δ (100% − 0%) |
|---|---|---|---|---|---|---|---|
| **Cresci-2017 (3 seeds)** |  |  |  |  |  |  |  |
| LightGBM | edge deletion | 98.24 ± 0.16 | 98.24 ± 0.16 | 98.24 ± 0.16 | 98.24 ± 0.16 | 98.24 ± 0.16 | +0.00 |
| GCN | edge deletion | 95.48 ± 0.20 | 95.57 ± 0.19 | 95.60 ± 0.18 | 95.53 ± 0.46 | 92.37 ± 0.83 | -3.11 |
| GraphSAGE | edge deletion | 98.17 ± 0.09 | 98.17 ± 0.12 | 98.06 ± 0.07 | 97.50 ± 0.08 | 95.02 ± 0.32 | -3.16 |
| GAT | edge deletion | 95.84 ± 0.11 | 95.75 ± 0.07 | 95.48 ± 0.10 | 95.39 ± 0.32 | 92.12 ± 0.31 | -3.71 |
| Hybrid P1-h | edge deletion | 97.64 ± 0.05 | 97.58 ± 0.08 | 97.54 ± 0.11 | 97.33 ± 0.18 | 96.33 ± 0.15 | -1.31 |
| RGF (proposed) | edge deletion | 97.28 ± 0.32 | 97.24 ± 0.21 | 97.24 ± 0.14 | 96.97 ± 0.23 | 96.49 ± 0.29 | -0.80 |
| **Cresci-2015 (3 seeds)** |  |  |  |  |  |  |  |
| LightGBM | edge deletion | 98.02 ± 0.36 | 98.02 ± 0.36 | 98.02 ± 0.36 | 98.02 ± 0.36 | 98.02 ± 0.36 | +0.00 |
| GCN | edge deletion | 97.46 ± 0.36 | 97.42 ± 0.39 | 97.25 ± 0.43 | 97.05 ± 0.61 | 96.62 ± 0.48 | -0.84 |
| GraphSAGE | edge deletion | 97.96 ± 0.33 | 97.84 ± 0.38 | 97.67 ± 0.33 | 97.39 ± 0.51 | 96.96 ± 0.44 | -1.00 |
| GAT | edge deletion | 97.43 ± 0.24 | 97.27 ± 0.29 | 97.06 ± 0.32 | 96.86 ± 0.43 | 96.21 ± 0.35 | -1.22 |
| R-GCN (BotRGCN-style) | edge deletion | 97.85 ± 0.46 | 97.78 ± 0.51 | 97.63 ± 0.41 | 97.35 ± 0.44 | 96.99 ± 0.47 | -0.87 |
| Hybrid P1-h | edge deletion | 98.33 ± 0.16 | 98.33 ± 0.11 | 98.29 ± 0.26 | 98.31 ± 0.29 | 98.19 ± 0.37 | -0.14 |
| RGF (proposed) | edge deletion | 98.07 ± 0.07 | 98.00 ± 0.20 | 97.83 ± 0.21 | 97.62 ± 0.36 | 97.31 ± 0.37 | -0.76 |
| **Cresci-2017 (3 seeds)** |  |  |  |  | — |  |  |
| LightGBM | edge injection | 98.24 ± 0.16 | 98.24 ± 0.16 | 98.24 ± 0.16 | — | 98.24 ± 0.16 | +0.00 |
| GCN | edge injection | 95.48 ± 0.20 | 95.11 ± 0.20 | 94.44 ± 0.06 | — | 92.47 ± 0.45 | -3.01 |
| GraphSAGE | edge injection | 98.17 ± 0.09 | 97.98 ± 0.18 | 97.76 ± 0.12 | — | 97.36 ± 0.10 | -0.82 |
| GAT | edge injection | 95.84 ± 0.11 | 95.18 ± 0.37 | 94.57 ± 0.35 | — | 93.33 ± 0.25 | -2.51 |
| Hybrid P1-h | edge injection | 97.64 ± 0.05 | 97.60 ± 0.11 | 97.54 ± 0.16 | — | 97.20 ± 0.21 | -0.44 |
| RGF (proposed) | edge injection | 97.28 ± 0.32 | 97.17 ± 0.39 | 97.00 ± 0.22 | — | 96.64 ± 0.44 | -0.64 |
| **Cresci-2015 (3 seeds)** |  |  |  |  | — |  |  |
| LightGBM | edge injection | 98.02 ± 0.36 | 98.02 ± 0.36 | 98.02 ± 0.36 | — | 98.02 ± 0.36 | +0.00 |
| GCN | edge injection | 97.46 ± 0.36 | 93.10 ± 0.74 | 90.88 ± 0.93 | — | 86.78 ± 1.56 | -10.68 |
| GraphSAGE | edge injection | 97.96 ± 0.33 | 95.59 ± 0.52 | 93.53 ± 0.58 | — | 93.03 ± 0.54 | -4.94 |
| GAT | edge injection | 97.43 ± 0.24 | 93.54 ± 0.44 | 90.56 ± 1.23 | — | 86.66 ± 2.31 | -10.77 |
| R-GCN (BotRGCN-style) | edge injection | 97.85 ± 0.46 | 96.35 ± 0.57 | 94.96 ± 1.08 | — | 93.65 ± 0.39 | -4.20 |
| Hybrid P1-h | edge injection | 98.33 ± 0.16 | 97.70 ± 0.20 | 96.87 ± 0.62 | — | 95.89 ± 0.98 | -2.44 |
| RGF (proposed) | edge injection | 98.07 ± 0.07 | 95.63 ± 0.29 | 94.49 ± 0.54 | — | 93.61 ± 0.63 | -4.46 |

*Edge deletion removes each edge incident to a test account with probability ρ; edge injection adds round(ρ × degree) random edges per test account (ρ ∈ {0, 25, 50, 100}%). LightGBM uses no graph and is unaffected. Cresci-2017 edges come from the metadata kNN graph. Mean ± SD over seeds 0-2.*

### TABLE 8. Bot-class F1 (%) when a fraction ρ of test bots receive the median metadata of training humans

| Method | ρ = 0% | ρ = 25% | ρ = 50% | ρ = 100% | Δ (100% − 0%) | MCC at ρ = 100% |
|---|---|---|---|---|---|---|
| **Cresci-2017 (3 seeds)** |  |  |  |  |  |  |
| LightGBM | 98.24 ± 0.16 | 88.14 ± 0.40 | 75.51 ± 0.11 | 40.17 ± 1.19 | -58.07 | 24.22 ± 0.76 |
| GCN | 95.48 ± 0.20 | 95.43 ± 0.20 | 95.29 ± 0.30 | 94.82 ± 0.47 | -0.65 | 79.57 ± 1.48 |
| GraphSAGE | 98.17 ± 0.09 | 97.51 ± 0.44 | 96.05 ± 1.33 | 90.25 ± 4.50 | -7.92 | 70.96 ± 9.59 |
| GAT | 95.84 ± 0.11 | 95.88 ± 0.16 | 95.94 ± 0.09 | 95.99 ± 0.15 | +0.15 | 83.47 ± 0.67 |
| Hybrid P1-h | 97.64 ± 0.05 | 92.55 ± 2.46 | 86.37 ± 6.00 | 68.88 ± 17.52 | -28.76 | 43.84 ± 15.19 |
| RGF (proposed) | 97.28 ± 0.32 | 96.94 ± 0.27 | 96.39 ± 0.22 | 94.47 ± 0.60 | -2.82 | 79.13 ± 1.57 |
| Concat fusion | 97.87 ± 0.09 | 93.26 ± 1.91 | 85.26 ± 5.41 | 59.54 ± 15.95 | -38.33 | 34.59 ± 12.68 |
| w/o gate (mean) | 97.48 ± 0.34 | 96.61 ± 0.42 | 95.05 ± 0.60 | 88.89 ± 2.78 | -8.59 | 65.80 ± 6.47 |
| w/o modality dropout | 97.24 ± 0.39 | 97.11 ± 0.28 | 96.87 ± 0.40 | 95.96 ± 0.52 | -1.28 | 83.72 ± 1.67 |
| **Cresci-2015 (3 seeds)** |  |  |  |  |  |  |
| LightGBM | 98.02 ± 0.36 | 83.66 ± 0.15 | 64.91 ± 0.35 | 0.00 ± 0.00 | -98.02 | -14.74 ± 3.37 |
| GCN | 97.46 ± 0.36 | 88.77 ± 0.96 | 77.59 ± 2.28 | 46.57 ± 3.08 | -50.89 | 29.87 ± 3.24 |
| GraphSAGE | 97.96 ± 0.33 | 89.29 ± 0.76 | 78.07 ± 2.61 | 46.38 ± 4.19 | -51.58 | 29.85 ± 3.51 |
| GAT | 97.43 ± 0.24 | 90.89 ± 1.04 | 81.83 ± 2.56 | 60.71 ± 4.82 | -36.73 | 39.79 ± 5.04 |
| R-GCN (BotRGCN-style) | 97.85 ± 0.46 | 87.92 ± 0.79 | 74.17 ± 1.93 | 35.69 ± 5.13 | -62.17 | 23.14 ± 4.10 |
| Hybrid P1-h | 98.33 ± 0.16 | 83.95 ± 0.16 | 65.19 ± 0.31 | 0.49 ± 0.84 | -97.85 | -10.71 ± 4.99 |
| RGF (proposed) | 98.07 ± 0.07 | 90.52 ± 0.74 | 80.16 ± 2.31 | 55.25 ± 4.79 | -42.82 | 38.29 ± 4.57 |
| Concat fusion | 98.42 ± 0.15 | 87.17 ± 0.44 | 71.39 ± 1.65 | 26.68 ± 4.37 | -71.74 | 18.75 ± 3.19 |
| w/o gate (mean) | 98.34 ± 0.08 | 89.73 ± 1.46 | 78.54 ± 2.67 | 48.83 ± 8.40 | -49.52 | 34.34 ± 5.99 |
| w/o modality dropout | 97.95 ± 0.25 | 90.27 ± 0.82 | 80.11 ± 2.09 | 54.70 ± 3.92 | -43.25 | 37.18 ± 4.20 |

*The attacker knows which test accounts are bots and the human median (white-box on the feature, not the model). The graph is not rebuilt; on Cresci-2017 this leaves kNN neighbours computed from the original metadata. Mean ± SD over seeds 0-2.*

### TABLE 9. Transductive versus inductive evaluation (%, mean ± SD over 5 seeds)

| Method | F1 trans. | F1 induct. | MCC trans. | MCC induct. | ΔMCC |
|---|---|---|---|---|---|
| **Cresci-2017 (5 seeds)** |  |  |  |  |  |
| LightGBM | 98.34 ± 0.18 | 98.34 ± 0.18 | 93.34 ± 0.69 | 93.34 ± 0.69 | +0.00 |
| GCN | 95.51 ± 0.15 | 95.11 ± 0.18 | 81.61 ± 0.47 | 79.61 ± 0.69 | -2.00 |
| GraphSAGE | 98.22 ± 0.09 | 98.08 ± 0.09 | 92.61 ± 0.40 | 92.01 ± 0.36 | -0.59 |
| GAT | 95.76 ± 0.17 | 95.11 ± 0.27 | 82.58 ± 0.84 | 79.62 ± 1.29 | -2.96 |
| Hybrid P1-h | 97.71 ± 0.18 | 97.74 ± 0.15 | 90.73 ± 0.70 | 90.78 ± 0.62 | +0.05 |
| RGF (proposed) | 97.28 ± 0.28 | 97.28 ± 0.15 | 88.59 ± 1.20 | 88.54 ± 0.64 | -0.05 |
| **Cresci-2015 (5 seeds)** |  |  |  |  |  |
| LightGBM | 98.08 ± 0.28 | 98.08 ± 0.28 | 94.77 ± 0.80 | 94.77 ± 0.80 | +0.00 |
| GCN | 97.31 ± 0.43 | 97.31 ± 0.39 | 92.61 ± 1.21 | 92.61 ± 1.10 | +0.00 |
| GraphSAGE | 97.89 ± 0.26 | 97.77 ± 0.33 | 94.20 ± 0.72 | 93.88 ± 0.90 | -0.33 |
| GAT | 97.29 ± 0.33 | 97.13 ± 0.37 | 92.51 ± 0.92 | 92.07 ± 1.05 | -0.44 |
| Hybrid P1-h | 98.39 ± 0.16 | 98.29 ± 0.09 | 95.63 ± 0.45 | 95.34 ± 0.25 | -0.29 |
| RGF (proposed) | 97.98 ± 0.15 | 97.64 ± 0.46 | 94.49 ± 0.39 | 93.60 ± 1.27 | -0.88 |

*Inductive: test accounts are absent from the training graph and send no messages to seen accounts. Tree models do not use the graph, so both columns are identical. R-GCN, XGBoost and random forest were not run inductively.*

### TABLE 10. Cross-dataset evaluation (%, mean ± SD over seeds 0-2)

| Method | Accuracy | Human FPR | F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| **Train Cresci-2015 → test Cresci-2017 (1507 bots, 695 humans, 672 shared accounts removed)** |  |  |  |  |  |
| LightGBM | 57.18 ± 2.73 | 2.49 ± 0.51 | 54.91 ± 4.02 | 37.66 ± 2.92 | 68.91 ± 4.03 |
| GCN | 67.65 ± 1.60 | 4.89 ± 0.63 | 69.78 ± 2.01 | 47.72 ± 1.83 | 81.38 ± 3.31 |
| GraphSAGE | 60.73 ± 5.58 | 2.64 ± 1.09 | 59.86 ± 8.06 | 41.53 ± 5.44 | 79.08 ± 1.55 |
| Hybrid P1-h | 55.84 ± 0.61 | 3.55 ± 1.58 | 53.24 ± 0.83 | 35.30 ± 1.52 | 67.56 ± 1.28 |
| Hybrid P1-p | 55.71 ± 2.84 | 3.31 ± 0.88 | 52.89 ± 4.29 | 35.34 ± 3.16 | 72.25 ± 1.95 |
| Hybrid P3-h | 58.75 ± 2.09 | 4.94 ± 1.88 | 58.00 ± 2.35 | 37.43 ± 3.78 | 77.51 ± 6.12 |
| Hybrid P3-p | 66.61 ± 1.93 | 4.60 ± 1.14 | 68.45 ± 2.18 | 46.69 ± 2.74 | 78.48 ± 2.14 |
| RGF (proposed) | 66.24 ± 1.26 | 3.26 ± 0.79 | 67.72 ± 1.62 | 47.17 ± 1.39 | 81.40 ± 2.83 |
| **Train Cresci-2017 → test Cresci-2015 (0 bots, 390 humans, 671 shared accounts removed)** |  |  |  |  |  |
| LightGBM | 46.50 ± 3.34 | 53.50 ± 3.34 | — | — | — |
| GCN | 14.79 ± 2.14 | 85.21 ± 2.14 | — | — | — |
| GraphSAGE | 21.28 ± 3.85 | 78.72 ± 3.85 | — | — | — |
| Hybrid P1-h | 29.66 ± 0.90 | 70.34 ± 0.90 | — | — | — |
| Hybrid P1-p | 29.23 ± 3.45 | 70.77 ± 3.45 | — | — | — |
| Hybrid P3-h | 23.42 ± 0.65 | 76.58 ± 0.65 | — | — | — |
| Hybrid P3-p | 16.92 ± 3.70 | 83.08 ± 3.70 | — | — | — |
| RGF (proposed) | 17.52 ± 1.71 | 82.48 ± 1.71 | — | — | — |

*Human FPR = humans classified as bots / all humans. Removing accounts shared by the two datasets (by user ID) removes every Cresci-2015 test bot, so the Cresci-2017 → Cresci-2015 direction measures only false positives on 390 humans; F1, MCC and ROC-AUC are undefined there.*

### TABLE 11. Calibration on the test split (mean ± SD over 5 seeds)

| Method | ECE C-17 (%) | Brier C-17 | ECE C-15 (%) | Brier C-15 |
|---|---|---|---|---|
| LightGBM | 0.95 ± 0.23 | 0.0202 ± 0.0014 | 1.15 ± 0.21 | 0.0197 ± 0.0029 |
| XGBoost | 0.87 ± 0.13 | 0.0194 ± 0.0012 | 1.26 ± 0.34 | 0.0192 ± 0.0030 |
| Random forest | 8.72 ± 0.24 | 0.0402 ± 0.0009 | 10.42 ± 0.49 | 0.0360 ± 0.0019 |
| GCN | 1.57 ± 0.15 | 0.0519 ± 0.0019 | 1.71 ± 0.33 | 0.0268 ± 0.0054 |
| GraphSAGE | 1.02 ± 0.29 | 0.0235 ± 0.0011 | 1.46 ± 0.23 | 0.0217 ± 0.0022 |
| GAT | 1.27 ± 0.31 | 0.0502 ± 0.0015 | 1.74 ± 0.32 | 0.0271 ± 0.0029 |
| R-GCN (BotRGCN-style) | — | — | 1.66 ± 0.27 | 0.0216 ± 0.0041 |
| Hybrid P1-h | 1.10 ± 0.34 | 0.0275 ± 0.0014 | 1.00 ± 0.20 | 0.0170 ± 0.0016 |
| RGF (proposed) | 1.17 ± 0.15 | 0.0330 ± 0.0021 | 1.71 ± 0.38 | 0.0193 ± 0.0013 |
| RGF before temperature scaling | 1.05 ± 0.17 | 0.0329 ± 0.0021 | 1.54 ± 0.37 | 0.0191 ± 0.0013 |
| RGF temperature T | 1.01 ± 0.09 |  | 0.98 ± 0.16 |  |

*ECE: 15 equal-width bins of predicted-class confidence. Only RGF applies temperature scaling, with T chosen on the same validation split used for early stopping; on both datasets ECE and Brier score are slightly higher after scaling, a change within seed variation (paired t, p = 0.38 and 0.27).*

### TABLE 12. Selective prediction: error (%) on the most confident fraction of test accounts

| Method | Err 100% C-17 | Err 90% C-17 | Err 80% C-17 | Err 100% C-15 | Err 90% C-15 | Err 80% C-15 |
|---|---|---|---|---|---|---|
| LightGBM | 2.50 ± 0.26 | 0.70 ± 0.11 | 0.29 ± 0.09 | 2.43 ± 0.37 | 0.77 ± 0.18 | 0.49 ± 0.35 |
| XGBoost | 2.43 ± 0.22 | 0.63 ± 0.09 | 0.25 ± 0.08 | 2.41 ± 0.31 | 0.71 ± 0.39 | 0.45 ± 0.33 |
| Random forest | 3.13 ± 0.24 | 0.87 ± 0.12 | 0.51 ± 0.10 | 2.45 ± 0.19 | 0.75 ± 0.29 | 0.61 ± 0.27 |
| GCN | 6.79 ± 0.21 | 3.80 ± 0.23 | 2.26 ± 0.22 | 3.43 ± 0.56 | 1.36 ± 0.58 | 0.78 ± 0.59 |
| GraphSAGE | 2.71 ± 0.14 | 1.16 ± 0.15 | 0.63 ± 0.13 | 2.70 ± 0.33 | 0.65 ± 0.23 | 0.47 ± 0.26 |
| GAT | 6.42 ± 0.27 | 3.53 ± 0.18 | 2.14 ± 0.29 | 3.49 ± 0.44 | 0.92 ± 0.20 | 0.49 ± 0.26 |
| R-GCN (BotRGCN-style) | — | — | — | 2.58 ± 0.57 | 0.80 ± 0.35 | 0.57 ± 0.39 |
| Hybrid P1-h | 3.45 ± 0.26 | 1.25 ± 0.12 | 0.61 ± 0.17 | 2.04 ± 0.21 | 0.71 ± 0.26 | 0.42 ± 0.37 |
| RGF (proposed) | 4.15 ± 0.43 | 1.86 ± 0.23 | 1.19 ± 0.21 | 2.56 ± 0.18 | 0.42 ± 0.25 | 0.31 ± 0.27 |

*Accounts ranked by |p − 0.5|; Err 100% = 100 − accuracy. Temperature scaling does not change this ranking. Mean ± SD over 5 seeds.*

### TABLE 13. Reliability-gate weights of RGF and accuracy of each expert alone

| Group | n | w meta | w text | w graph |
|---|---|---|---|---|
| **Cresci-2017 — mean gate weight (seeds 0-2)** |  |  |  |  |
| All test accounts | 2874 | 0.165 ± 0.005 | 0.277 ± 0.019 | 0.559 ± 0.023 |
| Bots | 2179 | 0.187 ± 0.006 | 0.231 ± 0.013 | 0.581 ± 0.019 |
| Humans | 695 | 0.093 ± 0.001 | 0.419 ± 0.037 | 0.488 ± 0.037 |
| Text missing | 1161 | 0.263 ± 0.012 | 0.000 ± 0.000 | 0.737 ± 0.012 |
| **Cresci-2017 — each expert alone, accounts where its modality exists: F1 / MCC (%)** |  |  |  |  |
| Expert alone |  | 93.28 / 72.10 (n = 2874) | 86.50 / 64.33 (n = 1713) | 95.47 / 82.08 (n = 2874) |
| **Cresci-2015 — mean gate weight (seeds 0-2)** |  |  |  |  |
| All test accounts | 1061 | 0.453 ± 0.040 | 0.290 ± 0.015 | 0.257 ± 0.026 |
| Bots | 671 | 0.549 ± 0.046 | 0.311 ± 0.035 | 0.141 ± 0.012 |
| Humans | 390 | 0.289 ± 0.036 | 0.253 ± 0.041 | 0.458 ± 0.052 |
| Text missing | 238 | 0.905 ± 0.007 | 0.000 ± 0.000 | 0.095 ± 0.007 |
| No graph link | 676 | 0.606 ± 0.042 | 0.394 ± 0.042 | 0.000 ± 0.000 |
| **Cresci-2015 — each expert alone, accounts where its modality exists: F1 / MCC (%)** |  |  |  |  |
| Expert alone |  | 96.76 / 91.02 (n = 1061) | 87.55 / 70.94 (n = 823) | 98.78 / 98.12 (n = 385) |

*Weights are averaged over accounts, then over seeds. Unavailable experts receive weight 0 by construction (text-missing and isolated rows). Gate weights describe what the model relies on, not which modality is correct.*

### TABLE 14. Computational cost on CPU (mean ± SD over 5 seeds, transductive)

| Method | Fit (s) C-17 | Infer. (ms/1k) C-17 | Size C-17 | Fit (s) C-15 | Infer. (ms/1k) C-15 | Size C-15 |
|---|---|---|---|---|---|---|
| LightGBM | 11.2 ± 1.9 | 9.9 ± 2.3 | 196.6 trees | 4.0 ± 0.1 | 4.5 ± 0.3 | 89.8 trees |
| XGBoost | 18.5 ± 3.7 | 5.0 ± 0.9 | 334.0 trees | 6.3 ± 0.9 | 3.3 ± 0.3 | 137.2 trees |
| Random forest | 30.4 ± 0.5 | 41.1 ± 1.4 | 500.0 trees | 11.4 ± 0.1 | 46.1 ± 1.2 | 500.0 trees |
| GCN | 23.1 ± 6.2 | 4.9 ± 0.1 | 25,026 | 2.5 ± 0.8 | 2.1 ± 0.7 | 25,026 |
| GraphSAGE | 20.1 ± 2.0 | 6.6 ± 0.1 | 49,986 | 4.3 ± 0.9 | 2.3 ± 0.0 | 49,986 |
| GAT | 107.5 ± 2.5 | 7.1 ± 0.2 | 25,158 | 17.7 ± 1.6 | 2.1 ± 0.2 | 25,158 |
| R-GCN (BotRGCN-style) | — | — | — | 3.6 ± 0.6 | 2.8 ± 0.3 | 53,890 |
| Hybrid P1-h | 20.3 ± 0.6 | 10.4 ± 0.6 | 25,026 | 6.7 ± 0.3 | 7.4 ± 0.8 | 25,026 |
| RGF (proposed) | 43.2 ± 5.8 | 10.2 ± 0.2 | 38,089* | 9.4 ± 0.8 | 4.9 ± 0.2 | 38,089* |

*Kaggle CPU session; fit time includes early stopping; inference in milliseconds per 1,000 accounts. Size = trainable parameters for neural models and number of trees for tree models (not comparable units). *Parameters used by RGF; the checkpoint registers 50,571 because it also holds a 12,482-parameter head used only by the concatenation ablation.*

### TABLE 15. Positioning against related work

| Study | Modalities | Approach | Reported finding | Relation to this work |
|---|---|---|---|---|
| BotRGCN [Feng et al., 2021] | metadata, text, follow graph | relational GCN | TwiBot-20 acc 0.8462, F1 0.8707 (reported) | R-GCN baseline re-implemented here |
| RGT [Feng et al., 2022] | metadata, text, heterogeneous graph | relation-aware graph transformer | outperforms prior methods on TwiBot-20 | not re-run |
| BotMoE [Liu et al., 2023] | metadata, text, graph | community-aware modality experts | most robust to feature manipulation in its study | closest architecture; not re-run |
| BotBuster [Ng & Carley, 2023] | per-platform account fields | mixture of experts | handles incomplete account information | missing-input motivation |
| BotCHF [Wang et al., 2026] | multi-modal | camouflage-aware fusion | fusion weights differ across datasets | per-account weighting is not new |
| LMBot [Cai et al., 2024] | text, graph (distilled) | graph-to-LM distillation | graph-less deployment at state-of-the-art accuracy | alternative to graph availability |
| Hays et al., 2023 | simple features | shallow decision trees | near state-of-the-art; datasets do not generalise | consistent with our strong tree baselines |
| Echeverría et al., 2018 | account features | leave-one-botnet-out | >97% accuracy does not transfer to unseen bot classes | consistent with our cross-dataset result |
| Mukherjee et al., 2026 | graph | optimal-transport edge attacks | attacks bypass five defences | edge injection is a real threat |
| This work | metadata, text, graph | availability-masked gated experts | see Tables 4-14 | same protocol for all baselines |

*Findings are summarised from the verified reference library (VERIFIED_REFERENCES.csv); numbers from other papers use different datasets, splits and protocols and are not comparable with Tables 4-14, so no numeric ranking is given.*
