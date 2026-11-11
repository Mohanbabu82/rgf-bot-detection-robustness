# Paper tables (generated from result files; do not edit numbers by hand)

Provenance: THIS-WORK = new experiments; REPRODUCED-CORRECTED = BERT/SBERT + GCN + LightGBM hybrid baseline implemented here under the leakage-free protocol. All numbers come from this run. Metrics are bot-class unless stated; mean ± SD over seeds.


## Cresci-2017


### T1 Data (Cresci-2017)

| Train | Validation | Test | Edge homophily (train) | Test accounts without graph links | Test accounts without text |
| --- | --- | --- | --- | --- | --- |
| 10057 | 1437 | 2874 | 0.894 | 0.0% | 40.6% |


### T2 Main comparison (Cresci-2017)

| Method | Provenance | Seeds | Acc | Prec | Rec | F1 | MCC | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM (meta+text) | THIS-WORK | 5 | 97.50 ± 0.26 | 99.09 ± 0.14 | 97.60 ± 0.30 | 98.34 ± 0.18 | 93.34 ± 0.69 | 99.41 ± 0.07 | 99.82 ± 0.03 |
| XGBoost (meta+text) | THIS-WORK | 5 | 97.57 ± 0.22 | 99.09 ± 0.13 | 97.70 ± 0.24 | 98.39 ± 0.15 | 93.51 ± 0.57 | 99.45 ± 0.06 | 99.82 ± 0.03 |
| Random Forest (meta+text) | THIS-WORK | 5 | 96.87 ± 0.24 | 98.23 ± 0.34 | 97.63 ± 0.23 | 97.93 ± 0.15 | 91.52 ± 0.67 | 99.14 ± 0.11 | 99.71 ± 0.06 |
| GCN | THIS-WORK | 5 | 93.21 ± 0.21 | 95.79 ± 0.26 | 95.23 ± 0.48 | 95.51 ± 0.15 | 81.61 ± 0.47 | 97.10 ± 0.25 | 98.98 ± 0.16 |
| GraphSAGE | THIS-WORK | 5 | 97.29 ± 0.14 | 98.13 ± 0.22 | 98.30 ± 0.12 | 98.22 ± 0.09 | 92.61 ± 0.40 | 99.08 ± 0.12 | 99.69 ± 0.05 |
| GAT | THIS-WORK | 5 | 93.58 ± 0.27 | 95.96 ± 0.40 | 95.56 ± 0.10 | 95.76 ± 0.17 | 82.58 ± 0.84 | 97.00 ± 0.25 | 98.87 ± 0.14 |
| Reconstructed hybrid (P1, hidden) [best of 4 by validation] | REPRODUCED-CORRECTED | 5 | 96.55 ± 0.26 | 98.27 ± 0.21 | 97.15 ± 0.27 | 97.71 ± 0.18 | 90.73 ± 0.70 | 99.00 ± 0.23 | 99.64 ± 0.14 |
| RGF (proposed) | THIS-WORK | 5 | 95.85 ± 0.43 | 96.82 ± 0.37 | 97.74 ± 0.27 | 97.28 ± 0.28 | 88.59 ± 1.20 | 98.48 ± 0.13 | 99.47 ± 0.06 |


### T3 Ablation of the proposed model (Cresci-2017)

| Variant | Seeds | Val F1 | F1 | MCC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| RGF (proposed) | 5 | 97.25 ± 0.33 | 97.28 ± 0.28 | 88.59 ± 1.20 | 98.48 ± 0.13 |
| RGF concat fusion | 5 | 97.86 ± 0.39 | 97.89 ± 0.08 | 91.27 ± 0.26 | 99.00 ± 0.17 |
| RGF w/o auxiliary losses | 5 | 97.67 ± 0.38 | 97.83 ± 0.22 | 91.04 ± 0.97 | 98.67 ± 0.37 |
| RGF w/o gate (mean of experts) | 5 | 97.40 ± 0.40 | 97.49 ± 0.24 | 89.47 ± 1.03 | 98.80 ± 0.13 |
| RGF w/o graph expert | 5 | 96.43 ± 0.50 | 96.62 ± 0.19 | 85.50 ± 0.82 | 98.42 ± 0.16 |
| RGF w/o metadata expert | 5 | 97.12 ± 0.41 | 97.15 ± 0.31 | 88.10 ± 1.30 | 98.20 ± 0.13 |
| RGF w/o modality dropout | 5 | 97.22 ± 0.40 | 97.31 ± 0.30 | 88.68 ± 1.29 | 98.46 ± 0.15 |
| RGF w/o text expert | 5 | 95.66 ± 0.87 | 95.76 ± 0.18 | 82.63 ± 0.62 | 97.28 ± 0.27 |


### T3b Modality ablation (same split and protocol for every row) (Cresci-2017)

| Method | Metadata | Text | Graph | Architecture | Seeds | F1 | MCC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM (meta) | yes |  |  | tree model | 5 | 97.44 ± 0.19 | 89.60 ± 0.75 | 98.79 ± 0.12 |
| LightGBM (text) |  | yes |  | tree model | 5 | 91.44 ± 0.39 | 63.68 ± 1.44 | 90.29 ± 0.82 |
| LightGBM (meta+text) | yes | yes |  | tree model | 5 | 98.34 ± 0.18 | 93.34 ± 0.69 | 99.41 ± 0.07 |
| GCN (meta) | yes |  | yes | GCN end-to-end | 5 | 94.40 ± 0.39 | 77.38 ± 1.52 | 95.96 ± 0.30 |
| GCN (text) |  | yes | yes | GCN end-to-end | 5 | 95.43 ± 0.19 | 81.25 ± 0.65 | 96.83 ± 0.31 |
| GCN | yes | yes | yes | GCN end-to-end | 5 | 95.51 ± 0.15 | 81.61 ± 0.47 | 97.10 ± 0.25 |
| Reconstructed hybrid (P3, hidden) | via GCN | via GCN | yes | P3: GCN 64-d embedding -> LightGBM | 5 | 95.23 ± 0.33 | 80.64 ± 1.14 | 96.78 ± 0.30 |
| Reconstructed hybrid (P3, prob) | via GCN | via GCN | yes | P3: GCN 2-d probabilities -> LightGBM | 5 | 95.38 ± 0.38 | 81.16 ± 1.60 | 96.64 ± 0.33 |
| Reconstructed hybrid (P1, hidden) | yes | yes | yes | P1: meta + text + GCN 64-d -> LightGBM | 5 | 97.71 ± 0.18 | 90.73 ± 0.70 | 99.00 ± 0.23 |
| Reconstructed hybrid (P1, prob) | yes | yes | yes | P1: meta + text + GCN 2-d -> LightGBM | 5 | 97.75 ± 0.24 | 90.86 ± 0.93 | 99.06 ± 0.25 |
| RGF (proposed) | yes | yes | yes | reliability-gated experts | 5 | 97.28 ± 0.28 | 88.59 ± 1.20 | 98.48 ± 0.13 |


### T4 Robustness, bot-class F1 (%) (Cresci-2017)

| perturbation | level | GAT | GCN | GraphSAGE | LightGBM (meta+text) | RGF (proposed) | RGF concat fusion | RGF w/o gate (mean of experts) | RGF w/o modality dropout | Reconstructed hybrid (P1, hidden) | Reconstructed hybrid (P1, prob) | Reconstructed hybrid (P3, hidden) | Reconstructed hybrid (P3, prob) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_edges | 0.0 | 95.84 | 95.48 | 98.17 | 98.24 | 97.28 | 97.87 | 97.48 | 97.24 | 97.64 | 97.61 | 95.17 | 95.26 |
| drop_edges | 0.25 | 95.75 | 95.57 | 98.17 | 98.24 | 97.24 | 97.87 | 97.48 | 97.27 | 97.58 | 97.7 | 95.01 | 95.44 |
| drop_edges | 0.5 | 95.48 | 95.6 | 98.06 | 98.24 | 97.24 | 97.83 | 97.43 | 97.16 | 97.54 | 97.82 | 94.77 | 95.29 |
| drop_edges | 0.75 | 95.39 | 95.53 | 97.5 | 98.24 | 96.97 | 97.71 | 97.15 | 96.77 | 97.33 | 97.7 | 94.53 | 95.5 |
| drop_edges | 1.0 | 92.12 | 92.37 | 95.02 | 98.24 | 96.49 | 97.5 | 96.59 | 93.15 | 96.33 | 96.13 | 93.28 | 92.54 |
| metadata_mimicry | 0.0 | 95.84 | 95.48 | 98.17 | 98.24 | 97.28 | 97.87 | 97.48 | 97.24 | 97.64 | 97.61 | 95.17 | 95.26 |
| metadata_mimicry | 0.25 | 95.88 | 95.43 | 97.51 | 88.14 | 96.94 | 93.26 | 96.61 | 97.11 | 92.55 | 93.11 | 95.04 | 95.2 |
| metadata_mimicry | 0.5 | 95.94 | 95.29 | 96.05 | 75.51 | 96.39 | 85.26 | 95.05 | 96.87 | 86.37 | 86.58 | 94.49 | 95.14 |
| metadata_mimicry | 1.0 | 95.99 | 94.82 | 90.25 | 40.17 | 94.47 | 59.54 | 88.89 | 95.96 | 68.88 | 65.45 | 93.35 | 94.52 |
| missing_text | 0.0 | 95.84 | 95.48 | 98.17 | 98.24 | 97.28 | 97.87 | 97.48 | 97.24 | 97.64 | 97.61 | 95.17 | 95.26 |
| missing_text | 0.25 | 95.63 | 95.34 | 96.44 | 96.4 | 96.75 | 97.26 | 97.07 | 96.67 | 96.76 | 96.51 | 94.93 | 95.18 |
| missing_text | 0.5 | 95.34 | 95.01 | 94.62 | 94.25 | 96.26 | 96.64 | 96.35 | 96.1 | 95.86 | 95.36 | 94.74 | 94.91 |
| missing_text | 0.75 | 95.07 | 94.93 | 92.48 | 92.24 | 95.69 | 95.93 | 95.75 | 95.46 | 94.84 | 93.76 | 94.56 | 94.73 |
| missing_text | 1.0 | 94.71 | 94.62 | 90.28 | 89.75 | 95.04 | 95.22 | 95.1 | 94.76 | 93.62 | 92.08 | 94.12 | 94.55 |
| noisy_edges | 0.0 | 95.84 | 95.48 | 98.17 | 98.24 | 97.28 | 97.87 | 97.48 | 97.24 | 97.64 | 97.61 | 95.17 | 95.26 |
| noisy_edges | 0.25 | 95.18 | 95.11 | 97.98 | 98.24 | 97.17 | 97.88 | 97.47 | 97.07 | 97.6 | 97.73 | 94.47 | 94.9 |
| noisy_edges | 0.5 | 94.57 | 94.44 | 97.76 | 98.24 | 97.0 | 97.79 | 97.43 | 96.74 | 97.54 | 97.77 | 93.67 | 93.84 |
| noisy_edges | 1.0 | 93.33 | 92.47 | 97.36 | 98.24 | 96.64 | 97.61 | 97.19 | 96.09 | 97.2 | 97.53 | 91.52 | 91.95 |


### T5 Unseen accounts: transductive vs inductive (Cresci-2017)

| Method | Transductive F1 | Inductive F1 | Transductive MCC | Inductive MCC |
| --- | --- | --- | --- | --- |
| GAT | 95.76 ± 0.17 | 95.11 ± 0.27 | 82.58 ± 0.84 | 79.62 ± 1.29 |
| GCN | 95.51 ± 0.15 | 95.11 ± 0.18 | 81.61 ± 0.47 | 79.61 ± 0.69 |
| GraphSAGE | 98.22 ± 0.09 | 98.08 ± 0.09 | 92.61 ± 0.40 | 92.01 ± 0.36 |
| LightGBM (meta+text) | 98.34 ± 0.18 | 98.34 ± 0.18 | 93.34 ± 0.69 | 93.34 ± 0.69 |
| RGF (proposed) | 97.28 ± 0.28 | 97.28 ± 0.15 | 88.59 ± 1.20 | 88.54 ± 0.64 |
| Reconstructed hybrid (P1, hidden) | 97.71 ± 0.18 | 97.74 ± 0.15 | 90.73 ± 0.70 | 90.78 ± 0.62 |
| Reconstructed hybrid (P1, prob) | 97.75 ± 0.24 | 97.76 ± 0.11 | 90.86 ± 0.93 | 90.88 ± 0.42 |
| Reconstructed hybrid (P3, hidden) | 95.23 ± 0.33 | 95.25 ± 0.27 | 80.64 ± 1.14 | 80.47 ± 1.11 |
| Reconstructed hybrid (P3, prob) | 95.38 ± 0.38 | 95.04 ± 0.19 | 81.16 ± 1.60 | 79.52 ± 1.04 |


### T6 Calibration and selective prediction (Cresci-2017)

| Method | ECE | Brier | Error @80% coverage | Error @90% coverage | ECE before temperature scaling |
| --- | --- | --- | --- | --- | --- |
| GAT | 1.27 ± 0.31 | 0.0502 ± 0.0015 | 2.14 ± 0.29 | 3.53 ± 0.18 | — |
| GCN | 1.57 ± 0.15 | 0.0519 ± 0.0019 | 2.26 ± 0.22 | 3.80 ± 0.23 | — |
| GCN (meta) | 1.83 ± 0.34 | 0.0630 ± 0.0029 | 3.07 ± 0.41 | 5.23 ± 0.52 | — |
| GCN (text) | 1.52 ± 0.10 | 0.0533 ± 0.0019 | 2.31 ± 0.32 | 3.99 ± 0.20 | — |
| GraphSAGE | 1.02 ± 0.29 | 0.0235 ± 0.0011 | 0.63 ± 0.13 | 1.16 ± 0.15 | — |
| LightGBM (meta) | 1.03 ± 0.32 | 0.0311 ± 0.0018 | 0.83 ± 0.13 | 1.65 ± 0.23 | — |
| LightGBM (meta+text) | 0.95 ± 0.23 | 0.0202 ± 0.0014 | 0.29 ± 0.09 | 0.70 ± 0.11 | — |
| LightGBM (text) | 1.48 ± 0.52 | 0.0956 ± 0.0040 | 7.02 ± 0.51 | 9.70 ± 0.62 | — |
| RGF (proposed) | 1.17 ± 0.15 | 0.0330 ± 0.0021 | 1.19 ± 0.21 | 1.86 ± 0.23 | 1.05 ± 0.17 |
| Random Forest (meta+text) | 8.72 ± 0.24 | 0.0402 ± 0.0009 | 0.51 ± 0.10 | 0.87 ± 0.12 | — |
| Reconstructed hybrid (P1, hidden) | 1.10 ± 0.34 | 0.0275 ± 0.0014 | 0.61 ± 0.17 | 1.25 ± 0.12 | — |
| Reconstructed hybrid (P1, prob) | 1.21 ± 0.09 | 0.0268 ± 0.0019 | 0.61 ± 0.22 | 1.08 ± 0.29 | — |
| Reconstructed hybrid (P3, hidden) | 1.47 ± 0.32 | 0.0557 ± 0.0033 | 2.45 ± 0.20 | 4.19 ± 0.44 | — |
| Reconstructed hybrid (P3, prob) | 1.57 ± 0.20 | 0.0540 ± 0.0030 | 2.52 ± 0.32 | 4.09 ± 0.46 | — |
| XGBoost (meta+text) | 0.87 ± 0.13 | 0.0194 ± 0.0012 | 0.25 ± 0.08 | 0.63 ± 0.09 | — |


### T9 Efficiency on CPU (Cresci-2017)

| Method | Fit time (s) | Inference (ms / 1k accounts) | Size (params or trees) |
| --- | --- | --- | --- |
| GAT | 107.5 | 7.1 | 25158 |
| GCN | 23.1 | 4.9 | 25026 |
| GCN (meta) | 22.6 | 4.2 | 450 |
| GCN (text) | 26.8 | 4.8 | 24770 |
| GraphSAGE | 20.1 | 6.6 | 49986 |
| LightGBM (meta) | 0.6 | 7.4 | 208 |
| LightGBM (meta+text) | 11.2 | 9.9 | 196 |
| LightGBM (text) | 13.5 | 12.6 | 248 |
| RGF (proposed) | 43.2 | 10.2 | 50571 |
| Random Forest (meta+text) | 30.4 | 41.1 | 500 |
| Reconstructed hybrid (P1, hidden) | 20.3 | 10.4 | 25026 |
| Reconstructed hybrid (P1, prob) | 19.9 | 10.4 | 25026 |
| Reconstructed hybrid (P3, hidden) | 14.5 | 6.9 | 25026 |
| Reconstructed hybrid (P3, prob) | 14.1 | 6.7 | 25026 |
| XGBoost (meta+text) | 18.5 | 5.0 | 334 |


### T10 Paired t-tests, RGF vs each method (bot-class F1, Holm-corrected) (Cresci-2017)

| Compared with | Seeds | Mean diff (pp) | Cohen d_z | p | p (Holm) | Significant (0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost (meta+text) | 5 | -1.11 | -5.2 | 0.0003 | 0.0044 | yes |
| LightGBM (meta+text) | 5 | -1.06 | -4.11 | 0.0008 | 0.0093 | yes |
| GraphSAGE | 5 | -0.94 | -3.13 | 0.0022 | 0.0220 | yes |
| Random Forest (meta+text) | 5 | -0.65 | -2.37 | 0.0061 | 0.0525 | no |
| RGF concat fusion | 5 | -0.61 | -2.4 | 0.0058 | 0.0525 | no |
| RGF w/o auxiliary losses | 5 | -0.55 | -1.86 | 0.0140 | 0.0983 | no |
| Reconstructed hybrid (P1, prob) | 5 | -0.47 | -1.25 | 0.0493 | 0.2610 | no |
| Reconstructed hybrid (P1, hidden) | 5 | -0.43 | -1.09 | 0.0712 | 0.2849 | no |
| RGF w/o gate (mean of experts) | 5 | -0.21 | -1.3 | 0.0435 | 0.2610 | no |
| LightGBM (meta) | 5 | -0.16 | -0.51 | 0.3187 | 0.6374 | no |
| RGF w/o modality dropout | 5 | -0.03 | -0.23 | 0.6272 | 0.6374 | no |
| RGF w/o metadata expert | 5 | 0.13 | 0.98 | 0.0946 | 0.2849 | no |
| RGF w/o graph expert | 5 | 0.65 | 4.48 | 0.0006 | 0.0073 | yes |
| GAT | 5 | 1.52 | 4.04 | 0.0008 | 0.0093 | yes |
| RGF w/o text expert | 5 | 1.52 | 8.82 | 0.0000 | 0.0007 | yes |
| GCN | 5 | 1.77 | 10.77 | 0.0000 | 0.0003 | yes |
| GCN (text) | 5 | 1.85 | 12.8 | 0.0000 | 0.0002 | yes |
| Reconstructed hybrid (P3, prob) | 5 | 1.9 | 6.84 | 0.0001 | 0.0016 | yes |
| Reconstructed hybrid (P3, hidden) | 5 | 2.05 | 8.63 | 0.0000 | 0.0007 | yes |
| GCN (meta) | 5 | 2.88 | 11.14 | 0.0000 | 0.0003 | yes |
| LightGBM (text) | 5 | 5.84 | 17.41 | 0.0000 | 0.0001 | yes |


### T10b Paired t-tests, RGF vs each method (MCC, Holm-corrected) (Cresci-2017)

| Compared with | Seeds | Mean diff (pp) | Cohen d_z | p | p (Holm) | Significant (0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost (meta+text) | 5 | -4.92 | -5.33 | 0.0003 | 0.0042 | yes |
| LightGBM (meta+text) | 5 | -4.75 | -4.39 | 0.0006 | 0.0072 | yes |
| GraphSAGE | 5 | -4.01 | -3.1 | 0.0023 | 0.0228 | yes |
| Random Forest (meta+text) | 5 | -2.93 | -2.6 | 0.0044 | 0.0394 | yes |
| RGF concat fusion | 5 | -2.68 | -2.37 | 0.0061 | 0.0488 | yes |
| RGF w/o auxiliary losses | 5 | -2.44 | -1.91 | 0.0128 | 0.0899 | no |
| Reconstructed hybrid (P1, prob) | 5 | -2.27 | -1.47 | 0.0304 | 0.1824 | no |
| Reconstructed hybrid (P1, hidden) | 5 | -2.14 | -1.28 | 0.0461 | 0.2307 | no |
| LightGBM (meta) | 5 | -1.01 | -0.73 | 0.1780 | 0.3559 | no |
| RGF w/o gate (mean of experts) | 5 | -0.87 | -1.19 | 0.0568 | 0.2307 | no |
| RGF w/o modality dropout | 5 | -0.09 | -0.14 | 0.7639 | 0.7639 | no |
| RGF w/o metadata expert | 5 | 0.49 | 0.92 | 0.1081 | 0.3242 | no |
| RGF w/o graph expert | 5 | 3.09 | 5.34 | 0.0003 | 0.0042 | yes |
| RGF w/o text expert | 5 | 5.96 | 8.86 | 0.0000 | 0.0007 | yes |
| GAT | 5 | 6.01 | 3.58 | 0.0013 | 0.0145 | yes |
| GCN | 5 | 6.98 | 7.67 | 0.0001 | 0.0011 | yes |
| GCN (text) | 5 | 7.34 | 9.89 | 0.0000 | 0.0005 | yes |
| Reconstructed hybrid (P3, prob) | 5 | 7.43 | 5.25 | 0.0003 | 0.0042 | yes |
| Reconstructed hybrid (P3, hidden) | 5 | 7.95 | 9.53 | 0.0000 | 0.0005 | yes |
| GCN (meta) | 5 | 11.21 | 12.55 | 0.0000 | 0.0002 | yes |
| LightGBM (text) | 5 | 24.91 | 26.73 | 0.0000 | 0.0000 | yes |


### T8a Mean gate weights of RGF by account group (Cresci-2017)

| group | n | w_meta | w_text | w_graph |
| --- | --- | --- | --- | --- |
| all | 2874.0 | 0.165 | 0.277 | 0.559 |
| bots | 2179.0 | 0.187 | 0.231 | 0.581 |
| humans | 695.0 | 0.093 | 0.419 | 0.488 |
| text missing | 1161.333 | 0.263 | 0.0 | 0.737 |


### T8b Each RGF expert alone (accounts where its modality exists) (Cresci-2017)

| group | n | f1_bot | mcc |
| --- | --- | --- | --- |
| graph | 2874.0 | 0.9547 | 0.8208 |
| metadata | 2874.0 | 0.9328 | 0.721 |
| text | 1712.6667 | 0.865 | 0.6433 |


### T8c Permutation importance for RGF (drop in F1 / MCC, percentage points) (Cresci-2017)

| group | delta_f1 | delta_mcc |
| --- | --- | --- |
| remove text (all test accounts) | 2.24 | 8.43 |
| remove graph (all test edges) | 0.8 | 3.75 |
| metadata: tweet_freq | 0.5 | 1.96 |
| metadata: follower_ratio | 0.24 | 1.01 |
| metadata: age_days | 0.1 | 0.44 |
| metadata: username_entropy | 0.0 | 0.03 |


### T8d LightGBM (meta+text) split-importance share (Cresci-2017)

| group | share |
| --- | --- |
| text (all dimensions) | 0.6567 |
| metadata: age_days | 0.1254 |
| metadata: tweet_freq | 0.1063 |
| metadata: follower_ratio | 0.0897 |
| metadata: username_entropy | 0.0219 |


## Cresci-2015


### T1 Data (Cresci-2015)

| Train | Validation | Test | Edge homophily (train) | Test accounts without graph links | Test accounts without text |
| --- | --- | --- | --- | --- | --- |
| 3710 | 530 | 1061 | 0.973 | 64.3% | 21.0% |


### T2 Main comparison (Cresci-2015)

| Method | Provenance | Seeds | Acc | Prec | Rec | F1 | MCC | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM (meta+text) | THIS-WORK | 5 | 97.57 ± 0.37 | 97.98 ± 0.66 | 98.18 ± 0.45 | 98.08 ± 0.28 | 94.77 ± 0.80 | 99.36 ± 0.35 | 99.53 ± 0.25 |
| XGBoost (meta+text) | THIS-WORK | 5 | 97.59 ± 0.31 | 98.04 ± 0.51 | 98.15 ± 0.29 | 98.09 ± 0.24 | 94.81 ± 0.67 | 99.53 ± 0.25 | 99.71 ± 0.17 |
| Random Forest (meta+text) | THIS-WORK | 5 | 97.55 ± 0.19 | 98.09 ± 0.45 | 98.03 ± 0.22 | 98.06 ± 0.14 | 94.73 ± 0.41 | 99.17 ± 0.34 | 99.29 ± 0.38 |
| GCN | THIS-WORK | 5 | 96.57 ± 0.56 | 96.52 ± 0.90 | 98.12 ± 0.34 | 97.31 ± 0.43 | 92.61 ± 1.21 | 99.11 ± 0.46 | 99.44 ± 0.33 |
| GraphSAGE | THIS-WORK | 5 | 97.30 ± 0.33 | 96.96 ± 0.46 | 98.84 ± 0.63 | 97.89 ± 0.26 | 94.20 ± 0.72 | 99.54 ± 0.11 | 99.72 ± 0.07 |
| GAT | THIS-WORK | 5 | 96.51 ± 0.44 | 95.79 ± 0.81 | 98.84 ± 0.34 | 97.29 ± 0.33 | 92.51 ± 0.92 | 99.39 ± 0.16 | 99.63 ± 0.08 |
| BotRGCN-style R-GCN | THIS-WORK | 5 | 97.42 ± 0.57 | 97.58 ± 0.68 | 98.36 ± 0.35 | 97.97 ± 0.45 | 94.44 ± 1.24 | 99.47 ± 0.21 | 99.67 ± 0.13 |
| Reconstructed hybrid (P1, hidden) [best of 4 by validation] | REPRODUCED-CORRECTED | 5 | 97.96 ± 0.21 | 98.54 ± 0.45 | 98.24 ± 0.32 | 98.39 ± 0.16 | 95.63 ± 0.45 | 99.46 ± 0.35 | 99.61 ± 0.29 |
| RGF (proposed) | THIS-WORK | 5 | 97.44 ± 0.18 | 97.75 ± 0.46 | 98.21 ± 0.57 | 97.98 ± 0.15 | 94.49 ± 0.39 | 99.54 ± 0.24 | 99.71 ± 0.14 |


### T3 Ablation of the proposed model (Cresci-2015)

| Variant | Seeds | Val F1 | F1 | MCC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| RGF (proposed) | 5 | 97.67 ± 0.71 | 97.98 ± 0.15 | 94.49 ± 0.39 | 99.54 ± 0.24 |
| RGF concat fusion | 5 | 98.42 ± 0.31 | 98.41 ± 0.11 | 95.66 ± 0.31 | 99.61 ± 0.17 |
| RGF w/o auxiliary losses | 5 | 97.98 ± 0.37 | 98.06 ± 0.19 | 94.68 ± 0.53 | 99.60 ± 0.19 |
| RGF w/o gate (mean of experts) | 5 | 98.27 ± 0.75 | 98.24 ± 0.16 | 95.22 ± 0.44 | 99.59 ± 0.18 |
| RGF w/o graph expert | 5 | 97.59 ± 0.59 | 97.41 ± 0.23 | 92.90 ± 0.64 | 99.30 ± 0.29 |
| RGF w/o metadata expert | 5 | 94.06 ± 1.02 | 93.98 ± 0.34 | 83.32 ± 0.90 | 95.50 ± 0.44 |
| RGF w/o modality dropout | 5 | 97.70 ± 0.63 | 97.89 ± 0.23 | 94.24 ± 0.65 | 99.53 ± 0.22 |
| RGF w/o text expert | 5 | 98.36 ± 0.30 | 98.07 ± 0.40 | 94.72 ± 1.12 | 99.36 ± 0.16 |


### T3b Modality ablation (same split and protocol for every row) (Cresci-2015)

| Method | Metadata | Text | Graph | Architecture | Seeds | F1 | MCC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM (meta) | yes |  |  | tree model | 5 | 97.93 ± 0.44 | 94.37 ± 1.19 | 99.38 ± 0.23 |
| LightGBM (text) |  | yes |  | tree model | 5 | 89.73 ± 1.03 | 70.05 ± 3.22 | 90.90 ± 1.24 |
| LightGBM (meta+text) | yes | yes |  | tree model | 5 | 98.08 ± 0.28 | 94.77 ± 0.80 | 99.36 ± 0.35 |
| GCN (meta) | yes |  | yes | GCN end-to-end | 5 | 97.95 ± 0.43 | 94.40 ± 1.20 | 99.16 ± 0.21 |
| GCN (text) |  | yes | yes | GCN end-to-end | 5 | 93.39 ± 0.63 | 81.53 ± 1.77 | 95.13 ± 0.92 |
| GCN | yes | yes | yes | GCN end-to-end | 5 | 97.31 ± 0.43 | 92.61 ± 1.21 | 99.11 ± 0.46 |
| Reconstructed hybrid (P3, hidden) | via GCN | via GCN | yes | P3: GCN 64-d embedding -> LightGBM | 5 | 97.21 ± 0.46 | 92.36 ± 1.28 | 99.10 ± 0.34 |
| Reconstructed hybrid (P3, prob) | via GCN | via GCN | yes | P3: GCN 2-d probabilities -> LightGBM | 5 | 97.14 ± 0.52 | 92.16 ± 1.45 | 98.78 ± 0.28 |
| Reconstructed hybrid (P1, hidden) | yes | yes | yes | P1: meta + text + GCN 64-d -> LightGBM | 5 | 98.39 ± 0.16 | 95.63 ± 0.45 | 99.46 ± 0.35 |
| Reconstructed hybrid (P1, prob) | yes | yes | yes | P1: meta + text + GCN 2-d -> LightGBM | 5 | 98.44 ± 0.19 | 95.74 ± 0.52 | 99.51 ± 0.26 |
| RGF (proposed) | yes | yes | yes | reliability-gated experts | 5 | 97.98 ± 0.15 | 94.49 ± 0.39 | 99.54 ± 0.24 |


### T4 Robustness, bot-class F1 (%) (Cresci-2015)

| perturbation | level | BotRGCN-style R-GCN | GAT | GCN | GraphSAGE | LightGBM (meta+text) | RGF (proposed) | RGF concat fusion | RGF w/o gate (mean of experts) | RGF w/o modality dropout | Reconstructed hybrid (P1, hidden) | Reconstructed hybrid (P1, prob) | Reconstructed hybrid (P3, hidden) | Reconstructed hybrid (P3, prob) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_edges | 0.0 | 97.85 | 97.43 | 97.46 | 97.96 | 98.02 | 98.07 | 98.42 | 98.34 | 97.95 | 98.33 | 98.36 | 97.45 | 97.5 |
| drop_edges | 0.25 | 97.78 | 97.27 | 97.42 | 97.84 | 98.02 | 98.0 | 98.37 | 98.29 | 97.88 | 98.33 | 98.34 | 97.22 | 97.4 |
| drop_edges | 0.5 | 97.63 | 97.06 | 97.25 | 97.67 | 98.02 | 97.83 | 98.3 | 98.1 | 97.74 | 98.29 | 98.34 | 97.18 | 97.22 |
| drop_edges | 0.75 | 97.35 | 96.86 | 97.05 | 97.39 | 98.02 | 97.62 | 98.1 | 97.93 | 97.43 | 98.31 | 98.34 | 97.12 | 97.06 |
| drop_edges | 1.0 | 96.99 | 96.21 | 96.62 | 96.96 | 98.02 | 97.31 | 97.86 | 97.67 | 97.12 | 98.19 | 98.14 | 96.72 | 96.44 |
| metadata_mimicry | 0.0 | 97.85 | 97.43 | 97.46 | 97.96 | 98.02 | 98.07 | 98.42 | 98.34 | 97.95 | 98.33 | 98.36 | 97.45 | 97.5 |
| metadata_mimicry | 0.25 | 87.92 | 90.89 | 88.77 | 89.29 | 83.66 | 90.52 | 87.17 | 89.73 | 90.27 | 83.95 | 84.92 | 89.16 | 89.27 |
| metadata_mimicry | 0.5 | 74.17 | 81.83 | 77.59 | 78.07 | 64.91 | 80.16 | 71.39 | 78.54 | 80.11 | 65.19 | 67.06 | 78.6 | 78.41 |
| metadata_mimicry | 1.0 | 35.69 | 60.71 | 46.57 | 46.38 | 0.0 | 55.25 | 26.68 | 48.83 | 54.7 | 0.49 | 6.75 | 50.51 | 49.67 |
| missing_text | 0.0 | 97.85 | 97.43 | 97.46 | 97.96 | 98.02 | 98.07 | 98.42 | 98.34 | 97.95 | 98.33 | 98.36 | 97.45 | 97.5 |
| missing_text | 0.25 | 98.02 | 96.96 | 97.54 | 97.92 | 97.82 | 98.15 | 98.39 | 98.47 | 98.08 | 98.26 | 98.32 | 97.48 | 97.6 |
| missing_text | 0.5 | 97.97 | 95.86 | 97.1 | 97.51 | 97.7 | 97.83 | 98.12 | 98.22 | 97.61 | 98.04 | 98.07 | 96.99 | 97.27 |
| missing_text | 0.75 | 97.74 | 95.02 | 96.9 | 97.21 | 97.73 | 97.61 | 98.05 | 98.07 | 97.28 | 97.9 | 97.95 | 96.95 | 96.88 |
| missing_text | 1.0 | 97.76 | 94.42 | 96.68 | 96.9 | 97.47 | 97.47 | 97.93 | 97.98 | 97.21 | 97.73 | 97.88 | 96.66 | 96.84 |
| noisy_edges | 0.0 | 97.85 | 97.43 | 97.46 | 97.96 | 98.02 | 98.07 | 98.42 | 98.34 | 97.95 | 98.33 | 98.36 | 97.45 | 97.5 |
| noisy_edges | 0.25 | 96.35 | 93.54 | 93.1 | 95.59 | 98.02 | 95.63 | 97.46 | 96.86 | 95.49 | 97.7 | 97.99 | 93.45 | 93.29 |
| noisy_edges | 0.5 | 94.96 | 90.56 | 90.88 | 93.53 | 98.02 | 94.49 | 97.0 | 96.53 | 94.4 | 96.87 | 97.73 | 89.72 | 90.38 |
| noisy_edges | 1.0 | 93.65 | 86.66 | 86.78 | 93.03 | 98.02 | 93.61 | 96.65 | 96.29 | 93.43 | 95.89 | 97.49 | 85.96 | 86.36 |


### T5 Unseen accounts: transductive vs inductive (Cresci-2015)

| Method | Transductive F1 | Inductive F1 | Transductive MCC | Inductive MCC |
| --- | --- | --- | --- | --- |
| GAT | 97.29 ± 0.33 | 97.13 ± 0.37 | 92.51 ± 0.92 | 92.07 ± 1.05 |
| GCN | 97.31 ± 0.43 | 97.31 ± 0.39 | 92.61 ± 1.21 | 92.61 ± 1.10 |
| GraphSAGE | 97.89 ± 0.26 | 97.77 ± 0.33 | 94.20 ± 0.72 | 93.88 ± 0.90 |
| LightGBM (meta+text) | 98.08 ± 0.28 | 98.08 ± 0.28 | 94.77 ± 0.80 | 94.77 ± 0.80 |
| RGF (proposed) | 97.98 ± 0.15 | 97.64 ± 0.46 | 94.49 ± 0.39 | 93.60 ± 1.27 |
| Reconstructed hybrid (P1, hidden) | 98.39 ± 0.16 | 98.29 ± 0.09 | 95.63 ± 0.45 | 95.34 ± 0.25 |
| Reconstructed hybrid (P1, prob) | 98.44 ± 0.19 | 98.38 ± 0.19 | 95.74 ± 0.52 | 95.58 ± 0.51 |
| Reconstructed hybrid (P3, hidden) | 97.21 ± 0.46 | 97.13 ± 0.41 | 92.36 ± 1.28 | 92.16 ± 1.11 |
| Reconstructed hybrid (P3, prob) | 97.14 ± 0.52 | 97.11 ± 0.50 | 92.16 ± 1.45 | 92.08 ± 1.42 |


### T6 Calibration and selective prediction (Cresci-2015)

| Method | ECE | Brier | Error @80% coverage | Error @90% coverage | ECE before temperature scaling |
| --- | --- | --- | --- | --- | --- |
| BotRGCN-style R-GCN | 1.66 ± 0.27 | 0.0216 ± 0.0041 | 0.57 ± 0.39 | 0.80 ± 0.35 | — |
| GAT | 1.74 ± 0.32 | 0.0271 ± 0.0029 | 0.49 ± 0.26 | 0.92 ± 0.20 | — |
| GCN | 1.71 ± 0.33 | 0.0268 ± 0.0054 | 0.78 ± 0.59 | 1.36 ± 0.58 | — |
| GCN (meta) | 1.40 ± 0.24 | 0.0212 ± 0.0038 | 0.66 ± 0.27 | 1.01 ± 0.25 | — |
| GCN (text) | 2.82 ± 0.77 | 0.0686 ± 0.0069 | 4.62 ± 0.82 | 5.76 ± 0.85 | — |
| GraphSAGE | 1.46 ± 0.23 | 0.0217 ± 0.0022 | 0.47 ± 0.26 | 0.65 ± 0.23 | — |
| LightGBM (meta) | 1.23 ± 0.27 | 0.0204 ± 0.0039 | 0.45 ± 0.23 | 0.61 ± 0.33 | — |
| LightGBM (meta+text) | 1.15 ± 0.21 | 0.0197 ± 0.0029 | 0.49 ± 0.35 | 0.77 ± 0.18 | — |
| LightGBM (text) | 4.09 ± 1.13 | 0.1046 ± 0.0099 | 8.29 ± 1.39 | 10.87 ± 1.53 | — |
| RGF (proposed) | 1.71 ± 0.38 | 0.0193 ± 0.0013 | 0.31 ± 0.27 | 0.42 ± 0.25 | 1.54 ± 0.37 |
| Random Forest (meta+text) | 10.42 ± 0.49 | 0.0360 ± 0.0019 | 0.61 ± 0.27 | 0.75 ± 0.29 | — |
| Reconstructed hybrid (P1, hidden) | 1.00 ± 0.20 | 0.0170 ± 0.0016 | 0.42 ± 0.37 | 0.71 ± 0.26 | — |
| Reconstructed hybrid (P1, prob) | 1.29 ± 0.24 | 0.0168 ± 0.0013 | 0.38 ± 0.35 | 0.65 ± 0.39 | — |
| Reconstructed hybrid (P3, hidden) | 1.98 ± 0.40 | 0.0282 ± 0.0035 | 0.64 ± 0.40 | 1.34 ± 0.41 | — |
| Reconstructed hybrid (P3, prob) | 1.69 ± 0.33 | 0.0291 ± 0.0045 | 1.06 ± 0.36 | 1.57 ± 0.46 | — |
| XGBoost (meta+text) | 1.26 ± 0.34 | 0.0192 ± 0.0030 | 0.45 ± 0.33 | 0.71 ± 0.39 | — |


### T9 Efficiency on CPU (Cresci-2015)

| Method | Fit time (s) | Inference (ms / 1k accounts) | Size (params or trees) |
| --- | --- | --- | --- |
| BotRGCN-style R-GCN | 3.6 | 2.8 | 53890 |
| GAT | 17.7 | 2.1 | 25158 |
| GCN | 2.5 | 2.1 | 25026 |
| GCN (meta) | 4.0 | 1.2 | 450 |
| GCN (text) | 1.9 | 1.8 | 24770 |
| GraphSAGE | 4.3 | 2.3 | 49986 |
| LightGBM (meta) | 0.1 | 3.1 | 93 |
| LightGBM (meta+text) | 4.0 | 4.5 | 89 |
| LightGBM (text) | 7.8 | 11.7 | 191 |
| RGF (proposed) | 9.4 | 4.9 | 50571 |
| Random Forest (meta+text) | 11.4 | 46.1 | 500 |
| Reconstructed hybrid (P1, hidden) | 6.7 | 7.4 | 25026 |
| Reconstructed hybrid (P1, prob) | 6.2 | 7.1 | 25026 |
| Reconstructed hybrid (P3, hidden) | 3.3 | 5.9 | 25026 |
| Reconstructed hybrid (P3, prob) | 2.9 | 4.3 | 25026 |
| XGBoost (meta+text) | 6.3 | 3.3 | 137 |


### T10 Paired t-tests, RGF vs each method (bot-class F1, Holm-corrected) (Cresci-2015)

| Compared with | Seeds | Mean diff (pp) | Cohen d_z | p | p (Holm) | Significant (0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| Reconstructed hybrid (P1, prob) | 5 | -0.46 | -1.7 | 0.0193 | 0.2311 | no |
| RGF concat fusion | 5 | -0.43 | -3.2 | 0.0020 | 0.0385 | yes |
| Reconstructed hybrid (P1, hidden) | 5 | -0.41 | -1.61 | 0.0228 | 0.2508 | no |
| RGF w/o gate (mean of experts) | 5 | -0.27 | -3.18 | 0.0021 | 0.0385 | yes |
| XGBoost (meta+text) | 5 | -0.12 | -0.4 | 0.4226 | 1.0000 | no |
| LightGBM (meta+text) | 5 | -0.1 | -0.29 | 0.5469 | 1.0000 | no |
| RGF w/o text expert | 5 | -0.09 | -0.19 | 0.6923 | 1.0000 | no |
| RGF w/o auxiliary losses | 5 | -0.08 | -0.75 | 0.1710 | 1.0000 | no |
| Random Forest (meta+text) | 5 | -0.08 | -0.54 | 0.2915 | 1.0000 | no |
| BotRGCN-style R-GCN | 5 | 0.01 | 0.02 | 0.9655 | 1.0000 | no |
| GCN (meta) | 5 | 0.03 | 0.06 | 0.9022 | 1.0000 | no |
| LightGBM (meta) | 5 | 0.05 | 0.1 | 0.8399 | 1.0000 | no |
| RGF w/o modality dropout | 5 | 0.08 | 0.44 | 0.3808 | 1.0000 | no |
| GraphSAGE | 5 | 0.09 | 0.36 | 0.4652 | 1.0000 | no |
| RGF w/o graph expert | 5 | 0.57 | 2.8 | 0.0033 | 0.0547 | no |
| GCN | 5 | 0.67 | 2.06 | 0.0100 | 0.1394 | no |
| GAT | 5 | 0.69 | 2.82 | 0.0032 | 0.0547 | no |
| Reconstructed hybrid (P3, hidden) | 5 | 0.77 | 2.19 | 0.0081 | 0.1214 | no |
| Reconstructed hybrid (P3, prob) | 5 | 0.84 | 2.02 | 0.0106 | 0.1394 | no |
| RGF w/o metadata expert | 5 | 3.99 | 10.72 | 0.0000 | 0.0004 | yes |
| GCN (text) | 5 | 4.59 | 7.28 | 0.0001 | 0.0017 | yes |
| LightGBM (text) | 5 | 8.25 | 8.66 | 0.0000 | 0.0009 | yes |


### T10b Paired t-tests, RGF vs each method (MCC, Holm-corrected) (Cresci-2015)

| Compared with | Seeds | Mean diff (pp) | Cohen d_z | p | p (Holm) | Significant (0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| Reconstructed hybrid (P1, prob) | 5 | -1.26 | -1.75 | 0.0175 | 0.2096 | no |
| RGF concat fusion | 5 | -1.17 | -3.24 | 0.0019 | 0.0367 | yes |
| Reconstructed hybrid (P1, hidden) | 5 | -1.14 | -1.68 | 0.0197 | 0.2167 | no |
| RGF w/o gate (mean of experts) | 5 | -0.73 | -3.14 | 0.0022 | 0.0388 | yes |
| XGBoost (meta+text) | 5 | -0.32 | -0.42 | 0.4035 | 1.0000 | no |
| LightGBM (meta+text) | 5 | -0.29 | -0.31 | 0.5315 | 1.0000 | no |
| Random Forest (meta+text) | 5 | -0.25 | -0.59 | 0.2578 | 1.0000 | no |
| RGF w/o text expert | 5 | -0.24 | -0.18 | 0.7109 | 1.0000 | no |
| RGF w/o auxiliary losses | 5 | -0.2 | -0.7 | 0.1949 | 1.0000 | no |
| BotRGCN-style R-GCN | 5 | 0.05 | 0.04 | 0.9412 | 1.0000 | no |
| GCN (meta) | 5 | 0.09 | 0.06 | 0.8993 | 1.0000 | no |
| LightGBM (meta) | 5 | 0.12 | 0.08 | 0.8622 | 1.0000 | no |
| RGF w/o modality dropout | 5 | 0.25 | 0.48 | 0.3472 | 1.0000 | no |
| GraphSAGE | 5 | 0.28 | 0.42 | 0.3970 | 1.0000 | no |
| RGF w/o graph expert | 5 | 1.59 | 2.86 | 0.0031 | 0.0509 | no |
| GCN | 5 | 1.88 | 2.07 | 0.0098 | 0.1402 | no |
| GAT | 5 | 1.98 | 2.88 | 0.0030 | 0.0509 | no |
| Reconstructed hybrid (P3, hidden) | 5 | 2.12 | 2.1 | 0.0093 | 0.1402 | no |
| Reconstructed hybrid (P3, prob) | 5 | 2.32 | 1.92 | 0.0128 | 0.1664 | no |
| RGF w/o metadata expert | 5 | 11.17 | 11.24 | 0.0000 | 0.0003 | yes |
| GCN (text) | 5 | 12.95 | 7.18 | 0.0001 | 0.0018 | yes |
| LightGBM (text) | 5 | 24.44 | 8.04 | 0.0001 | 0.0012 | yes |


### T8a Mean gate weights of RGF by account group (Cresci-2015)

| group | n | w_meta | w_text | w_graph |
| --- | --- | --- | --- | --- |
| all | 1061.0 | 0.453 | 0.29 | 0.257 |
| bots | 671.0 | 0.549 | 0.311 | 0.141 |
| humans | 390.0 | 0.289 | 0.253 | 0.458 |
| isolated (no graph) | 675.667 | 0.606 | 0.394 | 0.0 |
| text missing | 238.333 | 0.905 | 0.0 | 0.095 |


### T8b Each RGF expert alone (accounts where its modality exists) (Cresci-2015)

| group | n | f1_bot | mcc |
| --- | --- | --- | --- |
| graph | 385.3333 | 0.9878 | 0.9812 |
| metadata | 1061.0 | 0.9676 | 0.9102 |
| text | 822.6667 | 0.8755 | 0.7094 |


### T8c Permutation importance for RGF (drop in F1 / MCC, percentage points) (Cresci-2015)

| group | delta_f1 | delta_mcc |
| --- | --- | --- |
| metadata: follower_ratio | 10.58 | 25.21 |
| metadata: tweet_freq | 2.96 | 7.34 |
| remove graph (all test edges) | 0.76 | 2.16 |
| remove text (all test accounts) | 0.6 | 1.69 |
| metadata: age_days | 0.46 | 1.27 |
| metadata: username_entropy | 0.07 | 0.18 |


### T8d LightGBM (meta+text) split-importance share (Cresci-2015)

| group | share |
| --- | --- |
| text (all dimensions) | 0.507 |
| metadata: follower_ratio | 0.1953 |
| metadata: age_days | 0.1487 |
| metadata: tweet_freq | 0.0932 |
| metadata: username_entropy | 0.0558 |


## T7 Cross-dataset generalisation (train on one, test on the other)

| Train on | Test on | Method | Seeds | Shared accounts removed | Test bots | Test humans | Acc | Specificity | F1 | MCC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cresci-2015 | Cresci-2017 | GCN | 3 | 672 | 1507 | 695 | 67.65 ± 1.60 | 95.11 ± 0.63 | 69.78 ± 2.01 | 47.72 ± 1.83 | 81.38 ± 3.31 |
| Cresci-2015 | Cresci-2017 | GraphSAGE | 3 | 672 | 1507 | 695 | 60.73 ± 5.58 | 97.36 ± 1.09 | 59.86 ± 8.06 | 41.53 ± 5.44 | 79.08 ± 1.55 |
| Cresci-2015 | Cresci-2017 | LightGBM (meta+text) | 3 | 672 | 1507 | 695 | 57.18 ± 2.73 | 97.51 ± 0.51 | 54.91 ± 4.02 | 37.66 ± 2.92 | 68.91 ± 4.03 |
| Cresci-2015 | Cresci-2017 | RGF (proposed) | 3 | 672 | 1507 | 695 | 66.24 ± 1.26 | 96.74 ± 0.79 | 67.72 ± 1.62 | 47.17 ± 1.39 | 81.40 ± 2.83 |
| Cresci-2015 | Cresci-2017 | Reconstructed hybrid (P1, hidden) | 3 | 672 | 1507 | 695 | 55.84 ± 0.61 | 96.45 ± 1.58 | 53.24 ± 0.83 | 35.30 ± 1.52 | 67.56 ± 1.28 |
| Cresci-2015 | Cresci-2017 | Reconstructed hybrid (P1, prob) | 3 | 672 | 1507 | 695 | 55.71 ± 2.84 | 96.69 ± 0.88 | 52.89 ± 4.29 | 35.34 ± 3.16 | 72.25 ± 1.95 |
| Cresci-2015 | Cresci-2017 | Reconstructed hybrid (P3, hidden) | 3 | 672 | 1507 | 695 | 58.75 ± 2.09 | 95.06 ± 1.88 | 58.00 ± 2.35 | 37.43 ± 3.78 | 77.51 ± 6.12 |
| Cresci-2015 | Cresci-2017 | Reconstructed hybrid (P3, prob) | 3 | 672 | 1507 | 695 | 66.61 ± 1.93 | 95.40 ± 1.14 | 68.45 ± 2.18 | 46.69 ± 2.74 | 78.48 ± 2.14 |
| Cresci-2017 | Cresci-2015 | GCN | 3 | 671 | 0 | 390 | 14.79 ± 2.14 | 14.79 ± 2.14 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | GraphSAGE | 3 | 671 | 0 | 390 | 21.28 ± 3.85 | 21.28 ± 3.85 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | LightGBM (meta+text) | 3 | 671 | 0 | 390 | 46.50 ± 3.34 | 46.50 ± 3.34 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | RGF (proposed) | 3 | 671 | 0 | 390 | 17.52 ± 1.71 | 17.52 ± 1.71 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | Reconstructed hybrid (P1, hidden) | 3 | 671 | 0 | 390 | 29.66 ± 0.90 | 29.66 ± 0.90 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | Reconstructed hybrid (P1, prob) | 3 | 671 | 0 | 390 | 29.23 ± 3.45 | 29.23 ± 3.45 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | Reconstructed hybrid (P3, hidden) | 3 | 671 | 0 | 390 | 23.42 ± 0.65 | 23.42 ± 0.65 | — (no bots) | — | — |
| Cresci-2017 | Cresci-2015 | Reconstructed hybrid (P3, prob) | 3 | 671 | 0 | 390 | 16.92 ± 3.70 | 16.92 ± 3.70 | — (no bots) | — | — |


## Figures
- fig_robustness_cresci2015.png
- fig_robustness_cresci2017.png
- fig_curves_cresci2015.png
- fig_confusion_cresci2015.png
- fig_curves_cresci2017.png
- fig_confusion_cresci2017.png
