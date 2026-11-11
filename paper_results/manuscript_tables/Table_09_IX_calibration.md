TABLE IX. Calibration on the test split (mean ± SD over 5 seeds)

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