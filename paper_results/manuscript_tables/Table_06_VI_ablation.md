TABLE VI. Ablation of RGF (%, mean ± SD over 5 seeds)

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

*ΔMCC = variant − RGF in percentage points; † / ‡ as in Table V (here † = variant significantly better than RGF). Concat fusion also removes the auxiliary losses, so it changes two components.*