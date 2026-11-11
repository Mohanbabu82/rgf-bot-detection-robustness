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
