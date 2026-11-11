# Reproducibility report

New experiments for a paper drafted from scratch, on Cresci-2017 and Cresci-2015 (TwiBot-20 is not used).
**No experimental results exist yet**: no model has been trained on either dataset. This report records how the results will be produced. After each Kaggle
session, notebook step 19 writes `results/REPRODUCIBILITY_RUN.md` with the environment and completeness of that run.

## Environments

| | Where the code was written and tested | Target run environment |
| --- | --- | --- |
| Platform | Linux, Python 3.13.18, 2 CPU cores, 7.8 GB RAM, no GPU, no network | Kaggle notebook, CPU (Accelerator: None), Internet on |
| Packages | numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1 | Kaggle image + `sentence-transformers` (installed by step 2); versions recorded per run |
| Not available | torch, lightgbm, xgboost, transformers, sentence-transformers | All required (`requirements.txt`) |

## Randomness

- Seeds: 0, 1, 2, 3, 4 (`config/kaggle_cpu.json`). Robustness, cross-dataset and explainability runs use seeds 0–2.
  Checkpoints are saved for seed 0.
- Each run seeds `numpy` and `torch`. Cresci-2017 and Cresci-2015 splits (stratified 70/10/20) are drawn with the run seed.
- Blending halves, k-fold folds and perturbations are seeded.
- GCN results are reproducible for a given seed on CPU (`test_gcn_seed_reproducibility`).

## Commands

The notebook runs these; they can also be run from a terminal inside `botrecon_repo/`:

```bash
python -m pytest -q -rs                                         # all tests must pass (none skipped on Kaggle)
python -m botrecon.plan --part main     --cresci <cresci2017_root> [--cresci2015 <cresci2015_root>] --minutes 40
python -m botrecon.plan --part extended --cresci <cresci2017_root> [--cresci2015 <cresci2015_root>] --minutes 40
python -m botrecon.suite_report --results results --out results/paper
```

`--minutes` is a per-session limit. Rerunning the same command resumes, skipping finished runs (keyed by dataset,
setting, method, seed).

## Outputs (machine-readable)

| File | Content |
| --- | --- |
| `results/suite_main.jsonl` | One row per (dataset, setting, method, seed): validation and test metrics, calibration, timing, model size, provenance |
| `results/suite_robust.jsonl`, `suite_cross.jsonl`, `suite_explain.jsonl` | Robustness curves, cross-dataset transfer, explainability |
| `results/runs.csv` | Reconstruction study (manuscript-faithful, corrected, diagnostic) |
| `results/preds/*.npz` | Test predictions per run (curves, paired tests) |
| `results/checkpoints/` | Saved models (seed 0) |
| `results/paper/` | Tables T1–T10 (Markdown + CSV), figures, `paper_facts.json`, flat CSV copies of every results file |
| `results/data_audit.json`, `label_audit.json`, `protocol_audit.json` | Dataset counts and protocol checks on the real data |
| `results/walkthrough/` | Step-by-step outputs of notebook steps 8–14 (single seed, not a result) |
| `results/logs/` | Full console log of every session |
| `results/pytest.txt` | Test output from step 2 |

## Estimated cost (not measured on real data)

Measured on 2 CPU cores at full dataset size: kNN graph about 3 s, metadata features under 0.1 s, a 100-epoch GCN
about 5–10 s. The full plan is estimated at 1.5–2 h of Kaggle CPU time: about three 40-minute sessions, or one
unattended *Save & Run All*. BERT on tweets (variant A) is the slowest step and is off by default.

## Known limitations

- PyTorch, LightGBM, XGBoost and transformer code paths are covered by tests that have only run on synthetic data, and
  only on Kaggle.
- The Kaggle Cresci-2017 copy is third-party; group sizes are checked against the published release in step 6.
- Cresci-2015 must be downloaded from the Botometer bot repository and uploaded as a private Kaggle dataset; until
  then only Cresci-2017 runs. The relation-file direction convention of Cresci-2015 is not relied on (graphs are
  symmetrised); only R-GCN uses direction.
- The Cresci-2017 fake followers come from the 2015 study; cross-dataset tests remove shared accounts by user ID.
- Design choices (feature formulas, SBERT checkpoint, graph symmetrisation) are documented in `EXPERIMENT_PROTOCOL.md`.
- No number from the earlier manuscript is used or reproduced; the paper reports only numbers produced here.
