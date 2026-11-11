# Accuracy Is Not Robustness: Reliability-Gated Multi-Modal Fusion for Social Bot Detection

Code, result files and figure/table scripts for the article:

> *Accuracy Is Not Robustness: Reliability-Gated Multi-Modal Fusion for Social Bot Detection under Missing and Adversarial Modalities* (manuscript submitted to IEEE Access).

The study compares tree-based, graph-based, hybrid and fusion bot detectors on **Cresci-2015** and **Cresci-2017** under one leakage-controlled protocol. Each model is trained on clean data and then stress-tested with:

- missing text;
- edge deletion and edge injection;
- metadata mimicry;
- inductive (unseen-account) evaluation;
- cross-dataset transfer.

The proposed model, **RGF**, combines metadata, text and graph experts through a gate that is restricted to the modalities available for each account.

## Repository layout

| Path | Contents |
| --- | --- |
| `botrecon/` | Python package: data loaders, features, graphs, models (LightGBM, XGBoost, random forest, GCN, GraphSAGE, GAT, R-GCN, hybrid, RGF), perturbations, metrics, statistics, reports |
| `tests/` | Unit tests (89 passed before the reported run) |
| `notebooks/reproduce_kaggle_cpu.ipynb` | End-to-end reproduction on a Kaggle CPU session |
| `config/kaggle_cpu.json` | Configuration used for the reported run |
| `requirements.txt` | Python dependencies |
| `EXPERIMENT_PROTOCOL.md`, `ARCHITECTURE.md`, `REPRODUCIBILITY_REPORT.md`, `TEST_REPORT.md`, `CODE_AUDIT_REPORT.md` | Protocol, model details, environment, tests and code audit |
| `CODE_README.md` | Original code documentation (written before the experiments were run) |
| `paper_results/figure_package/data/raw/` | **Result files of the reported run.** `suite_main.csv`, `suite_robust.csv`, `suite_cross.csv`, `suite_explain.csv`, data and protocol audits, the run's own `tables.md`, and SHA-256 checksums |
| `paper_results/figure_package/scripts/` | Scripts that regenerate every figure and table from those files: `diagrams.py`, `plots.py`, `tables.py`. `verify.py` (461 cross-checks against the run's report), `stat_audit.py` (paired t-tests, Holm correction, Nadeau–Bengio corrected tests, confidence intervals) |
| `paper_results/figure_package/audit_stats/` | Outputs of the statistical audit |
| `paper_results/manuscript_figures/` | Figures 1–12 as numbered in the article (PDF, SVG, 600-dpi PNG) |
| `paper_results/manuscript_tables/` | Tables I–XI as numbered in the article (CSV, Markdown, LaTeX) |
| `paper_results/supplementary_tables/` | Supplementary Tables S1–S4 |

The visual package in `paper_results/figure_package/` keeps its original working numbering. The article renumbers figures and tables in order of first citation; the files in `manuscript_figures/` and `manuscript_tables/` use the article's numbering.

## Data

The Cresci-2015 and Cresci-2017 datasets are **not** included in this repository. Download them from the Bot Repository of the Observatory on Social Media (Indiana University), https://botometer.osome.iu.edu/bot-repository/datasets.html, and cite the original papers:

- S. Cresci et al., "Fame for sale: Efficient detection of fake Twitter followers," *Decision Support Systems*, 2015.
- S. Cresci et al., "The paradigm-shift of social spambots," *WWW '17 Companion*, 2017.

## Reproducing the results

**Full experiment.** Open `notebooks/reproduce_kaggle_cpu.ipynb` on Kaggle (CPU, no GPU) and attach:

- this code;
- the two Cresci datasets;
- the `sentence-transformers/all-MiniLM-L6-v2` weights.

Then run all cells. The reported run took 7,002 s on 4 CPU threads with:

- Python 3.13;
- PyTorch 2.11;
- LightGBM 4.6;
- XGBoost 3.4;
- scikit-learn 1.6;
- sentence-transformers 5.7.

**Figures, tables and statistics from the saved results.** No training is needed:

```bash
pip install pandas numpy scipy matplotlib
cd paper_results/figure_package/scripts
python diagrams.py && python plots.py && python tables.py
python verify.py      # must print: 461 checks, 0 mismatches
python stat_audit.py  # statistical audit -> ../audit_stats/
```

## Protocol summary

- **Splits:** stratified 70/10/20 per seed. Every transformation is fitted on training rows only, and test labels are masked and asserted absent.
- **Seeds:** five (0–4) for clean results, three (0–2) for robustness and cross-dataset evaluation.
- **Graphs:**
  - Cresci-2017 has no released graph, so a kNN graph (k = 10) is built on standardised metadata. It is not rebuilt after perturbation; see the article's limitations.
  - Cresci-2015 uses its follower/friend graph.
- **Cross-dataset evaluation:** accounts shared by the two collections are removed by user ID.

## License

The code is released under the MIT License (see `LICENSE`). The datasets are subject to their original terms.

## Citation

If you use this code, please cite the article (full citation to be added on publication).
