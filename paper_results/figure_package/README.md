# Visual package: Accuracy Is Not Robustness (RGF, IEEE Access)

```
figure_package/
├── FIGURE_TABLE_GUIDE.md        captions, source, script, section and interpretation check for every item
├── VISUAL_PACKAGE_PREVIEW.pdf   all 12 figures with captions + all 15 tables, IEEE two-column layout
├── figures/                     fig01-fig12 as .pdf and .svg (vector) and .png (600 dpi), 7.16 in wide
├── tables/                      table01-table15 as .tex, .csv, .md; all_tables.tex; ALL_TABLES.md;
│                                table04_significance_detail.csv (all paired t-tests, Holm-corrected)
├── data/raw/                    result files from the Kaggle run, unchanged (+ SHA256SUMS)
├── data/derived/                exact numbers behind each plotted figure (fig05-fig12)
└── scripts/
    ├── common.py                loaders, method names, colour-blind-safe palette, IEEE styling
    ├── diagrams.py              Figs. 1-4
    ├── plots.py                 Figs. 5-12
    ├── tables.py                Tables I-XV
    ├── verify.py                461 cross-checks against the run's own report (must print 0 mismatches)
    └── build_preview.py         VISUAL_PACKAGE_PREVIEW.pdf
```

Reproduce: `cd scripts && python diagrams.py && python plots.py && python tables.py && python verify.py && python build_preview.py`

Requires Python 3 with pandas, numpy, scipy and matplotlib (≥ 3.8). `build_preview.py` also needs pdflatex. In LaTeX, use the `.tex` tables with `\usepackage{amsmath}`; all figures are sized for `figure*`.
