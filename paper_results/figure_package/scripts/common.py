"""Shared loaders, naming and IEEE styling for every figure and table script.

All numbers come from the result files in ../data/raw (copied unchanged from the Kaggle run
results/paper/, checksums in data/raw/SHA256SUMS). Nothing here creates or edits values.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, DER, FIG, TAB = ROOT / "data/raw", ROOT / "data/derived", ROOT / "figures", ROOT / "tables"
for p in (DER, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {"cresci2017": "Cresci-2017", "cresci2015": "Cresci-2015"}

# Short, consistent method names used in every figure and table.
SHORT = {
    "LightGBM (meta+text)": "LightGBM",
    "XGBoost (meta+text)": "XGBoost",
    "Random Forest (meta+text)": "Random forest",
    "GCN": "GCN", "GraphSAGE": "GraphSAGE", "GAT": "GAT",
    "BotRGCN-style R-GCN": "R-GCN (BotRGCN-style)",
    "Reconstructed hybrid (P1, hidden)": "Hybrid P1-h",
    "Reconstructed hybrid (P1, prob)": "Hybrid P1-p",
    "Reconstructed hybrid (P3, hidden)": "Hybrid P3-h",
    "Reconstructed hybrid (P3, prob)": "Hybrid P3-p",
    "RGF (proposed)": "RGF (proposed)",
    "RGF concat fusion": "Concat fusion",
    "RGF w/o auxiliary losses": "w/o auxiliary losses",
    "RGF w/o gate (mean of experts)": "w/o gate (mean)",
    "RGF w/o graph expert": "w/o graph expert",
    "RGF w/o metadata expert": "w/o metadata expert",
    "RGF w/o modality dropout": "w/o modality dropout",
    "RGF w/o text expert": "w/o text expert",
    "LightGBM (meta)": "LightGBM (meta)", "LightGBM (text)": "LightGBM (text)",
    "GCN (meta)": "GCN (meta)", "GCN (text)": "GCN (text)",
}

# Main comparison set (Fig. 5-9). Hybrid = best of 4 by validation F1 on both datasets (paper_facts.json).
MAIN = ["LightGBM (meta+text)", "XGBoost (meta+text)", "Random Forest (meta+text)", "GCN", "GraphSAGE",
        "GAT", "BotRGCN-style R-GCN", "Reconstructed hybrid (P1, hidden)", "RGF (proposed)"]
ROBUST = ["LightGBM (meta+text)", "GCN", "GraphSAGE", "GAT", "BotRGCN-style R-GCN",
          "Reconstructed hybrid (P1, hidden)", "RGF (proposed)"]
ABLATIONS = ["RGF (proposed)", "RGF concat fusion", "RGF w/o gate (mean of experts)", "RGF w/o auxiliary losses",
             "RGF w/o modality dropout", "RGF w/o metadata expert", "RGF w/o text expert", "RGF w/o graph expert"]

# Okabe-Ito colour-blind-safe palette; RGF always black and drawn last. Markers differ too (grey-scale safe).
STYLE = {
    "LightGBM (meta+text)": ("#E69F00", "s"), "XGBoost (meta+text)": ("#F0C05A", "D"),
    "Random Forest (meta+text)": ("#999999", "v"), "GCN": ("#56B4E9", "o"), "GraphSAGE": ("#0072B2", "^"),
    "GAT": ("#009E73", "P"), "BotRGCN-style R-GCN": ("#CC79A7", "X"),
    "Reconstructed hybrid (P1, hidden)": ("#D55E00", "h"), "RGF (proposed)": ("#000000", "*"),
}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False, "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
    })


ONE_COL, TWO_COL = 3.5, 7.16   # IEEE column widths in inches


def save(fig, name):
    """Vector PDF + SVG for the journal, 600-dpi PNG for review and Word."""
    for ext in ("pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)


def load(name):
    return pd.read_csv(RAW / f"suite_{name}.csv")


def mean_sd(df, keys, col):
    g = df.groupby(keys)[col].agg(["mean", "std", "count"]).reset_index()
    return g.rename(columns={"mean": f"{col}_mean", "std": f"{col}_sd", "count": "seeds"})


def pct(x):
    """Result files store fractions in [0, 1]; papers report percent."""
    return 100 * x


def fmt(m, s, d=2):
    return f"{m:.{d}f} ± {s:.{d}f}"
