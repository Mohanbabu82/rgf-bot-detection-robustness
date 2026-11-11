"""Figures 1-4: technical diagrams. Every element is taken from the implementation
(botrecon/nn.py, methods.py, suite.py) and the run's data audit; nothing is illustrative only.
Run:  python diagrams.py   ->  figures/fig01..fig04 .pdf/.svg/.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import style, save, TWO_COL

INK, GREY = "#222222", "#6b6b6b"
FILL = {"data": "#eef3f8", "prep": "#f4f4f4", "model": "#fdf1e3", "rgf": "#e8f4ec", "eval": "#f3eef7", "warn": "#fbeaea"}


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title=None, lines=(), fc="#ffffff", ec=INK, lw=0.7, fs=7, tfs=7.5, ls="-", align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    n = len(lines) + (1 if title else 0)
    step = 0.13 if fs <= 6.5 else 0.145
    top = y + h / 2 + (n - 1) * step / 2
    tx = x + w / 2 if align == "center" else x + 0.06
    if title:
        ax.text(tx, top, title, ha=align, va="center", fontsize=tfs, weight="bold", color=INK)
        top -= step
    for i, s in enumerate(lines):
        ax.text(tx, top - i * step, s, ha=align, va="center", fontsize=fs, color=INK)
    return (x, y, w, h)


def arrow(ax, p, q, color=INK, lw=0.8, ls="-", rad=0.0, head=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={head/20},head_width={head/40}",
                                 mutation_scale=10, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def R(b): return (b[0] + b[2], b[1] + b[3] / 2)
def L(b): return (b[0], b[1] + b[3] / 2)
def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def B(b): return (b[0] + b[2] / 2, b[1])


# ---------------------------------------------------------------- Figure 1
def fig1():
    W, H = TWO_COL, 3.05
    fig, ax = canvas(W, H)
    cw, gap, y0, h = 1.06, 0.14, 0.55, 2.2
    xs = [0.05 + i * (cw + gap) for i in range(6)]
    heads = ["1  Datasets", "2  Preprocessing", "3  Features", "4  Models", "5  Robustness tests", "6  Evaluation"]
    fills = [FILL["data"], FILL["prep"], FILL["prep"], FILL["model"], FILL["warn"], FILL["eval"]]
    for x, t, f in zip(xs, heads, fills):
        box(ax, x, y0, cw, h, fc=f, lw=0.6)
        ax.text(x + cw / 2, y0 + h - 0.13, t, ha="center", va="center", fontsize=7.5, weight="bold")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + cw + 0.01, y0 + h / 2), (b - 0.01, y0 + h / 2))
    def inner(i, yy, hh, title, lines, fc="#ffffff", ls="-"):
        return box(ax, xs[i] + 0.05, yy, cw - 0.1, hh, title, lines, fc=fc, fs=6.2, tfs=6.6, ls=ls)
    inner(0, 1.62, 0.88, "Cresci-2017", ["14,368 accounts", "75.8% bots", "no follow graph"])
    inner(0, 0.62, 0.88, "Cresci-2015", ["5,301 accounts", "63.2% bots", "follower/friend graph", "(64.3% test isolated)"])
    inner(1, 1.83, 0.67, "Deduplicate by ID", ["0 conflicts dropped"])
    inner(1, 1.17, 0.6, "Stratified split", ["70 / 10 / 20", "seeds 0-4"])
    inner(1, 0.62, 0.49, "Test labels masked", ["(stored as -1)"])
    inner(2, 1.83, 0.67, "Metadata  d=4", ["median impute,", "log1p, z-score", "(train-fitted)"])
    inner(2, 1.17, 0.6, "Text  d=384", ["MiniLM-L6-v2,", "frozen; empty -> 0"])
    inner(2, 0.62, 0.49, "Graph", ["kNN k=10 / follow"])
    inner(3, 1.86, 0.64, "Tree baselines", ["LightGBM, XGBoost,", "random forest"])
    inner(3, 1.28, 0.53, "GNN baselines", ["GCN, SAGE, GAT,", "R-GCN (C-15)"])
    inner(3, 0.88, 0.36, None, ["Hybrid P1/P3"])
    inner(3, 0.62, 0.22, None, ["RGF (proposed)"], fc=FILL["rgf"])
    inner(4, 1.83, 0.67, "Missing modality", ["text removal ρ", "edge deletion ρ"])
    inner(4, 1.17, 0.6, "Adversarial", ["metadata mimicry ρ", "edge injection ρ"])
    inner(4, 0.62, 0.49, "Test nodes only;", ["clean-trained model"])
    inner(5, 1.83, 0.67, "Discrimination", ["F1 (bot), MCC,", "ROC-AUC, PR-AUC"])
    inner(5, 1.17, 0.6, "Probability", ["ECE, Brier,", "selective error"])
    inner(5, 0.62, 0.49, "Transfer & tests", ["inductive, cross-", "dataset; Holm t"])
    ax.text(W / 2, 0.3, "Every model is fitted on the training split only; the validation split drives early stopping "
            "and RGF's temperature; the test split is used once per seed.", ha="center", fontsize=6.5, color=GREY)
    save(fig, "fig01_research_framework")


# ---------------------------------------------------------------- Figure 2
def fig2():
    W, H = TWO_COL, 3.75
    fig, ax = canvas(W, H)
    rows = {"m": 2.95, "t": 2.1, "g": 1.25}
    ins = {"m": ("Metadata", [r"$\mathbf{m}_v\in\mathbb{R}^{4}$"]),
           "t": ("Profile text", [r"$\mathbf{t}_v\in\mathbb{R}^{384}$ (frozen)"]),
           "g": ("Graph", [r"row-mean $\bar{A}$, no self-loops"])}
    exps = {"m": ("Metadata expert", ["Linear 4→64, LeakyReLU, Dropout"]),
            "t": ("Text expert", ["Linear 384→64, LeakyReLU,", r"Dropout, $\times\,a^t_v$"]),
            "g": ("Graph expert", [r"2 mean-aggregation layers on", r"$[\mathbf{h}^m\Vert\mathbf{h}^t]$: 128→64→64, $\times\,a^g_v$"])}
    heads = {}
    for k, y in rows.items():
        bi = box(ax, 0.05, y, 1.0, 0.6, ins[k][0], ins[k][1], fc=FILL["data"], fs=6.0, tfs=6.6)
        be = box(ax, 1.25, y, 1.55, 0.6, exps[k][0], exps[k][1], fc=FILL["rgf"], fs=5.8, tfs=6.6)
        bh = box(ax, 3.0, y + 0.1, 0.8, 0.4, None, [rf"head → $\mathbf{{z}}^{k}_v\in\mathbb{{R}}^2$"], fs=5.8)
        arrow(ax, R(bi), L(be)); arrow(ax, R(be), L(bh)); heads[k] = bh
    arrow(ax, (2.02, rows["t"]), (2.02, rows["g"] + 0.6), color=GREY, lw=0.6, ls="--")
    ax.text(2.07, rows["g"] + 0.72, r"$\mathbf{h}^m,\mathbf{h}^t$ of neighbours", fontsize=5.6, color=GREY)
    av = box(ax, 0.05, 0.2, 1.0, 0.75, "Availability", [r"$a^m_v=1$", r"$a^t_v$: text non-empty", r"$a^g_v=\mathbb{1}[\deg(v)>0]$"],
             fc=FILL["prep"], fs=5.8, tfs=6.6)
    rs = box(ax, 1.25, 0.2, 1.55, 0.75, r"Reliability signals $\mathbf{r}_v\in\mathbb{R}^6$",
             [r"$a^t_v,\ a^g_v,\ \log(1+\deg v)/5,$", r"$H(\mathbf{z}^m_v), H(\mathbf{z}^t_v), H(\mathbf{z}^g_v)$"],
             fc=FILL["prep"], fs=5.8, tfs=6.4)
    gt = box(ax, 3.0, 0.2, 0.95, 0.75, r"Gate $g_\phi$", ["MLP 6→32→3", r"scores $\mathbf{s}_v\in\mathbb{R}^3$"],
             fc=FILL["model"], fs=5.8, tfs=6.6)
    ms = box(ax, 4.2, 0.2, 1.45, 0.75, "Masked softmax",
             [r"$\mathbf{o}_v=[1,a^t_v,a^g_v]$", r"$w_{vk}\propto o_{vk}\,e^{s_{vk}},\ \Sigma_k w_{vk}=1$"],
             fc=FILL["model"], fs=5.8, tfs=6.6)
    arrow(ax, R(av), L(rs)); arrow(ax, R(rs), L(gt)); arrow(ax, R(gt), L(ms))
    ax.plot([0.55, 0.55, 4.92], [0.2, 0.08, 0.08], color=GREY, lw=0.6, ls="--")
    arrow(ax, (4.92, 0.08), (4.92, 0.2), color=GREY, lw=0.6, ls="--")
    ax.text(2.0, 0.105, r"availability mask $\mathbf{o}_v$", fontsize=5.6, color=GREY, ha="center", va="bottom")
    arrow(ax, B(heads["g"]), (2.6, 0.95), color=GREY, lw=0.6, ls="--")
    ax.text(3.45, 1.08, "entropies of all three heads\n(gradient stopped)", fontsize=5.6, color=GREY, va="center")
    arrow(ax, (0.55, rows["g"]), (1.4, 0.95), color=GREY, lw=0.6, ls="--")
    ax.text(0.62, 1.07, "deg(v)", fontsize=5.6, color=GREY)
    fu = box(ax, 4.2, 1.6, 1.45, 1.7, "Weighted-logit fusion",
             [r"$\mathbf{z}_v=\Sigma_{k}\, w_{vk}\,\mathbf{z}^k_v$", "",
              "unavailable experts", "receive weight 0"], fc=FILL["model"], fs=5.8, tfs=6.6)
    for k, b in heads.items():
        arrow(ax, R(b), (4.2, b[1] + b[3] / 2 if k != "g" else 1.75))
    arrow(ax, T(ms), B(fu)); ax.text(4.97, 1.25, r"$\mathbf{w}_v$", fontsize=6.2)
    pr = box(ax, 5.85, 1.6, 1.25, 1.7, "Prediction",
             [r"$p_v=\mathrm{softmax}(\mathbf{z}_v/T)_1$", "", "T fitted on", "validation NLL", "(class unchanged)"],
             fc=FILL["eval"], fs=5.8, tfs=6.6)
    arrow(ax, R(fu), L(pr))
    save(fig, "fig02_rgf_architecture")


# ---------------------------------------------------------------- Figure 3
def fig3():
    W, H = TWO_COL, 2.95
    fig, ax = canvas(W, H)
    y = 1.55
    b1 = box(ax, 0.05, y, 1.05, 0.95, "Modality dropout", [r"$\beta_t,\beta_g\sim\mathrm{Bernoulli}(0.7)$", "per account, per epoch", r"$\tilde a=a\,\beta$ (text, graph)", "metadata never dropped"], fc=FILL["warn"], fs=5.8, tfs=6.4)
    b2 = box(ax, 1.3, y, 1.05, 0.95, "Forward pass", ["full batch, whole graph", r"experts $z^m, z^t, z^g$", r"gate on $r(\tilde a)$; deg kept", "fused z (T = 1)"], fc=FILL["rgf"], fs=5.8, tfs=6.4)
    b3 = box(ax, 2.55, y, 1.35, 0.95, "Loss on training nodes", [r"$L_{fuse}=\mathrm{CE}(z,y)$", r"$L_k=\mathrm{CE}(z^k,y)$ where", "expert k available", r"$L=L_{fuse}+0.5\,\Sigma_k L_k$"], fc=FILL["model"], fs=5.8, tfs=6.4)
    b4 = box(ax, 4.1, y, 0.95, 0.95, "AdamW step", ["lr 0.005", r"wd $5\times10^{-4}$"], fc=FILL["prep"], fs=5.8, tfs=6.4)
    b5 = box(ax, 5.25, y, 1.85, 0.95, "Validation (eval mode)", ["no dropout, no modality", r"dropout, $T=1$: CE$(z,y_{val})$", "keep best state; stop after", "40 epochs w/o gain (≤ 300)"], fc=FILL["eval"], fs=5.8, tfs=6.4)
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        arrow(ax, R(a), L(b))
    ax.annotate("", xy=(0.55, y + 0.95), xytext=(6.15, y + 0.95), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK,
                connectionstyle="arc3,rad=0.1", mutation_scale=8))
    ax.text(3.35, 2.86, "next epoch", fontsize=6.2, ha="center", va="center", color=INK)
    b6 = box(ax, 3.0, 0.3, 1.75, 0.65, "Restore best parameters", ["(lowest validation CE)"], fc=FILL["prep"], fs=5.8, tfs=6.4)
    b7 = box(ax, 5.25, 0.15, 1.85, 0.95, "Temperature scaling", [r"$T^*=\arg\min_T\,\mathrm{NLL}(z/T,\,y_{val})$", r"60 log-spaced $T\in[0.25, 5]$", "same validation split"], fc=FILL["eval"], fs=5.8, tfs=6.4)
    arrow(ax, (5.5, y), (4.75, 0.8), rad=0.0); arrow(ax, R(b6), L(b7))
    ax.text(0.1, 0.62, "Test labels are masked (-1);\nassertions reject them in the\ntraining and validation sets.", fontsize=6.0, color=GREY, va="center")
    save(fig, "fig03_training_workflow")


# ---------------------------------------------------------------- Figure 4
def fig4():
    W, H = TWO_COL, 3.0
    fig, ax = canvas(W, H)
    c = box(ax, 2.68, 1.25, 1.8, 0.62, "Model trained once on clean data", ["per dataset and seed"], fc=FILL["rgf"], fs=6.2, tfs=6.6)
    items = [
        ("Clean", ["5 seeds, test split", "F1, MCC, ROC/PR-AUC"], FILL["eval"]),
        ("Missing modality", ["text removed, ρ = 0-1", "edges deleted, ρ = 0-1"], FILL["warn"]),
        ("Adversarial", ["metadata mimicry ρ = 0-1", "edges injected ρ = 0-1"], FILL["warn"]),
        ("Inductive", ["test nodes unseen in", "training graph (5 seeds)"], FILL["eval"]),
        ("Cross-dataset", ["C-15 → C-17 and C-17 → C-15", "shared IDs removed; seeds 0-2"], FILL["eval"]),
        ("Calibration", ["ECE (15 bins), Brier,", "before/after T (RGF)"], FILL["eval"]),
        ("Selective prediction", ["error on most confident", "80% / 90% of accounts"], FILL["eval"]),
    ]
    pos = [(0.05, 2.2), (2.4, 2.2), (4.75, 2.2), (0.05, 1.25), (4.75, 1.25), (1.25, 0.25), (3.6, 0.25)]
    for (t, ls, f), (x, y) in zip(items, pos):
        b = box(ax, x, y, 2.3 if y > 1 else 2.3, 0.62, t, ls, fc=f, fs=6.0, tfs=6.6)
        cx, cy = c[0] + c[2] / 2, c[1] + c[3] / 2
        tx, ty = b[0] + b[2] / 2, b[1] + b[3] / 2
        if abs(ty - cy) < 0.1:
            p, q = ((c[0], cy), (b[0] + b[2], ty)) if tx < cx else ((c[0] + c[2], cy), (b[0], ty))
        else:
            p = (cx + 0.5 * (tx - cx) / 2.35, c[1] + (c[3] if ty > cy else 0))
            q = (tx, b[1] if ty > cy else b[1] + b[3])
        arrow(ax, p, q, lw=0.7)
    ax.text(W / 2, 0.06, "Perturbations touch test accounts only; the graph is not rebuilt. Robustness and cross-dataset: seeds 0-2.",
            ha="center", fontsize=6.0, color=GREY)
    save(fig, "fig04_evaluation_framework")


if __name__ == "__main__":
    style()
    fig1(); fig2(); fig3(); fig4()
