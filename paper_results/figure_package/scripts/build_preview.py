"""Builds VISUAL_PACKAGE_PREVIEW.pdf: all figures with IEEE-style captions, then all tables.
Captions here are the submission captions (same text as FIGURE_TABLE_GUIDE.md, condensed to one paragraph).
Run after diagrams.py, plots.py and tables.py:  python build_preview.py   (needs pdflatex)
"""
import subprocess, shutil, tempfile
from pathlib import Path
from common import ROOT, FIG, TAB

CAPTIONS = {
 "fig01_research_framework": "Research framework: two Cresci benchmarks; leakage-controlled preprocessing (stratified 70/10/20 splits over five seeds, training-fitted transforms, masked test labels); metadata, text and graph features; tree, GNN and hybrid baselines and the proposed RGF; test-time perturbations of models trained on clean data; and evaluation.",
 "fig02_rgf_architecture": "Reliability-gated fusion (RGF). Three experts produce class logits; the text and graph experts are multiplied by their availability flags. A gate MLP maps six reliability signals (availability flags, scaled log-degree, stop-gradient expert entropies) to scores, a softmax restricted to available experts gives the weights, and the fused logits are the weighted sum of expert logits divided by a validation-fitted temperature $T$.",
 "fig03_training_workflow": "Training of RGF: per-account modality dropout ($p=0.3$, metadata never dropped), fused plus auxiliary expert cross-entropy ($\\lambda=0.5$), full-batch AdamW, early stopping on validation cross-entropy (patience 40, at most 300 epochs), then temperature selection on the same validation split.",
 "fig04_evaluation_framework": "Evaluation protocol. Each model is trained once per dataset and seed on clean data and evaluated on seven axes. Perturbations affect test accounts only; the graph is not rebuilt.",
 "fig05_clean_performance": "Clean test performance on Cresci-2017 (circles) and Cresci-2015 (squares): (a) bot-class F1, (b) MCC, (c) ROC-AUC; mean $\\pm$ SD over five seeds. R-GCN is run on Cresci-2015 only.",
 "fig06_accuracy_vs_robustness": "Accuracy is not robustness. (a, b) Bot-class F1 when a fraction $\\rho$ of test bots receive the median metadata of training humans (mean $\\pm$ SD, three seeds). (c, d) Ranks on clean data and under full mimicry, with F1 (\\%).",
 "fig07_missing_text": "Bot-class F1 when the profile text of a fraction $\\rho$ of test accounts is removed (mean $\\pm$ SD, three seeds).",
 "fig08_graph_perturbation": "Bot-class F1 under (a, c) deletion of a fraction $\\rho$ of test-account edges and (b, d) injection of round($\\rho\\times$degree) random edges per test account (mean $\\pm$ SD, three seeds). LightGBM (dashed) uses no graph.",
 "fig09_cross_dataset": "Cross-dataset transfer (three seeds). (a) Train Cresci-2015, test Cresci-2017. (b) Share of humans flagged as bots in both directions; the Cresci-2017$\\rightarrow$Cresci-2015 test set has no bots after shared accounts are removed.",
 "fig10_ablation": "Ablation of RGF. (a, b) Clean F1 and MCC (five seeds). (c) F1 at full metadata mimicry and full text removal for the three ablations evaluated under perturbation (three seeds).",
 "fig11_calibration_selective": "Probability quality (five seeds): (a) ECE (log scale; hollow = RGF before temperature scaling), (b) Brier score, (c, d) error on the 80\\% and 90\\% most confident test accounts.",
 "fig12_computational_cost": "CPU cost (five seeds): (a) training time, (b) inference time per 1,000 accounts, (c) parameters (neural) or trees (tree models). RGF uses 38,089 of 50,571 registered parameters.",
}


def main():
    body = []
    for name, cap in CAPTIONS.items():
        body.append("\\begin{figure*}[p]\\centering\\includegraphics[width=\\textwidth]{" + str(FIG / name) +
                    ".pdf}\\caption{" + cap + "}\\end{figure*}")
    tex = ("\\documentclass[9pt,twocolumn]{extarticle}\\usepackage[letterpaper,margin=0.7in,columnsep=0.24in]{geometry}"
           "\\usepackage{times}\\usepackage[T1]{fontenc}\\usepackage[utf8]{inputenc}\\usepackage{amsmath,graphicx}"
           "\\renewcommand{\\thetable}{\\Roman{table}}\\begin{document}\n" + "\n".join(body) +
           "\n\\clearpage\n\\input{" + str(TAB / "all_tables.tex") + "}\n\\end{document}")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "p.tex").write_text(tex, encoding="utf-8")
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "p.tex"], cwd=d, capture_output=True)
        shutil.copy(Path(d) / "p.pdf", ROOT / "VISUAL_PACKAGE_PREVIEW.pdf")
    print("written", ROOT / "VISUAL_PACKAGE_PREVIEW.pdf")


if __name__ == "__main__":
    main()
