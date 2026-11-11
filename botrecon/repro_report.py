"""Notebook step 19: write results/REPRODUCIBILITY_RUN.md from what actually ran (no numbers invented)."""
from __future__ import annotations

import json
import os
import time

from .env import check_environment
from .methods import make_all


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def completeness(main, datasets, seeds):
    """Expected vs finished runs per dataset and setting (main suite only)."""
    rows = []
    for ds in datasets:
        for setting in ("transductive", "inductive"):
            names = [m.name for m in make_all(ds, setting)]
            exp = len(names) * len(seeds)
            done = 0 if main is None or main.empty else int(((main.dataset == ds) & (main.setting == setting)).sum())
            missing = []
            if main is not None and not main.empty:
                for n in names:
                    k = int(((main.dataset == ds) & (main.setting == setting) & (main.method == n)).sum())
                    if k < len(seeds):
                        missing.append(f"{n} ({k}/{len(seeds)})")
            else:
                missing = names
            rows.append({"dataset": ds, "setting": setting, "expected": exp, "finished": done, "missing": missing})
    return rows


def write(ctx, path="results/REPRODUCIBILITY_RUN.md") -> str:
    from .suite import read_rows
    env = check_environment(strict=False)
    main = read_rows("results/suite_main.jsonl")
    comp = completeness(main, ctx.available, ctx.cfg["seeds"])
    audit = _load("results/protocol_audit.json") or []
    fails = [r for r in audit if r.get("status") == "FAIL"]
    tests = open("results/pytest.txt").read().strip().splitlines()[-1] if os.path.exists("results/pytest.txt") else "not run"
    lines = [f"# Reproducibility record ({time.strftime('%Y-%m-%d %H:%M %Z')})", "",
             "Independent reconstruction; not the original implementation. Numbers live in results/ and",
             "results/paper/; this file records how they were produced.", "",
             "## Environment", "", f"- Python {env['python']} on {env['platform']}",
             f"- CPU threads {env['cpu_count']}, RAM {env['ram_gb']} GB, GPU {env.get('gpu')}",
             "- Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
             f"- Missing packages: {env['missing'] or 'none'}", "",
             "## Configuration", "", "```json", json.dumps(ctx.cfg, indent=1), "```", "",
             "## Data", "", f"- Dataset paths: {json.dumps(ctx.paths)}",
             f"- Protocol audit: {len(audit)} checks, {len(fails)} failed" +
             ("".join(f"\n  - FAIL: {r['check']} ({r['evidence']})" for r in fails)), "",
             "## Tests", "", f"- pytest: {tests}", "",
             "## Completeness of the experiment suite", "", "| Dataset | Setting | Expected runs | Finished |",
             "| --- | --- | --- | --- |"]
    lines += [f"| {r['dataset']} | {r['setting']} | {r['expected']} | {r['finished']} |" for r in comp]
    pend = [f"- {r['dataset']} / {r['setting']}: " + ", ".join(r["missing"][:12]) + (" ..." if len(r["missing"]) > 12 else "")
            for r in comp if r["missing"]]
    lines += ["", "## Still pending", ""] + (pend or ["- nothing: every configured run has finished"])
    lines += ["", "## Seeds", "", f"- main seeds {ctx.cfg['seeds']}; robustness/explainability seeds "
              f"{ctx.cfg['robust_seeds']}; checkpoints saved for seeds {ctx.cfg['checkpoint_seeds']}", "",
              "## Commands", "", "```", "python -m botrecon.plan --part main     --cresci <root> [--cresci2015 <root>]",
              "python -m botrecon.plan --part extended --cresci <root> [--cresci2015 <root>]",
              "python -m botrecon.suite_report --results results --out results/paper", "```"]
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    return text
