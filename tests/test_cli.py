"""End-to-end CLI test: run.main -> runs.csv -> report.build, with stubs in place
of PyTorch and LightGBM (unavailable in the build environment)."""
import pandas as pd

from botrecon import provenance as P


def test_cli_smoke_end_to_end(twibot_root, cresci_root, tmp_path, monkeypatch, stub_gcn, stub_clf):
    import botrecon.classifier as clf_mod
    import botrecon.gcn as gcn_mod
    from botrecon import run, report

    monkeypatch.setattr(gcn_mod, "train_gcn", lambda *a, device=None, **k: stub_gcn(*a, **k))
    monkeypatch.setattr(clf_mod, "LGBMStage", stub_clf)
    out = tmp_path / "runs.csv"
    common = ["--out", str(out), "--pred-dir", str(tmp_path / "preds"),
              "--cache-dir", str(tmp_path / "cache"), "--smoke", "--seeds", "0", "1"]
    run.main(["--dataset", "twibot20", "--data-root", twibot_root[0], "--family", "corrected",
              "--only", "text=B"] + common)
    run.main(["--dataset", "cresci2017", "--data-root", cresci_root[0], "--family", "literal",
              "--only", "fusion=P3,tap=prob"] + common)
    df = pd.read_csv(out)
    assert len(df) == 4 * 2 + 8 * 2
    assert (df.provenance == P.SMOKE_TEST).all()          # smoke runs can never pass as results
    assert df.pred_file.notna().all()
    assert (tmp_path / "cache").exists()
    md = report.build(str(out))                           # smoke rows excluded from tables
    assert P.REPRODUCED_CORRECTED + " |" not in md


def test_resume_and_time_budget(twibot_root, tmp_path, monkeypatch, stub_gcn, stub_clf):
    import botrecon.classifier as clf_mod
    import botrecon.gcn as gcn_mod
    from botrecon import run

    monkeypatch.setattr(gcn_mod, "train_gcn", lambda *a, device=None, **k: stub_gcn(*a, **k))
    monkeypatch.setattr(clf_mod, "LGBMStage", stub_clf)
    out = tmp_path / "runs.csv"
    base = ["--dataset", "twibot20", "--data-root", twibot_root[0], "--family", "corrected",
            "--only", "text=B", "--smoke", "--out", str(out), "--pred-dir", str(tmp_path / "p"),
            "--cache-dir", str(tmp_path / "c"), "--seeds", "0", "1", "2"]
    run.main(base + ["--max-minutes", "0"])            # budget already spent: nothing runs
    assert not out.exists()
    run.main(base)                                       # 4 configs x 3 seeds
    assert len(pd.read_csv(out)) == 12
    run.main(base)                                       # resume: nothing new
    assert len(pd.read_csv(out)) == 12
    df = pd.read_csv(out)
    assert df.run_tag.is_unique
