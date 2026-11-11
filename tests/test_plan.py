import pandas as pd

from botrecon import plan


def test_closest_literal(tmp_path):
    """Internal reconstruction audit only (Cresci-2017): picks the literal config closest to the old number."""
    rows = []
    for acc, fusion in ((0.80, "P1"), (0.925, "P3"), (0.70, "P1")):
        rows.append({"dataset": "cresci2017", "provenance": "REPRODUCED-LITERAL", "fusion": fusion, "tap": "prob",
                     "text": "B", "cresci_knn_features": "meta", "twibot_split": "official",
                     "cresci_label_rule": "literal_bot_substring" if acc > 0.9 else "explicit", "test_accuracy": acc})
    p = tmp_path / "runs.csv"; pd.DataFrame(rows).to_csv(p, index=False)
    assert plan.closest_literal(str(p), "cresci2017") == \
        "fusion=P3,tap=prob,text=B,cresci_label_rule=literal_bot_substring,cresci_knn_features=meta"
    assert plan.closest_literal(str(tmp_path / "missing.csv"), "cresci2017") is None


def test_plan_order_and_budget(monkeypatch, tmp_path):
    import botrecon.run as R
    import botrecon.suite as S
    import botrecon.suite_report as SR
    calls = []
    def suite_call(argv):
        names = argv[argv.index("--methods") + 1:]
        names = names[:next((i for i, x in enumerate(names) if x.startswith("--")), len(names))]
        calls.append(("suite", argv[argv.index("--dataset") + 1], argv[argv.index("--settings") + 1],
                      "ablations" if all(n.startswith("RGF w") or n == "RGF concat fusion" for n in names) else "main"))
    monkeypatch.setattr(S, "main", suite_call)
    monkeypatch.setattr(R, "main", lambda argv: calls.append(("run", argv[argv.index("--dataset") + 1],
                                                               "--diagnostic" in argv)))
    monkeypatch.setattr(SR, "build", lambda a, b: "")
    monkeypatch.chdir(tmp_path)
    plan.main(["--cresci", "cr", "--cresci2015", "c15", "--minutes", "30"])
    assert calls == [("suite", "cresci2017", "transductive", "main"), ("suite", "cresci2015", "transductive", "main"),
                     ("suite", "cresci2017", "transductive", "ablations"),
                     ("suite", "cresci2015", "transductive", "ablations"),
                     ("suite", "cresci2017", "inductive", "main"), ("suite", "cresci2015", "inductive", "main")]
    assert not any(c[0] == "run" for c in calls)       # no reconstruction of the earlier manuscript by default
    assert any(f.name.startswith("plan_") for f in (tmp_path / "results" / "logs").iterdir())
    calls.clear()
    plan.main(["--cresci", "cr", "--cresci2015", "c15", "--minutes", "30", "--part", "main"])
    assert calls == [("suite", "cresci2017", "transductive", "main"), ("suite", "cresci2015", "transductive", "main")]
    calls.clear()
    plan.main(["--cresci", "cr", "--cresci2015", "c15", "--minutes", "30", "--part", "main", "--with-reconstruction"])
    assert ("run", "cresci2017", False) in calls and not any(c[1] == "cresci2015" and c[0] == "run" for c in calls)
    calls.clear()
    plan.main(["--cresci", "cr", "--cresci2015", "c15", "--minutes", "1"])     # budget already below the reserve
    assert calls == []


def test_plan_cresci_only(monkeypatch, tmp_path):
    import botrecon.run as R
    import botrecon.suite as S
    import botrecon.suite_report as SR
    calls = []
    monkeypatch.setattr(S, "main", lambda argv: calls.append(("suite", argv[argv.index("--dataset") + 1],
                                                               "--cross-root" in argv)))
    monkeypatch.setattr(R, "main", lambda argv: calls.append(("run", argv[argv.index("--dataset") + 1], False)))
    monkeypatch.setattr(SR, "build", lambda a, b: "")
    monkeypatch.chdir(tmp_path)
    plan.main(["--cresci", "cr", "--minutes", "30"])
    assert calls[:2] == [("suite", "cresci2017", False), ("suite", "cresci2017", False)]
    assert all(c[1] == "cresci2017" for c in calls)


def test_plan_passes_the_partner_dataset_for_cross_tests(monkeypatch, tmp_path):
    import botrecon.suite as S
    import botrecon.suite_report as SR
    calls = []
    monkeypatch.setattr(S, "main", lambda argv: calls.append((argv[argv.index("--dataset") + 1],
                                                              argv[argv.index("--cross-root") + 1])))
    monkeypatch.setattr(SR, "build", lambda a, b: "")
    monkeypatch.chdir(tmp_path)
    plan.main(["--cresci", "cr", "--cresci2015", "c15", "--minutes", "30", "--part", "main"])
    assert calls == [("cresci2017", "c15"), ("cresci2015", "cr")]
