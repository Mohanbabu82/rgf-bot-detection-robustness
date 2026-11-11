import json
import os
import shutil
import zipfile

import pytest

from botrecon.datapaths import extract_all, find_cresci, find_cresci2015, find_twibot, locate
from botrecon.env import check_environment
from botrecon.settings import load_config


def test_environment_report_and_missing_package_error():
    info = check_environment({"numpy": "numpy", "sklearn": "scikit-learn"}, strict=True)
    assert info["packages"]["numpy"] and info["cpu_count"] >= 1 and info["missing"] == []
    with pytest.raises(RuntimeError, match="pip install definitely-not-a-package"):
        check_environment({"definitely_not_a_module": "definitely-not-a-package"}, strict=True)
    assert check_environment({"definitely_not_a_module": "x"}, strict=False)["missing"] == ["x"]


def test_locate_plain_folders(cresci_root, cresci2015_root, tmp_path):
    inp = tmp_path / "input"; inp.mkdir()
    shutil.copytree(cresci_root[0], inp / "cresci-2017.csv" / "datasets_full.csv")
    shutil.copytree(cresci2015_root[0], inp / "cresci-2015" / "cresci-2015.csv")
    out = locate(str(inp), str(tmp_path / "work"))
    assert out["cresci2017"].endswith("datasets_full.csv") and out["cresci2015"].endswith("cresci-2015.csv")
    assert "twibot20" not in out
    assert find_cresci2015(str(inp / "cresci-2017.csv")) is None        # Cresci-2017 is not mistaken for 2015


def test_locate_cresci2015_targz_with_inner_zips(cresci2015_root, tmp_path):
    import tarfile
    inner = tmp_path / "inner"; inner.mkdir()
    for g in os.listdir(cresci2015_root[0]):
        shutil.make_archive(str(inner / g), "zip", cresci2015_root[0], g)
    inp = tmp_path / "input"; (inp / "c15").mkdir(parents=True)
    with tarfile.open(inp / "c15" / "cresci-2015.csv.tar.gz", "w:gz") as t:
        for f in os.listdir(inner):
            t.add(inner / f, f"cresci-2015.csv/{f}")
    (inp / "code").mkdir(); shutil.make_archive(str(inp / "code" / "botrecon_repo"), "zip", str(inner))
    out = locate(str(inp), str(tmp_path / "work"))
    assert out["cresci2015"] and os.path.exists(os.path.join(out["cresci2015"], "FSF.csv", "users.csv"))
    assert not os.path.exists(tmp_path / "work" / "datasets" / "E13.csv")   # the botrecon_* archive is not extracted
    assert locate(str(inp), str(tmp_path / "work"))["notes"] == []           # second call: nothing re-extracted


def test_locate_nested_zips_like_the_official_release(cresci_root, tmp_path):
    inp = tmp_path / "input"; (inp / "ds").mkdir(parents=True)
    inner_dir = tmp_path / "inner"; inner_dir.mkdir()
    for g in os.listdir(cresci_root[0]):                       # each group zipped, as in the release
        shutil.make_archive(str(inner_dir / g), "zip", cresci_root[0], g)
    with zipfile.ZipFile(inp / "ds" / "cresci-2017.csv.zip", "w") as z:
        for f in os.listdir(inner_dir):
            z.write(inner_dir / f, f"datasets_full.csv/{f}")
    out = locate(str(inp), str(tmp_path / "work"))
    assert out["cresci2017"] and os.path.exists(os.path.join(out["cresci2017"], "genuine_accounts.csv", "users.csv"))
    assert out["cresci2015"] is None and out["notes"]


def test_locate_explicit_paths_and_errors(cresci_root, tmp_path):
    out = locate(str(tmp_path), str(tmp_path), cresci=cresci_root[0])
    assert out["cresci2017"] == cresci_root[0]
    with pytest.raises(FileNotFoundError):
        locate(str(tmp_path), str(tmp_path), cresci2015=str(tmp_path / "missing"))


def test_config_defaults_and_validation(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"_comment": "x", "seeds": [0, 1], "robust_seeds": [0], "checkpoint_seeds": [0],
                             "paths": {"cresci2017": "/x"}}))
    c = load_config(str(p))
    assert c["paths"] == {"cresci2017": "/x", "cresci2015": None} and c["text_variant"] == "B"
    p.write_text(json.dumps({"seeds": [0], "robust_seeds": [3]}))
    with pytest.raises(AssertionError):
        load_config(str(p))


def test_repository_config_file_is_valid():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c = load_config(os.path.join(here, "config", "kaggle_cpu.json"))
    assert c["input_dir"] == "/kaggle/input" and c["seeds"] == [0, 1, 2, 3, 4]


def _nest(src_root, dst_root, groups=None):
    """Kaggle layout after unpacking archives inside archives: <g>.csv/<g>.csv/users.csv"""
    for g in os.listdir(src_root):
        if groups is None or g.replace(".csv", "") in groups:
            shutil.copytree(os.path.join(src_root, g), os.path.join(dst_root, g, g))


def test_kaggle_double_nested_layout_and_incomplete_copy(cresci_root, cresci2015_root, tmp_path):
    from botrecon.data import load_cresci2015, load_cresci2017
    inp = tmp_path / "input"
    # an incomplete third-party copy (2 of 9 groups), like the one seen on Kaggle
    _nest(cresci_root[0], str(inp / "partial" / "cresci-2017.csv" / "datasets_full.csv"),
          groups={"genuine_accounts", "social_spambots_1"})
    out = locate(str(inp), str(tmp_path / "w1"))
    assert out["cresci2017"].endswith("datasets_full.csv")                   # not the inner group folder
    assert out["groups_found"]["cresci2017"] == 2 and any("INCOMPLETE" in n for n in out["notes"])
    # the complete official copy is preferred when both are attached
    _nest(cresci_root[0], str(inp / "full" / "cresci-2017.csv" / "datasets_full.csv"))
    _nest(cresci2015_root[0], str(inp / "c15" / "cresci-2015.csv"))
    out = locate(str(inp), str(tmp_path / "w2"))
    assert "full" in out["cresci2017"] and out["groups_found"] == {"cresci2017": 9, "cresci2015": 5}
    assert not any("INCOMPLETE" in n for n in out["notes"])
    d = load_cresci2017(out["cresci2017"])
    assert all(v["found"] for v in d.log["groups"].values()) and (d.group == "fake_followers").any()
    d15 = load_cresci2015(out["cresci2015"])
    assert set(d15.group) == {"E13", "TFP", "FSF", "INT", "TWT"} and d15.edges.shape[1] > 0
