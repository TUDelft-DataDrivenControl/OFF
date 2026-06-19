import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "03_Code"))

import off.off as off


def _make_off(run_id: int) -> off.OFF:
    off_obj = off.OFF.__new__(off.OFF)
    off_obj.__get_runid__ = lambda: run_id
    return off_obj


def test_dir_init_falls_back_to_cwd_when_off_path_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OFF_PATH", raising=False)
    monkeypatch.setenv("PWD", str(tmp_path / "not_used"))

    cwd = tmp_path / "workspace"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    off_obj = _make_off(run_id=1)
    off_obj.__dir_init__({"simulation folder": None})

    expected_sim_dir = cwd / "runs" / "off_run_1"
    assert Path(off_obj.sim_dir) == expected_sim_dir
    assert off_obj.root_dir == str(cwd)
    assert expected_sim_dir.exists()


def test_dir_init_uses_parent_when_cwd_is_03_code(tmp_path, monkeypatch):
    monkeypatch.delenv("OFF_PATH", raising=False)
    monkeypatch.setenv("PWD", str(tmp_path / "not_used"))

    code_dir = tmp_path / "project" / "03_Code"
    code_dir.mkdir(parents=True)
    monkeypatch.chdir(code_dir)

    off_obj = _make_off(run_id=2)
    off_obj.__dir_init__({"simulation folder": None})

    expected_sim_dir = tmp_path / "project" / "runs" / "off_run_2"
    assert Path(off_obj.sim_dir) == expected_sim_dir
    assert off_obj.root_dir == str(tmp_path / "project")
    assert expected_sim_dir.exists()
