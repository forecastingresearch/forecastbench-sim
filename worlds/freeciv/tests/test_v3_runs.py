"""Offline run configuration, scoring and fleet failure contracts."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

V3 = Path(__file__).resolve().parents[1] / "fbsim_v3"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_paper_defaults_and_run_reports(tmp_path, monkeypatch):
    monkeypatch.delenv("FBSIM_RUN", raising=False)
    cfg = module(V3 / "runcfg.py", "test_runcfg")
    default = cfg.load()
    assert (default.n_players, default.fork_turn, default.horizons) == (5, 60, (90, 120, 150, 180, 210))
    assert cfg.background(cfg.PAPER_BACKGROUND, default) == cfg.PAPER_BACKGROUND
    assert cfg.no_news(cfg.PAPER_NO_NEWS, default) == cfg.PAPER_NO_NEWS
    root = tmp_path / "example"
    root.mkdir()
    (root / "run.env").write_text("FBSIM_N_PLAYERS=1\nFBSIM_FORK_TURN=120\nFBSIM_HORIZONS=150,180,210,240,270\nFBSIM_NATCOND=0\n")
    monkeypatch.setattr(cfg, "runs_root", lambda: tmp_path)
    run = cfg.load("example")
    assert run.report_name == "turn_120_report.txt"
    assert cfg.background(cfg.PAPER_BACKGROUND, run) == "observed at turn 120. One AI civilization is playing alone."
    assert "121 through 150" in cfg.no_news(cfg.PAPER_NO_NEWS, run)
    assert not run.natcond


def test_unparsed_scores_keep_parse_denominator(monkeypatch):
    monkeypatch.delenv("FBSIM_RUN", raising=False)
    monkeypatch.setattr(sys, "argv", ["score_v2.py"])
    scorer = module(V3 / "run/score_v2.py", "test_run_scorer")
    items = {i: dict(set="bank", qAll=0.2, world="invented", family="invented", T=90) for i in ("a", "b")}
    forecasts = {("model", "a", "t1"): {"value": 0.2}, ("model", "b", "t1"): {"value": None}}
    rows = scorer.score_items(forecasts, items, {}, [], None)
    assert len(rows) == 2
    assert [r["parsed"] for r in rows] == [1, 0]
    assert "excess_brier" not in rows[1]
    legacy = scorer.score_items(forecasts, items, {}, [], 0.5)
    assert legacy[1]["excess_brier"] == pytest.approx(0.09)


@pytest.mark.parametrize("setup_rc,expected", [("0", 0), ("3", 1), ("", 1)])
def test_bringup_propagates_setup_failure(tmp_path, setup_rc, expected):
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    shutil.copyfile(V3 / "fleet/bringup.sh", fleet / "bringup.sh")
    (fleet / "paths.sh").write_text('STATE="$HERE/state"\nBUNDLE="$HERE/bundle"\nSSHKEY="$HERE/key"\nARCHKEY="$HERE/absent"\nRUNDIR="$HERE/run"\nFBSIM_FORK_TURN=60\n')
    (fleet / "rp.sh").write_text("echo 'invented-host 22'\n")
    for directory in ("state", "bundle", "run"):
        (fleet / directory).mkdir()
    (fleet / "run/run.env").write_text("")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    commands = {"ssh": '#!/bin/bash\ncase "${*: -1}" in *"cat /workspace/fleet/logs/pod_setup.rc"*) printf "%s" "$TEST_SETUP_RC";; esac\n',
                "rsync": "#!/bin/bash\nexit 0\n", "sleep": "#!/bin/bash\nexit 0\n"}
    for name, body in commands.items():
        script = fakebin / name
        script.write_text(body)
        script.chmod(0o755)
    env = dict(os.environ, PATH=f"{fakebin}:{os.environ['PATH']}", TEST_SETUP_RC=setup_rc)
    result = subprocess.run(["bash", str(fleet / "bringup.sh"), "invented-pod"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stdout + result.stderr
    assert ("bringup done" in result.stdout) == (expected == 0)
