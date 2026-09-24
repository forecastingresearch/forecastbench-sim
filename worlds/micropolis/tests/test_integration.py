"""Installation/entrypoint boundaries and end-to-end synthetic cached inputs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import micropolis_world.module_globals as g
from micropolis_world.city_sim import CitySimulation

ROOT = Path(__file__).resolve().parents[1]


def test_all_entrypoints_help_without_engine_or_keys(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY") and k not in {"MICROPOLIS_CORE_PATH", "PYTHONPATH"}}
    env.update(FBSIM_DATA_ROOT=str(tmp_path), LITELLM_LOCAL_MODEL_COST_MAP="True")
    for script in sorted((ROOT / "scripts").glob("*.py")):
        result = subprocess.run([sys.executable, str(script), "--help"], env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (script.name, result.stderr)
    assert list(tmp_path.iterdir()) == []


def test_missing_engine_fails_at_invocation(monkeypatch):
    monkeypatch.delenv("MICROPOLIS_CORE_PATH", raising=False)
    with pytest.raises(RuntimeError, match="MICROPOLIS_CORE_PATH"):
        g.engine_app_path()


def test_incompatible_engine_explains_missing_producer(monkeypatch, tmp_path):
    cli = tmp_path / "apps/micropolis/cli"
    cli.mkdir(parents=True)
    (cli / "run_sim.js").write_text("// synthetic interface fixture")
    monkeypatch.setenv("MICROPOLIS_CORE_PATH", str(tmp_path))
    assert g.engine_app_path() == cli.parent
    with pytest.raises(RuntimeError, match="continuation producer"):
        g.engine_app_path(continuations=True)


def test_credentials_are_explicit(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        g.ensure_api_keys()


def test_live_runner_command_contract(monkeypatch, tmp_path):
    """Mocked process contract only, not a live engine smoke."""
    cli = tmp_path / "apps/micropolis/cli"
    cli.mkdir(parents=True)
    (cli / "run_sim.js").write_text("// synthetic interface fixture")
    monkeypatch.setenv("MICROPOLIS_CORE_PATH", str(tmp_path))
    monkeypatch.setattr(g, "RUNS_DIR", tmp_path / "out")
    # Autouse guard blocks CitySimulation.run. Test the original implementation
    # through its source module with a fake subprocess, never executing pnpm.
    spec = importlib.util.spec_from_file_location("micropolis_world._runner_test", ROOT / "micropolis_world/city_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def fake_run(args, **kw):
        calls.append((args, kw))
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.CitySimulation("kyoto", 7).run(6)
    args, kw = calls[0]
    assert args[:3] == ["pnpm", "run", "run-sim"]
    assert args[args.index("--turns") + 1] == "6"
    assert "--no-disasters" in args
    assert kw["cwd"] == cli.parent


def test_cached_pipeline(tmp_path, synthetic_corpus):
    """Real corpus/IO/parsers/scorers; only the trajectory source is invented."""
    from micropolis_world.scenarios import build_corpus, build_batch_prompt_continuous, parse_batch_percentiles
    from micropolis_world.continuous_eval import Response, ResponseId, save_dataset
    from micropolis_world.gather import EvalPaths, gather_raw_responses, group_into_batches
    from micropolis_world.cache_keys import prompt_hash
    corpus = build_corpus([CitySimulation("kyoto", 7)], [1], [4], 1, "synthetic")
    model = "example/model"
    paths = EvalPaths("continuous")
    for bid, batch in group_into_batches(corpus).items():
        prompt = build_batch_prompt_continuous(batch[0]["context"], batch)
        path = paths.response_path(bid, model, prompt_hash(prompt))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(f"Q{i}: p10=1,p25=2,p50=3,p75=4,p90=5" for i in range(1, len(batch)+1)))
    batches, raw, _ = gather_raw_responses(corpus, [model], paths=paths, build_prompt=build_batch_prompt_continuous, cache_only=True)
    responses = {}
    for bid, batch in batches.items():
        parsed = parse_batch_percentiles(raw[(bid, model)], [q["question_id"] for q in batch])
        for question, value in zip(batch, parsed):
            responses[ResponseId(model, question["question_id"])] = Response(question["value"], value)
    data = tmp_path / "data.json"
    save_dataset(corpus, responses, [model], data)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/score_cached.py"), "--input", str(data), "--kind", "continuous"], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    assert len(rows) == len(corpus)
    assert all(r["status"] == "scored" and r["score"] >= 0 for r in rows)


def test_installed_default_reaches_engine_diagnostic(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in {"MICROPOLIS_CORE_PATH", "PYTHONPATH"}}
    env.update(FBSIM_DATA_ROOT=str(tmp_path), LITELLM_LOCAL_MODEL_COST_MAP="True")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_sim.py"), "--no-plot"], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Set MICROPOLIS_CORE_PATH" in result.stderr
    assert "config file not found" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_raw_binary_scoring_preserves_missingness(tmp_path):
    data = tmp_path / "binary.json"
    data.write_text(json.dumps({"questions": [{"question_id": "q", "answer": True}], "forecasts": [{"question_id": "q", "model_id": "a", "probability": 0.4}, {"question_id": "q", "model_id": "b", "probability": None}]}))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/score_cached.py"), "--input", str(data), "--kind", "binary"], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    assert rows[0]["metric"] == "realized_brier"
    assert rows[0]["score"] == pytest.approx(.36)
    assert rows[1]["status"] == "unparsed" and rows[1]["score"] is None
