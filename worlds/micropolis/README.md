# Micropolis world

This world adds simulation, question generation, cached gathering and binary/continuous forecast datasets to the existing workspace. It derives from [Fabio Rocha's Micropolis branch](https://github.com/fdrocha/forecastbench-sim/tree/8c70972b698f079d54b8a2255c81fbefc55e7d99/worlds/micropolis), source commit `8c70972b698f079d54b8a2255c81fbefc55e7d99`. See [PROVENANCE.json](PROVENANCE.json) for source hashes and adaptations; the source repository's GPL-3.0 text is retained in [LICENSE](LICENSE).

`fbsim-core` owns the shared schema/resolver. `fbsim-benchmark` owns the existing Micropolis parsers, raw replay scoring and cached trajectory converter; this world imports them. The five continuous quantiles are p10/p25/p50/p75/p90. Binary resolution uses half-open windows `(snapshot, resolution]` and engine turns (`tick // 16`). Original question/resolver bodies are retained. Their original specification remains at the pinned source's `binary_forecasts.md`.

Paper normalization, model panels, administered prompt wrappers, knowledge statement sets, historical configurations, frozen data and paper CSV/figure producers are not part of this world. The included prompt wrappers and example configs are new illustrative examples; they cannot reproduce historical prompt hashes or results. The paper's production configs, prompts, model panels and knowledge statements are in [forecastbench-sim-paper/production/micropolis](https://github.com/forecastingresearch/forecastbench-sim-paper/tree/main/production/micropolis). No simulator, maps or engine binary is vendored.

## Python installation and offline checks

From the repository root, Python 3.11+ and uv are required. The workspace already discovers `worlds/*`. To install just core, benchmark and this world as non-editable wheels:

```sh
uv venv --python 3.13 .venv-micropolis
uv run --no-project --python .venv-micropolis/bin/python packages/fbsim-benchmark/tools/build_wheels.py --output dist/micropolis-wheels
uv build --wheel --no-sources --out-dir dist/micropolis-wheels worlds/micropolis
uv pip install --python .venv-micropolis/bin/python --find-links dist/micropolis-wheels 'fbsim-core==0.1.0' 'fbsim-benchmark[test]==0.3.0' 'micropolis-world[dev]==0.1.0'
uv pip check --python .venv-micropolis/bin/python
```

For cached offline installation, add `--offline` to all uv commands (including the build helper). An optional `--find-links /path/to/wheelhouse` on installation supplies locally available dependencies. Offline mode fails on missing dependencies; it does not download. These packages do not install the engine. Full `uv sync` additionally installs other worlds' dependencies.

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True uv run --offline --no-project --python .venv-micropolis/bin/python -m pytest -q worlds/micropolis/tests
uv run --offline --no-project --python .venv-micropolis/bin/python -m pytest -q --import-mode=importlib packages/fbsim-core/tests packages/fbsim-benchmark/tests
uv run --offline --no-project --python .venv-micropolis/bin/python worlds/micropolis/scripts/run_sim.py --help
```

Tests block engine launches and provider transports. Corpus tests use invented trajectory rows; the process-contract test mocks pnpm. Neither is a live-engine validation. CLI `--help`, cached parsing/scoring and imports need no engine or credentials. A missing engine is reported when a runner is invoked, rather than during import. `credentials/.env` (repo root) is read only when a runner needs provider keys; variables exported in the shell take precedence, and no cloud secret store is used.

## Engine

The engine is vendored in `MicropolisCore/`: a trimmed copy of [fdrocha/MicropolisCore](https://github.com/fdrocha/MicropolisCore) (branch `fbs`) whose upstream commit, omissions and re-vendoring steps are in `MicropolisCore/VENDORED.md`. It includes the prebuilt engine, so running the sims needs only Node >=20 and pnpm 10.28.0 (the declared packageManager), installed once:

```sh
cd worlds/micropolis/MicropolisCore
pnpm install --frozen-lockfile
```

Rebuilding the engine (`pnpm run build:engine`) also needs make and an activated Emscripten toolchain providing `em++`. The engine is found next to the package source, so runners need the workspace (editable) install; the non-editable wheels above do not ship it. Engine assets keep their own licenses and additional terms (`MicropolisCore/LICENSE`, `MicropolisCore/MicropolisPublicNameLicense.md`); this package grants no new rights to them.

## Entrypoints and effects

All script paths are under `worlds/micropolis/scripts/`. Use `uv run --no-project --python .venv-micropolis/bin/python <script> ...` from the repository root. Each configuration-driven command accepts an explicit JSON5 file; continuous commands default to the invented `configs/example.json5` (six turns, horizon four). Binary evaluation and continuation extraction default to `configs/example-binary.json5` (97 logged turns, snapshot 48, horizon 48, one continuation seed), also shipped inside the installed package. Binary resolution requires a snapshot and horizon of at least 48 turns to include a yearly baseline and resolution checkpoint; the short continuous example is intentionally invalid for those two commands. These defaults are illustrative, not historical configurations. Replace `example/model` with a deliberately selected provider model before any paid evaluation.

| Script | What it does |
|---|---|
| `run_sim.py CONFIG --no-plot` | Runs the engine and writes log/event files; optional plot |
| `gen_corpus.py CONFIG` | Runs/reuses city trajectories, generates and resolves continuous questions |
| `gen_report.py CONFIG` | Produces a situation report; reads cached trajectory files |
| `generate_prompt.py CONFIG` | Builds an example continuous prompt; may run simulations on cache misses |
| `run_eval_binary.py CONFIG`, `run_eval_continuous.py CONFIG` | Builds corpus and gathers model responses, then writes the respective dataset |
| Same eval scripts with `--cache-only` | No provider calls; reads existing raw-response cache, but corpus construction can still simulate |
| Same eval scripts with `--dry-run` | Builds corpus without gathering; **can still simulate** |
| `extract_ground_truth.py CONFIG --jobs 1` | Runs reseeded continuations through external `run_continuations.js`; not an offline smoke command. Writes `ground_truth/<scenario>_T<snapshot>.jsonl`, one line per horizon: Yes `counts` and `yes_idx` (the 0-based continuations resolving Yes) per question, and per-continuation `values` per metric |
| `draw_natcond.py [--n 400] [--seed 42]` | Draws natural-conditional cells (question A at S+horizon_A, revealed fact B at the horizon just before it; `--max-gap` widens this) from the ground-truth `yes_idx`, after FreeCiv's draw: 0.05 < P(A) ≤ 0.95, B Yes in ≥ 100 of 1,000 replays, 0.02 < P(A\|B) < 0.98; irrelevant / mechanical / inferential blocks, template-balanced, ≤ 2 reveals per A and ≤ 8 cells per revealed event. Writes `natcond/natcond_<n>_seed<seed>.jsonl`; no engine or provider |
| `build_natcond_prompts.py [--draw FILE] [--show N]` | Builds the natural-conditional prompts after FreeCiv's: turn 1 asks question A alone, turn 2 continues the conversation with the revealed fact (no no-news control; FreeCiv's covers it). The world report comes from the binary eval's cached prompt. Writes `<draw>_prompts.jsonl`; no provider calls |
| `elicit_natcond.py --model-specs SPECS [--models LIST] [--limit N]` | Asks models the natural-conditional prompts: turn 1 per question, then turn 2 per cell on top of the model's own turn-1 reply; unparsed replies re-asked once. `--models` defaults to the 22-model panel (`configs/models-22.json5`); every model needs a registry entry (provider pin, effort). 16 calls in flight (`--concurrency`), 8 retries per call (`--retries`). Appends to `<draw>_elicit/results.jsonl` and resumes from it, re-asking prompts whose messages changed; **makes paid calls** (`--dry-run` counts them) |
| `check_determinism.py CONFIG` | Repeats live engine runs; opt-in only |
| `score_cached.py --input DATASET --kind binary\|continuous` | Reads gathered JSON and prints per-forecast realized Brier or raw quantile CRPS; no engine/provider/normalization |

The example configs write to their own `data/micropolis-example*` folders (their `data_dir`), apart from `data/micropolis`:

```sh
uv run --no-project --python .venv-micropolis/bin/python worlds/micropolis/scripts/run_sim.py worlds/micropolis/configs/example.json5 --no-plot
uv run --no-project --python .venv-micropolis/bin/python worlds/micropolis/scripts/gen_corpus.py worlds/micropolis/configs/example.json5
```

The config's `data_dir` selects one child of the repository's `data/` (a symlink set up by `scripts/setup_worktree.sh`). Cached gathering preserves the native content-addressed prompt/response scheme. Missing cached forecasts remain absent, unparsed forecasts remain null: the raw scorer does not apply paper eligibility or imputation rules.

Actual model gathering requires `OPENROUTER_API_KEY`, from `credentials/.env` or the shell. No GCP fallback is attempted. Optionally set `MICROPOLIS_MODEL_SPECS` to your own JSON5 routing/spec file before invoking Python; the installed registry is empty. An unqualified model uses provider defaults, while a suffixed slug must have a configured spec. No production model panel is supplied, and no model call was made to validate this integration.
