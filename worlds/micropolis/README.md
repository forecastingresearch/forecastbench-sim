# Micropolis world

This world adds simulation, question generation, cached gathering and binary/continuous forecast datasets to the existing workspace. It derives from [Fabio Rocha's Micropolis branch](https://github.com/fdrocha/forecastbench-sim/tree/8c70972b698f079d54b8a2255c81fbefc55e7d99/worlds/micropolis), source commit `8c70972b698f079d54b8a2255c81fbefc55e7d99`. See [PROVENANCE.json](PROVENANCE.json) for source hashes and adaptations; the source repository's GPL-3.0 text is retained in [LICENSE](LICENSE).

`fbsim-core` owns the shared schema/resolver. `fbsim-benchmark` owns the existing Micropolis parsers, raw replay scoring and cached trajectory converter; this world imports them. The five continuous quantiles are p10/p25/p50/p75/p90. Binary resolution uses half-open windows `(snapshot, resolution]` and engine turns (`tick // 16`). Original question/resolver bodies are retained. Their original specification remains at the pinned source's `binary_forecasts.md`.

Paper normalization, model panels, administered prompt wrappers, knowledge statement sets, historical configurations, frozen data and paper CSV/figure producers are not part of this world. The included prompt wrappers and six-turn config are new illustrative examples; they cannot reproduce historical prompt hashes or results. Existing private paper work remains separate. No simulator, maps or engine binary is vendored.

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

Tests block engine launches and provider transports. Corpus tests use invented trajectory rows; the process-contract test mocks pnpm. Neither is a live-engine validation. CLI `--help`, cached parsing/scoring and imports need no engine path or credentials. A missing engine is reported when a runner is invoked, rather than during import. No `.env` file or cloud secret store is loaded implicitly.

## External engine setup

Use [fdrocha/MicropolisCore, exploration](https://github.com/fdrocha/MicropolisCore/tree/4fe3a7238b735f9cdd3ec13c957ccc8a3a3b0481), pinned for this integration's **source-interface inspection** to `4fe3a7238b735f9cdd3ec13c957ccc8a3a3b0481`. It contains both `apps/micropolis/cli/run_sim.js` and `run_continuations.js`. The `fbs` branch at `d50b799d5915944263c0385fc43a8990e5d4e32e` has run_sim but lacks run_continuations; it is not a substitute for ground-truth extraction.

Prerequisites in that engine revision: Node >=20, pnpm 10.28.0 (declared packageManager), make, and an activated Emscripten toolchain providing `em++`. The engine makefile uses Emscripten APIs including `--emit-tsd`; no exact tested Emscripten version or historical producer-engine pin was established in this port. Its source specifies `pnpm run build:engine` → `make install`. Setup below is for a separately provisioned engine checkout and may download/build substantial dependencies; it was **not executed by the offline integration tests**:

```sh
# In a separately obtained fdrocha/MicropolisCore checkout:
git checkout 4fe3a7238b735f9cdd3ec13c957ccc8a3a3b0481
pnpm install --frozen-lockfile
pnpm run build:engine
export MICROPOLIS_CORE_PATH="$PWD"
```

Return to forecastbench-sim before the following commands. `MICROPOLIS_CORE_PATH` names the engine root, not apps/micropolis. Engine assets and their licenses/additional terms remain in that external repository; this package grants no new rights to them. Live engine/toolchain compatibility remains a runtime gate, not a claim about historical reproducibility.

## Entrypoints and effects

All script paths are under `worlds/micropolis/scripts/`. Use `uv run --no-project --python .venv-micropolis/bin/python <script> ...` from the repository root. Each configuration-driven command accepts an explicit JSON5 file; the default is the invented `configs/example.json5`. Replace `example/model` with a deliberately selected provider model before any paid evaluation.

| Script | What it does |
|---|---|
| `run_sim.py CONFIG --no-plot` | Runs the external engine and writes log/event files; optional plot |
| `gen_corpus.py CONFIG` | Runs/reuses city trajectories, generates and resolves continuous questions |
| `gen_report.py CONFIG` | Produces a situation report; reads cached trajectory files |
| `generate_prompt.py CONFIG` | Builds an example continuous prompt; may run simulations on cache misses |
| `run_eval_binary.py CONFIG`, `run_eval_continuous.py CONFIG` | Builds corpus and gathers model responses, then writes the respective dataset |
| Same eval scripts with `--cache-only` | No provider calls; reads existing raw-response cache, but corpus construction can still simulate |
| Same eval scripts with `--dry-run` | Builds corpus without gathering; **can still simulate** |
| `extract_ground_truth.py CONFIG --jobs 1` | Runs reseeded continuations through external `run_continuations.js`; not an offline smoke command |
| `check_determinism.py CONFIG` | Repeats live engine runs; opt-in only |
| `score_cached.py --input DATASET --kind binary\|continuous` | Reads gathered JSON and prints per-forecast realized Brier or raw quantile CRPS; no engine/provider/normalization |

Choose a fresh output root before running:

```sh
export FBSIM_DATA_ROOT=/path/to/new/run-data
uv run --no-project --python .venv-micropolis/bin/python worlds/micropolis/scripts/run_sim.py worlds/micropolis/configs/example.json5 --no-plot
uv run --no-project --python .venv-micropolis/bin/python worlds/micropolis/scripts/gen_corpus.py worlds/micropolis/configs/example.json5
```

The config's `data_dir` selects one child of `FBSIM_DATA_ROOT`; absent the environment variable, the root is `./data` relative to the working directory. Cached gathering preserves the native content-addressed prompt/response scheme. Missing cached forecasts remain absent, unparsed forecasts remain null: the raw scorer does not apply paper eligibility or imputation rules.

Actual model gathering requires explicitly exporting `OPENROUTER_API_KEY`. No GCP fallback is attempted. Optionally set `MICROPOLIS_MODEL_SPECS` to your own JSON5 routing/spec file before invoking Python; the installed registry is empty. An unqualified model uses provider defaults, while a suffixed slug must have a configured spec. No production model panel is supplied, and no model call was made to validate this integration.
