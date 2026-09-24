# Offline benchmark interfaces

This additive package provides cached-forecast validation/scoring, native response parsing and cached trajectory conversion for FreeCiv, Micropolis and Starsim. Existing core, world engines, public scripts and human-subject materials remain in their original locations. Paper normalization, exclusions and combined scores are not benchmark defaults.

## Install from the repository root

Use Python 3.11 or newer and uv. This installs the two local packages, not every world engine. Dependency installation may access the package index; use the offline form below when caches are already populated.

```sh
uv venv .venv-offline
uv pip install --python .venv-offline/bin/python ./packages/fbsim-core './packages/fbsim-benchmark[test]'
```

For an already populated uv cache, add `--offline` to both commands. For a local wheelhouse, add `--find-links /path/to/wheelhouse` to the install command. Without cached dependencies the offline command fails; it never silently downloads. World simulators and API keys are not required. The root uv workspace remains intact; its all-world installation is a separate workflow.

## Run the synthetic fixtures, offline

All paths below are relative to the repository root. These commands use the installed packages, not PYTHONPATH.

```sh
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark smoke
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark validate --input packages/fbsim-benchmark/fixtures/freeciv.json
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark validate --input packages/fbsim-benchmark/fixtures/micropolis.json
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark validate --input packages/fbsim-benchmark/fixtures/starsim.json
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark parse --world micropolis --format binary-single --input packages/fbsim-benchmark/fixtures/micropolis-response.txt --options '{"quiet":true}'
uv run --offline --no-project --python .venv-offline/bin/python -m pytest -q --import-mode=importlib packages/fbsim-core/tests packages/fbsim-benchmark/tests
```

The three validation examples return excess Brier scores approximately 0.04, 0.09 and 0.01 respectively. Floating-point display may include trailing digits. The parser example returns 0.75. Tests also exercise all three worlds' native parsers, cached converters and quantile scoring, without live simulation or model calls. See [semantics](docs/SEMANTICS.md), [native interfaces](docs/NATIVE_INTERFACES.md), [workflow map](docs/WORKFLOWS.md) and [provenance](docs/PROVENANCE.json). Exact paper reproduction requires the separate private paper package and versioned data; none is implied to be publicly available here.
