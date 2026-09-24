# Offline benchmark interfaces

This additive package provides cached-forecast validation/scoring, native response parsing and cached trajectory conversion for FreeCiv, Micropolis and Starsim. Existing core, world engines, public scripts and human-subject materials remain in their original locations. Paper normalization, exclusions and combined scores are not benchmark defaults.

## Install from the repository root

Use Python 3.11 or newer and uv. This installs the two local packages, not every world engine. Dependency installation may access the package index; use the offline form below when caches are already populated.

```sh
uv venv .venv-offline
uv run --no-project --python .venv-offline/bin/python packages/fbsim-benchmark/tools/build_wheels.py --output dist/offline-wheels
uv pip install --python .venv-offline/bin/python --find-links dist/offline-wheels 'fbsim-core==0.1.0' 'fbsim-benchmark[test]==0.3.0'
```

For already populated caches, add `--offline` to venv, run, build_wheels.py and pip install. A dependency wheelhouse can be supplied with an additional `--find-links /path/to/wheelhouse`. Without cached dependencies offline installation fails instead of downloading. The build helper copies both packages to temporary directories before invoking the standard build backend. This avoids editing the core's historically tracked egg-info metadata and installs non-editable wheels. Direct workspace source installs may still be editable and dirty that metadata; they are not this quickstart. World engines and API keys are not required. The root workspace remains intact; its all-world installation is separate.

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

Installed contract/provenance documentation is available without the checkout:

```sh
uv run --offline --no-project --python .venv-offline/bin/python -m fbsim_benchmark docs --name NATIVE_INTERFACES.md
```

The command returns a JSON object containing the document text. Successful CLI output is standard JSON; nonfinite native parser values and score overflows cause an error exit with no JSON on stdout. Starsim `binary` is strict; `binary-historical` explicitly preserves the administered parser, including its numeric-prefix bug. Never use that compatibility mode to validate new probabilities.
