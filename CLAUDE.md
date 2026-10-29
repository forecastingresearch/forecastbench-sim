Always use `uv run` to execute Python scripts in this project, not bare `python` or `python3`.

## Data and credentials

`data/` and `credentials/` at the repository root are symlinks to directories outside the checkout (README, "Local
data and credentials"; `scripts/setup_worktree.sh`). Never commit their contents or replace the symlinks.
- `data/<package>/` — everything a package writes. `data/freeciv/`: `games/`, `questions/`, `conditional/`,
  `evaluations/`, `results/`, `logs/recordings/`, `fbsim_v3/runs/<run>/` (one folder per FreeCiv v3 run); `data/micropolis/` (and `data/micropolis_*`); `data/pandemic/`
- `credentials/.env` — every key and local setting, loaded by `fbsim_core.local.load_env()`; `credentials/freeciv/` — fleet ssh keys
