Always use `uv run` to execute Python scripts in this project, not bare `python` or `python3`.

## Data directory

See `data/README.md` for full documentation. Key paths:
- `data/games/` — serialized game snapshots
- `data/questions/` — unconditional forecasting questions
- `data/conditional/` — conditional experiment question sets (republic, gold500, null_conditional)
- `data/evaluations/` — LLM eval outputs split into `results/`, `plots/`, `runs/`
