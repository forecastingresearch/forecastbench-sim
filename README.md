# ForecastBench-Sim

A multi-simulation forecasting benchmark that can host an **arbitrary number of
simulation "worlds."** A world runs a simulation, and models read a world-state report
and forecast future outcomes. The domain-agnostic machinery (question schema, resolver,
generator, scoring, eval harness) lives in `fbsim-core`; each world is a plugin.

Two worlds ship today:

- **FreeCiv** (`worlds/freeciv/`) — strategy-game world on the CivRealm engine.
- **Starsim** (`worlds/pandemic/`) — SIR epidemic world with conditional vaccine forecasting.

Adding a third is the [Adding a world](#adding-a-world) flow below.

```
packages/fbsim-core/      # schema, resolver, registry, generator machinery,
                          # metrics, eval (models/sampling/difficulty)
worlds/freeciv/           # FreeCiv strategy-game world (CivRealm engine + benchmark)
worlds/pandemic/          # Starsim SIR epidemic world (conditional vaccine forecasting)
```

It's a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/). Install with:

```bash
uv sync --all-packages
```

## Adding a world

A world provides these to `fbsim-core` (see `worlds/pandemic/` for a compact example):

1. **Serializer → `game_data`** in the canonical **turn-major** layout
   `time_series[metric][turn][entity_id] = value` (see `fbsim_core.questions.timeseries`).
   Core never infers the layout — the world emits it.
2. **Template registry** — build a `fbsim_core.questions.registry.TemplateRegistry`
   from the world's `QuestionTemplate`s and pass it to `QuestionResolver(registry)`.
   (`worlds/pandemic/pandemic_world/templates.py`)
3. **Report renderer** — a function producing the model-facing world-state text
   for a snapshot. (`pandemic_world/report.py`)
4. **Eval driver / adapter** — turn questions + report into prompts, query models
   (`fbsim_core.evaluation.models`), score with `fbsim_core.metrics`.
   (`pandemic_world/scale_eval.py`)
5. **(Optional) conditional executor** — how an intervention is applied. FreeCiv forks
   a savegame; pandemic re-runs the sim with a vaccine. Core keeps the generic
   control-vs-intervention structure; the executor is world-specific.

## Verify

```bash
# core (domain-agnostic) tests
cd packages/fbsim-core && uv run --project .. python -m pytest tests

# FreeCiv world: generate + resolve questions from a recorded game
uv run python worlds/freeciv/scripts/generate_questions.py \
    data/games/seed1619_data.json --snapshot-turn 60 --summary

# Pandemic world: scaled conditional/unconditional benchmark + chart (cached)
uv run python -m pandemic_world.scale_eval --eval
```

Both worlds resolve through the *same* core resolver and registry — that's the point.

## Offline cached-forecast interfaces

The additive [fbsim-benchmark package](packages/fbsim-benchmark/README.md) provides installed-package examples and synthetic fixture tests for FreeCiv, Micropolis and Starsim without running engines or calling models. It depends on the existing core; world and human-subject workflows remain in place. See its [workflow map](packages/fbsim-benchmark/docs/WORKFLOWS.md) for retained, migrated and legacy routes.

### Micropolis world

[Micropolis setup and workflows](worlds/micropolis/README.md) cover the external engine, question generation, cached/model gathering and raw scoring. Synthetic offline tests do not validate a live engine or reproduce paper results.
