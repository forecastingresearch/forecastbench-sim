# Pandemic runner timing

`pandemic_world.runner.run_region` labels simulation timesteps as days 0–90.
It retains the existing Starsim annual time grid (`dt=1`), disease-duration
semantics, mortality, and output schema. These labels do **not** denote physical
calendar days. The simulator start and vaccination calendar year now share an
explicit epoch: a campaign at timestep 25 is scheduled at `SIM_START + 25`.

For vaccination runs, `vax_day` must be a nonnegative integer. Day 0 and the final
step are valid; a day beyond the horizon (including 999) has no effect during the
run. This guard matters because Starsim 3.3.4 chooses the nearest timestep: bare
`years=[25]` maps to step 0, but adding the epoch to an out-of-range day would
instead map it to the final step. `vaccinate=False` continues to ignore vaccine
parameters. Zero coverage leaves control trajectories unchanged.

## Regression checks

With the pandemic package and its `dev` extra installed, run from the repository
root using the relevant environment:

```sh
uv run --offline --no-project --python <venv>/bin/python -m pytest -q worlds/pandemic/tests
```

The tests call the real runner and Starsim initialization. They check all timing
examples from issue #6, shifted epochs, invalid days, endpoint administration,
nonempty uptake, per-agent vaccination times and susceptibility, zero coverage,
beyond-horizon behavior, equal pre-intervention histories and a nonzero infection
trajectory that diverges at the campaign step. Initialization uses 100 agents;
trajectory checks use 500 agents over 13 steps. They need no model calls or
production data. Locally verified with Starsim 3.3.4 and Python 3.13; the existing
offline benchmark CI does not install Starsim or run this world test suite.

## Historical results and runner parity

This fix changes future simulations; it does not repair or regenerate existing
scenario metadata, cached trajectories, scores, or prompts. Artifacts produced
by the old public runner require a separate provenance audit before their
recorded campaign day can be trusted.

Read-only comparison with the historical paper runner found that it already
used `SIM_START + vax_day` and supported a mortality override. Its in-range timing
agrees with this fix, but that does not establish equivalence of scientific
outputs. The corrected continuous helper also schedules calendar year 2021 at
step 21; its fixed-state continuation, stream handling and coverage updates serve
a different purpose. Those frozen producers are unchanged. A shared execution
helper would expand scope without establishing equivalent behavior; any future
alignment should first specify mortality, time units and continuation semantics
and validate uptake as well as timing. This PR makes no paper-number validity
claim.
