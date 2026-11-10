# run_continuations.js

Branching variant of `run_sim.js`. It runs one **trunk** simulation for `S`
turns, snapshots the entire engine state (a byte copy of WASM linear memory),
and then runs many **continuations**: each restores that snapshot, reseeds the
engine RNG with a different seed, and runs `H` more turns. Everything is
streamed to **stdout as JSONL**; nothing is written to disk. Timing and progress
go to stderr.

Intended use: estimate P(event within H turns | state at turn S) by piping the
stream into a reader that applies the same event detectors used on
`run_sim.js`'s files.

```
pnpm tsx cli/run_continuations.js --city haight -S 1440 -H 480 --branch-seeds 1-100 \
    | python3 reader.py
```

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--city`, `-c` | `haight` | Builtin city, same choices as `run_sim.js`. |
| `--seed` | `42` | **Trunk** seed. Applied before `loadCity()` exactly as in `run_sim.js`, so the trunk is identical to `run_sim.js --seed N` for the same city. |
| `--branch-at`, `-S` | required | Turn at which to snapshot and branch. 1 turn = 16 ticks = 1 `cityTime` increment. |
| `--horizon`, `-H` | required | Turns each continuation runs past the branch point. |
| `--branch-seeds` | required | Inclusive range `lo-hi` (or a single `n`) of continuation seeds, run in ascending order. Shard across processes by giving each a disjoint range. |
| `--no-trunk-output` | off | Skip streaming the trunk run (turns 0..S). The trunk is still simulated. |
| `--no-disasters` | off | Disable random disasters, as in `run_sim.js`. |
| `--verbose-log` | off | Include low-signal events (`simulateRobots`, `simulateChurch`, `updateMap`, `updateHistory`, `updateDate`), as in `run_sim.js`. |
| `--intervene` | none | Intervention applied in every continuation at the branch point, after the reseed, retried with another seed when it misses (repeatable): `makeFire`, `makeFlood`, `makeTornado`, `makeEarthquake`, `makeMonster`, `makeMeltdown`, `makeFireBombs`, `demolishCoalPower`, `demolishNuclearPower`, `demolishPowerPlants`, `demolishAirport`. See [Interventions](#interventions). |
| `--progress` | off | One stderr line per finished continuation. |

There is no `--log-every-tick`; exactly one stats row is emitted per turn.

## Output stream

One JSON object per line. Every line has a `kind` field, and every line except
`header` has a `seed` field: `null` for the trunk, the branch seed for a
continuation. Lines appear in this order:

```
header
begin  (trunk)          ┐ omitted with --no-trunk-output
stats/event ... end     ┘
begin  (seed lo)
stats/event ... end
begin  (seed lo+1)
...
```

Within a run, `stats` and `event` lines are interleaved in simulation order.
Runs never interleave with each other. A reader therefore needs only one
run's worth of state at a time: reset on `begin`, finalize on `end`.

### `header` (once)

```json
{"kind":"header","city":"haight","trunkSeed":42,"branchAt":1440,"horizon":480,
 "branchTick":23040,"endTick":30720,"seedLo":1,"seedHi":100,
 "disasters":true,"trunkOutput":true,"interventions":[],
 "interventionMode":"per-continuation"}
```

`interventions` lists `{"turn": S-1, "action": ...}` for each `--intervene`.

### `begin` / `end` (per run)

```json
{"kind":"begin","seed":7,"run":"continuation","fromTick":23040,"toTick":30720}
{"kind":"end","seed":7,"run":"continuation","rows":480,"cityTime":6759}
```

`run` is `"trunk"` or `"continuation"`. `rows` is the number of `stats` lines
emitted in that run. `end.cityTime` is the engine clock at the end of the run.

### `stats` (one per turn)

The same fields as a row in `run_sim.js`'s `log-seed<N>.jsonl`, plus
`kind` and `seed`. See `snapshot()` in `run_sim.js` for the field list;
`census` is the nested tile census object.

```json
{"kind":"stats","seed":7,"tick":23056,"cityName":"haight","cityTime":6280,
 "cityYear":2030,"cityMonth":10,"cityClass":3,"cityScore":...,"census":{...}}
```

### `event` (one per engine callback)

The same fields as a row in `run_sim.js`'s `events-seed<N>.jsonl`, plus
`kind` and `seed`. Base fields are `tick`, `event`, `cityTime`, `cityYear`,
`cityMonth`; the rest depend on `event` exactly as in `run_sim.js`
(e.g. `sendMessage` carries `messageNum`, `messageText`, `x`, `y`,
`pictureFlag`, `important`).

```json
{"kind":"event","seed":7,"tick":23138,"event":"sendMessage","cityTime":6286,
 "cityYear":2030,"cityMonth":11,"messageNum":32,"x":7,"y":90,
 "pictureFlag":false,"important":false,"messageText":"Explosion detected!"}
```

## Tick and time numbering

`tick` counts `simTick()` calls from the start of the trunk and **continues**
into each continuation: a continuation's ticks run from `branchTick + 1` to
`endTick`, and its `cityTime`/`cityYear`/`cityMonth` continue from where the
trunk left off. All continuations start from the identical state, so their
first stats rows share the same `cityTime`.

## Interventions

Each `--intervene` action runs in every continuation, right after the snapshot
is restored and the RNG reseeded, so each continuation draws its own
realisation (disaster location and strength, then how it spreads). The header
says so with `"interventionMode": "per-continuation"`. The events it causes and
then an `intervention` event are continuation lines tagged tick
`branchTick - 1`, turn `S-1`, after the `begin` line, so a `(S-1, H]`
resolution window counts only what follows the intervention.

```json
{"kind":"event","seed":7,"tick":23039,"event":"intervention","cityTime":6279,
 "cityYear":2030,"cityMonth":9,"turn":1439,"action":"demolishAirport",
 "result":{"demolished":1}}
```

Demolitions carry `result.demolished`. A disaster takes effect only if the
engine sends its message during the call ("Flooding reported!" and so on).
When an intervention has no effect with branch seed `s`, the continuation is
restored and retried with RNG seed `s + k*1000000` (k = 1, 2, ...); the lines
of a missed attempt are dropped. The `begin` line records the seed that took
effect and the ones that missed:

```json
{"kind":"begin","seed":7,"run":"continuation","fromTick":23040,"toTick":30720,
 "interventionSeed":1000007,"missedSeeds":[7]}
```

After 100 misses for one branch seed (no river edge to flood, no nuclear plant
to melt down, a monster or tornado already active, nothing to demolish) the run
stops with an error on stderr and a nonzero exit. With `--intervene`, branch
seeds must be below 1000000.

## Guarantees

- Restoring the snapshot is exact: a continuation's output depends only on
  (city, trunk seed, S, H, branch seed), not on which seeds ran before it in
  the same process.
- Stats and event rows are field-for-field identical to `run_sim.js`'s, so a
  parser written for those files works here after reading `kind` and `seed`.
- stdout is written synchronously, so a slow reader applies backpressure
  rather than causing the producer to buffer in memory. Drain or discard
  stderr if you capture it.

## Reader sketch

`cli/count_stream.py` is a minimal consumer that parses every line and tallies
lines and bytes per `kind`; use it as a template. The core loop is:

```python
for raw in sys.stdin.buffer:
    rec = json.loads(raw)
    if rec["kind"] == "begin":   state = new_state(rec["seed"])
    elif rec["kind"] == "stats": on_stats(state, rec)
    elif rec["kind"] == "event": on_event(state, rec)
    elif rec["kind"] == "end":   results.append(finish(state))
```

Filter on `rec["seed"] is None` to skip or separately handle the trunk.

## Performance (haight, S=1440, H=480, one core, `-O3` engine)

About 110 ms per continuation (100 seeds in 11 s; 1000 seeds in under 2
minutes), roughly 60% engine and 40% snapshot/JSON serialization. Piping into
a Python reader costs nothing measurable; Python parses this stream several
times faster than one producer emits it. To scale, run one producer per core
with disjoint `--branch-seeds` ranges.
