# Monte Carlo rollouts from a Freeciv savegame

**Question asked:** Can we replace 0/1 binary resolutions in civbench with real-valued probabilities by running many rollouts from a single saved game?

**Answer:** Yes. Below is a demonstration with a "mundane" forecasting question and a real-valued resolution.

## TL;DR

20 rollouts from `s122/savegames/s122_T100.sav.xz` (turn 100), each advancing 8 turns with a different Freeciv RNG seed, all with the AI playing every player (NoOp on player 0):

| Player 1 (Chen Duxiu) score at turn 108 | value |
|-----------------------------------------|-------|
| min / max                                | 148 / 161 |
| mean ± std                               | 154.65 ± 3.34 |
| **P(score ≥ 155)**                       | **0.55 (Wilson 95% CI [0.34, 0.74])** |

So for the binary question "will Chen Duxiu's score be ≥ 155 at turn 108?" the ground-truth resolution is **0.55**, not 0 or 100. That's the RLVR-friendly signal you wanted.

## What it took: civbench-side vs Freeciv-side

I deliberately tagged every change as one or the other.

### **CIVBENCH** (Python, this repo)
- [src/civrealm/forking/savegame_modifier.py](src/civrealm/forking/savegame_modifier.py) — added `SavegameModifier.set_rng_from_seed(seed)`. Reproduces Freeciv's `fc_srand(seed)` (Mitchell-Moore additive RNG, including the 10 000-iteration warm-up loop) in Python, then writes the resulting 56 32-bit words + indices into the savegame's `[random]` block in the same `%8x`-padded format Freeciv uses.
- [src/civrealm/forking/fork_manager.py](src/civrealm/forking/fork_manager.py) — added `rng_seed` as a new modification type alongside `gold`, `government`, `tech`.
- [scripts/monte_carlo_rollout.py](scripts/monte_carlo_rollout.py) — driver that runs N rollouts in isolated subprocesses (civrealm's global state forbids in-process repeats), persisting partial results between rollouts.
- [scripts/analyze_mc_rollouts.py](scripts/analyze_mc_rollouts.py) — summarises distributions and computes Wilson CIs.

### **FREECIV** (C, in the docker container)
- [patches/freeciv_reseed.patch](patches/freeciv_reseed.patch) — a ~12-line addition to `sg_load_random` in `server/savegame/savegame3.c` that, after `fc_rand_set_state(loading->rstate)`, checks `/tmp/freeciv_reseed`. If present, treats its contents as a `unsigned long`, calls `fc_srand(seed)`, updates `loading->rstate`, and deletes the file.
- [patches/apply_to_container.sh](patches/apply_to_container.sh) — automates the patch + rebuild inside the `freeciv-web` container (see "Honest limitations" below).

## Methods compared

### Method B — savegame RNG edit (recommended for civbench, **fully demonstrated**)

For each rollout:
1. Read `s122_T100_*.sav.xz`.
2. Compute the Mitchell-Moore RNG state Freeciv would have after `fc_srand(seed)` (Python port of `utility/rand.c`, including the 10 000-step heat-up).
3. Write that state into the savegame's `[random]` block.
4. Hand off to the existing `ForkManager.run_fork()`. No Freeciv changes required.

**No Freeciv patch is needed.** Everything lives in this repo.

### Method A — server-side `fc_srand` on load (**patch written, full deploy blocked**)

For each rollout:
1. `docker exec freeciv-web bash -c "echo <seed> > /tmp/freeciv_reseed"`.
2. Load the *unmodified* savegame.
3. Inside the patched `sg_load_random`, the hook reseeds via `fc_srand(seed)` and consumes the file.

Method A's patch is committed in `patches/`. I authored it, applied it to `/docker/freeciv/freeciv/server/savegame/savegame3.c`, rebuilt `savegame3.lo` and confirmed the `CIVBENCH: reseeding RNG` log string was present in the resulting object file and in a freshly linked `freeciv-server` binary.

## Equivalence: why A and B yield identical results (and why this is not an empirical question)

Both methods cause the global `rand_state` to enter the same state immediately before the first in-game RNG call:

- Method B writes `S = fc_srand_state(seed)` into the savegame; on load, `fc_rand_set_state(loading->rstate)` sets the global rand_state to `S`.
- Method A leaves the savegame alone; on load, `fc_rand_set_state(loading->rstate)` restores the original state, then the patched code calls `fc_srand(seed)`, which writes `S` into the global rand_state.

Freeciv's RNG is fully deterministic (`utility/rand.c`), and civbench's `phasemode player` setting eliminates the network-ordering source of nondeterminism that civbench itself measures and tests for. So **given identical input save + identical rand_state, Freeciv produces bit-identical turn sequences**. The two methods aren't statistically close — they're algebraically identical.

This was the point of porting `fc_srand` exactly into Python, including the 10 000-iteration warm-up. If I had only written "some random bytes seeded by Python", the two methods would diverge.

## Honest limitations

1. **Method A was not run end-to-end.** The freeciv-web docker image's pre-built `libfreeciv-srv.a` was compiled with `-DFREECIV_WEB` in many places (`civserver.c`, `auth.c`, `cityhand.c`, `citytools.c`, `handchat.c`, `plrhand.c`, `ruleset.c`, `settings.c`, `unithand.c`). Rebuilding `savegame3.o` (which has no `FREECIV_WEB` guards) and relinking worked, but `civserver.o` had to be rebuilt with `-DFREECIV_WEB`, and several other objects ended up subtly mismatched, causing the relinked binary to time out before sending `begin_turn` packets. Fixing this properly requires running freeciv-web's own build pipeline (`/docker/freeciv-web/build.sh`) with proper configure flags, which has its own infrastructure dependencies. The patch and instructions are preserved in [patches/](patches/) so a clean Docker rebuild can apply them. I reverted the binary so civbench works normally.
2. **Sample variance is real at N=10.** Batch 1 (seeds 1–10) gave P=0.40; batch 2 (seeds 100–109) gave P=0.70. Pool both for the 0.55 estimate. For an RLVR signal you'd want enough rollouts to get the CI width below your usable resolution — probably 50–200 rollouts per question to get ±0.05 precision on a P near 0.5.
3. **AI is the source of stochasticity.** Player 0 (`S122`) is `/aitoggle`d on; all six players are AI-controlled and the RNG drives their decisions. If you want player-action-dependent rollouts, the agent that drives player 0 has to be the same across rollouts but the OTHER players' RNG-dependent behavior carries the variance.

## How to reproduce

```bash
# Method B, 10 rollouts from turn 100, 8 turns forward, seeds 1..10
uv run python scripts/monte_carlo_rollout.py \
    --base-seed 122 --checkpoint-turn 100 --run-turns 8 \
    --rng-seeds 1 2 3 4 5 6 7 8 9 10 --method save_edit \
    --tag mcB --output tmp/mc/method_b_n10.json

# Analyse
uv run python scripts/analyze_mc_rollouts.py \
    tmp/mc/method_b_n10.json tmp/mc/method_b_n10_batch2.json \
    --player 1 --threshold 155
```

For Method A: apply `patches/apply_to_container.sh` (will need build-system fixes — see "Honest limitations" above), then run `--method server_reseed` instead.

## Artifacts

- `tmp/mc/method_b_n10.json` — first batch (seeds 1–10)
- `tmp/mc/method_b_n10_batch2.json` — second batch (seeds 100–109)
- `tmp/mc/smoke.json`, `smoke_revert.json` — smoke-test outputs
- `patches/freeciv_reseed.patch` — the Freeciv-side change
- `patches/apply_to_container.sh` — automation
