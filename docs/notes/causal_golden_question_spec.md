# Causal golden question — pilot spec

Status: derisked, awaiting approval to launch fleets. 2026-07-12.

## What exactly we are testing

The real-world golden question — *"was the model's conditional forecast causally
correct?"* — is unanswerable outside a simulator because (1) only one branch of history
resolves, and (2) conditioning ≠ intervening (worlds where X happens naturally are a
selected subpopulation). In FreeCiv we can estimate all three ground truths per
(snapshot, target-question Y) pair and score a model's *update*, not its absolute Brier.

**Setup.** Snapshots: seeds 0–3 at turn 60 (existing MC worlds). Targets Y: each game's
55 existing binary questions resolving at turn 90 (H1). Conditioning event X: "player 0
knows tech T at turn 70," with T chosen per seed (phase-2 data-driven) so that X occurs
naturally in 20–80% of baseline rollouts. The intervention imposes X by granting T in
the turn-60 savegame.

**Ground truths (per seed, N = 40 rollouts per arm):**

| Quantity | How |
|---|---|
| P(Y) | baseline rollouts |
| P(Y \| do(X)) | rollouts of the T-granted fork |
| P(Y \| X observed) | baseline rollouts filtered to those where p0 discovered T naturally by turn 70 |

Derived: true causal effect **Δ_do = P(Y|do X) − P(Y)**; true evidential effect
**Δ_obs = P(Y|X obs) − P(Y)**; **confounding gap = Δ_obs − Δ_do** — a quantity
literally unmeasurable in the real world, and the pilot's unique selling point.

Known approximation: do(X) grants T at turn 60 while natural discovery happens anywhere
in 60–70; the interventional and evidential events are aligned on "knows T at 70," not
on discovery timing. Reported as a caveat.

**LLM elicitation (phase 4).** Per model, three framings per Y: unconditional p̂(Y);
interventional ("it is guaranteed that p0 will possess T by turn 70…"); evidential
("we later learn that p0 discovered T by turn 70…"). Scored on:

1. **Δ-accuracy**: correlation + MAE of elicited Δ̂ = p̂(Y|X) − p̂(Y) against true Δ_do,
   across the 220 (seed, Y) pairs. Paired scoring cancels both baseline miscalibration
   and the documented conditional-framing penalty (+36% Brier for Opus 4.5) that
   contaminates the existing conditional evals.
2. **Intervening-vs-conditioning distinction**: is p̂(Y|do-framing) ≠ p̂(Y|observed-framing),
   and does the difference correlate with the true confounding gap? (Expected headline:
   models give identical answers; ground truth doesn't.)
3. **Placebo**: null intervention — a tech p0 already knows, so Δ* = 0 exactly — with the
   same framings, isolating pure framing shift.

**Hypotheses.** H1: elicited Δ̂ correlates weakly with Δ_do. H2: models do not
distinguish do() from observation. H3: placebo framing shifts p̂ despite Δ* = 0.

## Precision

N=40 per arm → SE(p) ≤ 0.079, SE(Δ_do) ≤ 0.11 per question; 220 (seed,Y) pairs power
the correlation analyses. Weakest link: P(Y|X obs) uses only the f·40 conditioning
rollouts (SE up to ~0.16 at f=0.5) — acceptable for a pilot, bump N later if the
confounding-gap estimate matters.

## Decision gates

- **Phase-2 gate (after baseline fleets):** need, per seed, a tech T with natural
  frequency-by-70 in [0.2, 0.8]. Fallback X families if none: government change by 70
  (natural conditioning only), or gold-threshold events with `gold_add` intervention.
- **Phase-4 gate (after do(X) fleets):** proceed to LLM elicitation only if ≥10
  questions have |Δ_do| > 0.15 — otherwise the intervention is too weak for Δ-scoring
  to beat noise, and we pick a stronger T (or gold_add:500) and rerun phase 3 only.

## Derisk results (done)

- **Found + fixed a silent no-op**: `grant_player_tech` regexed for the old per-player
  `research="inventions"` field; current saves store techs in the `[research]` table.
  Patched (bit-flip + known-count increment, old format kept as fallback, raises on no
  match instead of silently succeeding). Offline-verified and end-to-end verified:
  granting Currency (id 20) to p0 at turn 60 flips 6/55 question outcomes and shifts
  p0's turn-90 gold +84 at identical RNG seed.
- Parallel 2-worker rollouts confirmed safe (file-locked port allocation); ~72–90 s per
  rollout, ~100 KB archived per rollout, fork dirs auto-deleted.
- Old `forkmcr60` fork dirs (regenerable; configs + source saves + deterministic reseed
  all retained) deleted → 16 GB reclaimed, 41 GB free.

## Cost to run

| Phase | Compute | Wall-clock (2 workers) | $ |
|---|---|---|---|
| 1. Baseline fleets: 4 seeds × 40 | ~4 h CPU | ~2 h | 0 |
| 2. Analysis + X choice | — | minutes | 0 |
| 3. do(X) fleets: 4 seeds × 40 (+1 placebo arm on one seed) | ~4.5 h CPU | ~2.5 h | 0 |
| 4. Elicitation: 220 Y × 3 framings × ~3 models ≈ 2k calls | API | ~1 h | ~$5–15 (Gemini Flash/Pro via existing GCP; frontier via Anthropic/OpenAI keys if desired) |

## Runner

`scripts/causal_pilot.py` (committed): archives full per-rollout `game_data` (gzipped)
so any conditioning event is definable post hoc; resume-safe; deletes fork dirs.
Analysis notebook/script to follow as `scripts/uplift_v2/causal_analyze.py`.
