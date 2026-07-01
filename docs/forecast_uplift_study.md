# Forecast-Uplift Study: Does giving an LLM game-true probabilities improve its outcomes?

**Branch:** `forecast-uplift-study`  ·  **Status:** in progress  ·  **Model:** `google/gemini-2.5-pro`

## Research question

CivBench originally measured LLM *forecast quality*. This study flips the loop: we test
whether **providing** an LLM with **game-true probabilities** (estimated by Monte-Carlo
simulation of FreeCiv) **improves a downstream outcome** — the classic "does a forecast
improve decision-making?" question that is nearly impossible to run in the real world but
is cheap and clean inside a simulator.

We fix **one outcome metric: gold (treasury)** and test whether giving the LLM ~5–6 of the
most relevant game-true probabilities makes that metric go up, relative to a no-forecast
control, with enough replicates to tell the arms apart.

## Game-true probability (`p_mc`)

For an event `E` resolvable from game state, the game-true probability is
`p_mc(E) = fraction of N Monte-Carlo rollouts in which E occurs`, where a rollout forks a
savegame at turn `T`, reseeds the FreeCiv RNG, and rolls forward under the built-in AI to
the resolution turn. Machinery already exists: `scripts/mc_resolve.py`,
`scripts/monte_carlo_rollout.py`, `src/civrealm/forking/`. N≈20 gives ±0.1 Wilson CIs;
we use larger N for the headline forecasts.

## What the literature says (drives the design)

12 sources read in full (value-of-information; Murphy's quality≠value; weather-forecast
decision studies; Tetlock/EA critiques; advice-taking/algorithm-aversion; LLM lookahead
e.g. RAP, Cicero). Bottom line: forecasts improve outcomes **iff** (1) the forecast is
decision-relevant (Value of Information > 0 — it would change the optimal action),
(2) the agent has levers to act on it, and (3) the agent actually uses it without
pathological over/under-weighting. Design consequences, adopted here:

- **Measure the outcome (gold), never forecast fidelity.** Accuracy ≠ value.
- **Engineer positive VOI on purpose.** Pick decisions whose gold-optimal action genuinely
  varies with the forecasted events; verify with a manipulation check.
- **Three arms, not two:** `control` (no forecast), `true` (real `p_mc`), and
  `scrambled` (shuffled/mismatched `p_mc`). This separates "acts on the *true* number"
  from "acts on *any* number / just the framing."
- **Log uptake:** capture the agent's reasoning and whether it references the forecasts,
  so a null can be diagnosed as "forecast didn't help" vs "agent ignored forecast."
- **Prefer near-horizon, high-leverage decisions** (larger measurable value).

## Two tracks (built in parallel)

### Track A — Agentic play (literal "the LLM plays")
Port the CivRealm LLM baseline agent (Mastaba/BaseLang from `bigai-ai/civrealm-llm-baseline`)
to run on Gemini via our `litellm` wrapper (`src/civrealm/evaluation/models.py`), stubbing
the stale Pinecone/langchain memory. The agent controls player 0 for a **short game (~50
turns)** with the stated objective of maximizing gold. Two arms, same model, independent
instances:
- **A (control):** plays with the normal state prompt.
- **B (forecasts):** the prompt additionally contains a compact table of ~5–6 game-true
  probabilities over multiple horizons (e.g. P(attacked), P(at war), P(civil disorder),
  P(discover Currency/Trade/Banking), P(lose a city)).

Outcome: **gold at turn 50**. Compare mean gold, paired by starting seed. Caveats
(known up front): the baseline agents play weakly, per-turn LLM-call counts are high, and
variance is large — so this track is exploratory and may need many/parallel games.

### Track B — Decision-value (recommended; cheap + powerful)
The LLM makes **one high-leverage strategic decision** at a checkpoint that strongly drives
gold, chosen from a small menu that maps to engine-supported interventions
(`SavegameModifier`: government / tech / gold). We apply the choice and MC-simulate the rest
under the AI to score the resulting gold at the horizon. Three arms (control / true /
scrambled) as above. Because outcomes are simulated and scenarios are shared across arms
(paired), a few hundred cheap decisions give real statistical power.

- **Unit of analysis:** a scenario = (seed, checkpoint turn T).
- **Decision:** pick a policy `a ∈ A` (e.g. government ∈ {Despotism, Monarchy, Republic,
  Democracy}, or an econ-vs-military branch) expected to maximize gold at T+H.
- **Forecasts (treatment):** ~5–6 `p_mc` of mediating/exogenous events (war, disorder,
  attack, tech, city loss) — informative but **not** the answer itself.
- **Outcome:** gold at T+H under the chosen policy, from MC rollout(s).
- **Manipulation check:** verify the gold-optimal policy actually varies with the forecasts
  across scenarios (else VOI≈0 → structural null).

## Statistical plan
- Primary test: paired comparison of gold across arms within scenario/seed
  (paired t / Wilcoxon; bootstrap CIs on the mean gold difference).
- Report effect size (mean gold uplift and standardized), CI, and n. Pre-register the target
  MDE; pick n so arms are distinguishable given observed variance (variance measured in a
  pilot slice first).
- Secondary: dose-response (does chosen-policy quality track forecast values?), uptake rate.

## Deliverables
- `scripts/uplift/` harness (shared `common.py`, per-track runners, analysis + plots).
- Results under `data/evaluations/` / `tmp/uplift/`; writeup with plots.
