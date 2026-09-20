# Elicitation and scoring for the fbsim v3 sets (2026-09-07)

Status 2026-09-09: FULL RUN COMPLETE for all 24 models (48,543 of 48,552 rows; DeepSeek V3 is short 9 follow-up rows whose
turn-1 answer was empty). Rows: `../fbsim_v3_corpus/elicit_v1/<slug>/results.jsonl` (last row per (item, arm) wins; `_discarded/`
holds DeepSeek V4 Flash answers from hosts that dropped the question — never score them). Scores with bootstrap CIs:
`../fbsim_v3_corpus/scores_v1/` (score_v1.py). Paper-format tables: `../fbsim_v3_corpus/results_v1/` (results_table.py:
freeciv_results_table.md, _wide.csv, _long.csv, and Fabio-schema _binary.csv / _continuous.csv). OpenRouter spend: full run
≈ $361, everything since 2026-09-07 (smoke, sample, stage, run) $432. Wall clock: 23 models in 41 min at the per-model worker
counts; DeepSeek V3 85 min because DeepInfra rate-limited it to ~8 calls/min. Settings that produced these rows: report-first
prompt, lowest effort per Fabio's models.csv (minimal for 5 OpenAI/Gemini models = 0 reasoning tokens; Anthropic low ≈ 100),
1,024-token budgets for the four budget models, first-party pins, DeepSeek V4 Flash first-party at effort low (≈15k reasoning
tokens, 1.3% runaways), Kimi K2 on Novita (1.3% runaways). Preliminary finding: every model is worse than a flat 0.5 on
binary excess Brier (forecasts compressed to 0.33–0.48 as truth runs 0.12–0.88, biased low); discrimination ρ(p,q) tracks
ECI at 0.71, continuous nCRPS at 0.61, excess Brier at 0.12. Open: a medium-effort re-run of the bank on ~6 models to test
whether the calibration gap is an effort artefact.

## Files
| file | what |
|---|---|
| `elicit_v1.py` | the harness: builds every prompt, calls OpenRouter, writes `results.jsonl` (one row per model × item × arm) |
| `models_v1.csv` | the 24 models from `~/Downloads/model_scores.csv` mapped to OpenRouter ids: reasoning mode (effort / budget / none), effort or budget, optional max_tokens and provider pin, list prices |
| `smoke_run.sh`, `smoke_report.py` | one process per model into `OUT/<slug>/`; health table (errors, parse modes, providers, tokens, cost, full-run projection) |
| `cost_estimate.py` | exact input-token counts (tiktoken o200k) × prices × an output-token assumption → `cost_estimate.csv` |
| `score_v1.py` | the scorer: `score_items.csv`, `score_summary.csv`, `SCORES.md` |
| `score_selftest.py` | oracle / flat / noisy / broken synthetic forecasters through the scorer; 20 checks |

## Calls per model
| arm | calls | what |
|---|---|---|
| t1 | 1,524 | 750 bank + 300 tails + 50 mirrors + 300 continuous + 124 natcond extra questions; one fresh prompt each |
| t2 | 400 | every natcond cell: the t1 conversation of its question continued with the uniform reveal sentence |
| nonews | 99 | 100 cells (stratified by block, seed 2026; v1.8) → 99 distinct questions; same t1 conversation continued with the no-news sentence |
| single | 0 | dropped in v1.8 (owner decision 2026-09-08); flags kept in natcond_600_v1.7_controls300.json |
| total | 2,023 | |

v1.8 prompt: the derisk wording with the report moved ahead of the question: [intro + "Question Background" + report] is one
content part with a `cache_control` breakpoint, [Question Title + criteria + instructions] the second. Cache test
(`cache_test.py`, 3 consecutive calls per model on one world): Anthropic reads 10,618 of ~10,900 tokens from cache on calls 2–3
(Fable $0.149 → $0.031 per call), OpenAI caches 6,912–7,764 of ~7,970, Google AI Studio all 8,445; Alibaba, DeepInfra and
Novita have no cache (cheap hosts). Jobs are ordered by world so a prefix stays warm. A turn-1 prompt is 7,565–8,544 tokens.

Reasoning (Fabio's models.csv of 2026-09-08 = lowest supported level per model): `effort` = minimal for GPT-5, GPT-5 Mini,
GPT-5 Nano, Gemini 3 Flash Preview, Gemini 3.1 Flash Lite; low for o3, o4 Mini, GPT-5.4 Nano, GPT-5.5, GPT-5.6 Luna/Sol,
Gemini 3.7 Flash, Sonnet 5, Opus 5, Fable, DeepSeek V4 Flash (Fabio lists "none" for the GPT-5.4+ models but chose low);
budget 1,024 tokens for Haiku 4.5, Gemini 2.5 Flash, Qwen3 235B A22B, Qwen3.5-Flash; nothing for Llama 4 Scout, DeepSeek V3,
GPT-4.1, Kimi K2. This matches Fabio's Micropolis run; Nick's StarSim table is at provider defaults pending his re-runs.
Every request sends `max_tokens` 16384 (bounds OpenRouter's in-flight credit reservation, which otherwise refuses calls
with HTTP 402 when many are in flight) and `provider: {ignore: [Novita]}`; a `provider` column pins an order with
fallbacks off. Every row logs provider, prompt/completion/reasoning tokens, cost, finish reason, the reasoning param
sent, and the reasoning text where returned. Sampling is otherwise provider default. Parse: strict = the required tag
block; a lenient fallback (the labelled p5..p95 run, or "probability … 0.xx" in the tail) is used when the tags are
missing and flagged `parse_mode: lenient`; a row that parses neither way is re-asked once and then kept with
`value: null`. HTTP 402/429 are retried with 15–90 s waits.

## Cost (central / high)
Output assumption: reasoning model at low effort = 1,200 hidden + 500 visible tokens per call; non-reasoning = 700.
"High" = output at twice that. Prices from OpenRouter on 2026-09-07 (`--live` refreshes them).

| model | reasoning | $/call-set | high |
|---|---|---|---|
| Claude Fable 5 | low | 416 | 629 |
| GPT-5.5 | low | 229 | 357 |
| Claude Opus 5 | low | 208 | 314 |
| Claude Sonnet 5 | low | 83 | 126 |
| GPT-5.6 Sol | low | 83 | 126 |
| o3 | low | 75 | 109 |
| GPT-5 | low | 68 | 110 |
| GPT-4.1 | none | 55 | 69 |
| Claude Haiku 4.5 | low | 42 | 63 |
| o4 Mini | low | 41 | 60 |
| Gemini 3.7 Flash | low | 31 | 47 |
| Gemini 3 Flash Preview | low | 23 | 36 |
| Qwen3 235B A22B | low | 17 | 25 |
| Gemini 2.5 Flash | low | 17 | 27 |
| Kimi K2 | none | 16 | 20 |
| GPT-5 Mini | low | 14 | 22 |
| Gemini 3.1 Flash Lite Preview | low | 11 | 18 |
| GPT-5.4 Nano | low | 9 | 15 |
| GPT-5.6 Luna | low | 9 | 14 |
| DeepSeek V3 | none | 8 | 10 |
| DeepSeek V4 Flash 0731 | low | 4 | 5 |
| GPT-5 Nano | low | 3 | 4 |
| Llama 4 Scout | none | 3 | 3 |
| Qwen3.5-Flash | low | 2 | 4 |
| **total, 24 models** | | **1,467** | **2,211** |

Fable, GPT-5.5 and Opus 5 are 58% of the total. Roughly two thirds of the expensive models' cost is input; moving the
report ahead of the question in the prompt would let OpenAI/Gemini/Anthropic prefix caches hit on 8 distinct reports
and cut that by half or more, at the price of no longer using the derisk prompt verbatim.

## Scoring (score_v1.py), one rule per set
- Binary (bank, tails, mirrors, extra): q = `qAll`. Brier = (p−q)² + q(1−q); **excess Brier (p−q)²** is the headline.
- Tails additionally: log loss in bits with p clipped to [0.001, 0.999]; **excess bits = KL(q‖p)** is the headline.
- Continuous: the five reported percentiles (sorted if non-monotone) scored by pinball loss at τ = .05/.25/.5/.75/.95
  against all 1,000 truth values, (2/5)·Σ; floor = same with the truth quantiles; **excess CRPS** = difference.
  Diagnostics: coverage of the 50% and 90% intervals, width, median error.
- Natural conditionals: **excess_t2 = (p₂ − p(Y|X))²** headline; `stay` = (p₁ − p(Y|X))² (no-update baseline);
  `gain` = stay − excess_t2; `excess_nonews` = (p_nonews − p(Y))²; `drift` = p_nonews − p₁; `excess_single` =
  (p_single − p(Y|X))²; movement `p₂ − p₁` vs target `p(Y|X) − p(Y)`: correlation, sign agreement on |Δ| ≥ 0.05, mean |move|.
- Aggregates: per model × set, × horizon, × family (× block and × reveal kind for natcond), bootstrap 95% CIs over items.
- Missing answers: binary imputed at 0.5 (`--impute-binary none` to drop); continuous and natcond rows dropped; parse rate
  reported next to every aggregate. Everything joins on (world, item id), never on text.

## How to run
```bash
uv run python elicit_v1.py --out ../fbsim_v3_corpus/elicit_v1 --dry-run              # job counts only, no calls
uv run python elicit_v1.py --out ../fbsim_v3_corpus/elicit_v1 --models "OpenAI: GPT-5 Nano" --limit 20   # smoke
uv run python elicit_v1.py --out ../fbsim_v3_corpus/elicit_v1                         # all 24 models, all sets
uv run python score_v1.py ../fbsim_v3_corpus/elicit_v1 --out ../fbsim_v3_corpus/scores_v1
```


## Smoke run (2026-09-07/08), `../fbsim_v3_corpus/elicit_smoke/`
Round 1 (unfixed harness, 24 × 6 workers in flight) failed on 339 calls with HTTP 402: OpenRouter reserves
max_tokens × price per in-flight call against the account balance ($50 at the time, ~$30 free), and the default
max_tokens for most models is 64k–128k. Fixes: max_tokens 16384 on every call, 402/429 retried with waits, 3 workers per
model. Round 2 then completed 23 of 24 models with 0 errors and 0 unparsed answers (lenient parse used on 8 rows in
all: GPT-5 Nano and GPT-5 Mini writing `<p5: 14>` lines without the `<percentiles>` block, one Gemini 2.5 Flash, one
Llama probability). Kimi K2 completes on Novita. Health table: `smoke_report.py`.

Findings that need an owner decision
1. DeepSeek V4 Flash 0731: unpinned routing spread 13 calls over 8 hosts (fp4/fp8/unknown); every host ignored the
   reasoning control (8k–13k reasoning tokens at effort low, at budget 1024 and at effort minimal) and took 1–6 min per
   call. The first-party DeepSeek endpoint is removed by the account's OpenRouter guardrail step (29 → 28 endpoints)
   before routing, so a pin to it returns 404. Options: enable it under openrouter.ai/settings/privacy and pin; pin one
   third-party host and accept ~10k reasoning tokens (cheap, ~$5 total, but ~10 h wall time at 16 workers); or drop it.
2. Kimi K2 (0711) is hosted only by Novita on OpenRouter (0905 likewise; only K2 Thinking has Google). Pinned to Novita
   as an interim choice against the team's Novita rule.
3. DeepSeek V3 (deepseek/deepseek-chat) exists only as DeepInfra fp4 or StreamLake (quantization unknown). DeepInfra pinned.
4. Anthropic models at effort low barely think: Sonnet 5 ≈ 100 reasoning + 120 visible tokens per answer, Opus 5 ≈ 140 +
   90, Fable ≈ 140 + 250; OpenAI models at low use 300–1,000 reasoning tokens, Gemini 350–870, the 1,024 budgets are
   honoured exactly by Qwen and within budget by Haiku/Gemini 2.5. "Low" is not the same amount of thinking across vendors.
5. Effort below "low" exists for OpenAI (minimal / none) and Gemini (minimal); Jaeho's stated rule is the lowest level
   per model, the run default is "low". Fabio's table of minimum levels was not on this machine; edit `effort` per model.
Provider pins now in models_v1.csv (validated with one call each): Anthropic → Anthropic (round 2 had gone to
"Claude Platform on AWS"/Bedrock), OpenAI → OpenAI, Google → Google AI Studio then Google, Qwen → Alibaba,
Llama 4 Scout → DeepInfra fp8, DeepSeek V3 → DeepInfra, Kimi → Novita, DeepSeek V4 Flash → DeepSeek (blocked, see 1).

Observed cost: per-call costs from the smoke project the full run at ≈ $1,080 for 24 models (Fable $320, Opus 5 $151,
GPT-5.5 $135; everything else ≤ $60), below the $1,467 list-price estimate because low-effort output is shorter than
assumed. The account held $47 after the smoke; the full run needs a top-up of about $1,100.
Scorer end-to-end on the smoke rows (`elicit_smoke/scores/`): every set aggregates and joins correctly; with ~26 bank
items per model the numbers are not results, but on those items most models score worse than the flat 0.5 forecaster
in excess Brier (0.09–0.21 vs 0.075), o3 and Gemini 2.5 Flash the exceptions — consistent with the derisk finding.


## Paid sample (2026-09-08), `../fbsim_v3_corpus/elicit_v1/` — final settings, rows reused by the full run
17 models × 78 calls (bank 30, tails 10, mirrors 5, continuous 10, natcond 10 cells + their t1 and no-news) and 6 cheap
models × 376 calls (bank 220 = 10 per family, tails 60, mirrors 30, continuous 45, natcond 10). 0 API errors; unparsed:
Llama 2 (`<0.2>` style, now parsed leniently), Kimi 2 (one 16k-token runaway, one Novita content filter).
- Observed cost (`smoke_report.py`, incl. caching): full run ≈ **$491 for 23 models** (Fable $122, GPT-5.5 $81, Opus 5 $51,
  GPT-4.1 $35, o3 $30, everything else ≤ $23); DeepSeek V4 Flash to be added once its host is decided (≈ $5–40).
  About $22 of that is already spent (the sample rows resume).
- Saturation (`saturation.py`): nothing to cut. On the 5 cheap probe models × 225 bank items, only 3% of items are within
  0.01 of the truth for every model; family means range 0.02 (civil war) to 0.30 (any-city-destroyed); tails 60 items
  0.01–2.4 bits with between-model SD 0.15; mirrors 30 items mean 0.28 with model means 0.19–0.47. On 23 models × 37 bank
  items the set means span 0.10–0.20 (SD 0.03). Natcond (10 cells): block C1 gain +0.12, C2 +0.02, A/B ≈ 0, D ≈ 0.
- Scoring change (owner 2026-09-08: match Fabio's normalized CRPS): continuous headline = `ncrps_global` = CRPS / one fixed
  constant per family, Fabio's Micropolis rule (his constants, inferred from his CSV: population 20000, traffic 20,
  crime/pollution/land value 60). Our constants (`draw_v1/continuous_norm_constants.json`) = median truth p05–p95 range per
  family, per metric inside NC1_value_at_T, rounded to one significant figure: wonders 4, techs 9, world techs 20, conquests 10,
  losses 10, founds 20, razings 20, world captures 50, cities 30, population 70, score 70, territory 300. Also reported:
  `excess_ncrps_global` (floor subtracted, same constant) and `excess_crps_norm` (per-item IQR); raw CRPS per item.
- DeepSeek V4 Flash (owner 2026-09-08: same as Nick): OpenRouter default routing (Novita still ignored), effort low, no pin.
  Expect 8k–13k reasoning tokens and 1–6 min per call; ~$10 and ~12 h wall time for its 2,023 calls at 8 workers.
- Preflight: `PREFLIGHT_prompts.md` = the exact request body per reasoning mode and the full message texts for every arm.


## Concurrency and priming (2026-09-09)
- Per-model `workers` column in models_v1.csv, set from each model's observed per-call time and remaining calls for a
  ~30-minute finish: 16 for OpenAI / Anthropic / Google, 14–21 for DeepInfra / Alibaba / Novita, 32 for Qwen3 235B,
  128 for DeepSeek V4 Flash (median 123–200 s per call across its third-party hosts). `--workers N` on the command line overrides.
- Cache priming: the first job of every (world, kind) prefix group is submitted first, then everything else in world order,
  in ONE pool submission with no barrier. The earlier group-by-group version (prime, batch, wait) held mean concurrency to
  ~4 whatever the worker count — measured in the one-per-cell stage (Sonnet 16 workers: mean 3.9 in flight) — and a
  phase barrier version stalled on a single slow call (Novita tail 316 s, DeepSeek hosts 572 s). Cost of the no-barrier
  version: at most ~W calls at the very start may miss a cache still being written.
- Measured at high concurrency, no rate-limit retries and no errors: Kimi K2 on Novita at 32 workers — 21 s median,
  26 s p90, tail to 316 s, 28 in flight; DeepSeek V4 Flash at 128 workers — 123 s median, 74 in flight while ramping,
  20.8 rows/min in the first 4 minutes, six hosts. DeepInfra sent two 429s at 8 workers (retried).
- Every row now records `secs` (wall time of the call incl. retries), `workers`, and `http_retries` (transient HTTP codes seen).
- Parser (lenient tier, flagged `parse_mode: lenient`): `<probability>` tag → "probability … 0.xx" in the tail → a bare
  `<0.xx>` at the end; `<percentiles>` block → the last labelled run of `p5: n` / `p5=n`. `reparse.py DIR` re-parses stored
  unparsed rows offline. Unparsed so far: Kimi 2 (16k-token runaway; Novita content filter), GPT-5 Nano 2 (stopped before
  the format block, now parsed from its `p25=2, p50=4 …` line), Llama 2 (`<0.2>`, now parsed).
- One-per-cell stage (2026-09-09, 308 calls per model: one item per family×horizon cell + one natcond cell per
  block×horizon; `state/one_per_cell.json`, `--cells`): run on 18 models + Kimi (32 workers) + DeepSeek V4 Flash (128);
  Fable, GPT-5.5, Opus 5, o3, GPT-4.1 deferred until the credit top-up (balance was $24). Rows live in elicit_v1/ and
  the full run resumes over them. Launch the full run with `full_run.sh` (all 24) or `full_run.sh "id1,id2"`.

- Content shape (2026-09-09): two content parts with a cache breakpoint only on first-party hosts (Anthropic, OpenAI,
  Google, all verified to keep both parts); one plain string elsewhere. Found because the OpenInference host serving
  DeepSeek V4 Flash dropped the second part: 134 answers to a question the model never saw ("please provide the missing
  question"); all discarded (`_discarded/`). Check used on every model/host: share of answers naming the question's subject
  civ (80–90% on every sound host; short Sonnet/Opus/o3 answers verified by reading) and a regex for "missing question".
- DeepSeek V4 Flash (2026-09-09): even with plain strings, DigitalOcean also loses the question on 8k prompts (6 of 25) and
  several hosts run to the 16k cap with no answer; per-call 90–540 s. The first-party endpoint is reachable again and is
  now pinned. The model ignores `effort` and `reasoning.max_tokens` (10–15k reasoning tokens, sometimes a runaway to the
  cap; `max_tokens` raised to 24000 so 12k-token thoughts finish) but honours `reasoning: {enabled: false}` (7 s,
  ~600 tokens, parses). `reasoning_mode: disabled` is implemented; the run default stays effort low per Fabio's table
  pending the owner's decision.
- Kimi K2: 4 of 308 answers ran to the cap with empty content (16,383 "reasoning" tokens on a non-reasoning model —
  a Novita artefact); `max_tokens` 6000 caps the waste; rows stay unparsed (imputed 0.5) and are reported.

## Medium-effort test (2026-09-09), `../fbsim_v3_corpus/elicit_medium/`
Same 200 bank items (40 per horizon, family round-robin, seed 11) for 7 models at Fabio's "medium" mapping (`--force-effort
medium`: effort medium; 2,048-token budgets for Haiku 4.5 and Qwen3 235B). Cost $34. Result: NOT an effort artefact. Excess
Brier moved by at most 0.015 in either direction (Fable 0.150→0.135, Sonnet 0.163→0.153, Qwen 0.154→0.144, GPT-5 0.115→0.125,
Haiku 0.113→0.123, o3 and Luna unchanged) while reasoning tokens rose 2–4× (Fable 132→387, GPT-5 0→4,088). Bias stayed
−0.07 to −0.21; every model stayed above the flat-0.5 excess of 0.077 on these items. Slopes rose slightly (Fable 0.20→0.25).
Across the 7 models the ECI correlation of excess Brier was −0.25 at low and −0.04 at medium; of slope +0.68 → +0.75.
Conclusion: the compressed, pessimistic forecasts come from what the models believe about FreeCiv dynamics, not from how
long they think. Script: effort_compare.py.
