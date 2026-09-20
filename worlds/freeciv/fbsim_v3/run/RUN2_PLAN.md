# Run 2 plan (2026-09-19): batch the unconditional sets, keep the natural conditionals as run 1 asked them

Decision (Jaeho, 2026-09-19): bank, tail, mirror and continuous questions are re-elicited at up to 50 questions per
prompt with Fabio's grouping; the natural-conditional arm (turn 1 alone, the revealed fact, the question again) stays
one question per prompt, so run 1's natural-conditional forecasts are reused for every model whose provider pin did
not change, and re-elicited only for the two models whose pin moved to match the Micropolis registry.

## A. Batched turn-1 prompts per model (cap 50, even split within a world; sets mixed in a seeded random order)

| world | bank | tails | mirrors | binary items | binary prompts | prompt sizes | continuous items | continuous prompts |
|---|---|---|---|---|---|---|---|---|
| seed7001 | 66 | 32 | 1 | 99 | 2 | 49, 50 | 34 | 1 |
| seed7003 | 116 | 52 | 8 | 176 | 4 | 44 | 48 | 1 |
| seed7005 | 112 | 38 | 13 | 163 | 4 | 40, 41 | 41 | 1 |
| seed7008 | 112 | 51 | 9 | 172 | 4 | 43 | 37 | 1 |
| seed7010 | 99 | 28 | 2 | 129 | 3 | 43 | 39 | 1 |
| seed7011 | 109 | 36 | 8 | 153 | 4 | 38, 39 | 38 | 1 |
| seed7014 | 99 | 32 | 5 | 136 | 3 | 45, 46 | 37 | 1 |
| seed7022 | 37 | 31 | 4 | 72 | 2 | 36 | 26 | 1 |
| all | 750 | 300 | 50 | 1,100 | 26 | 36 to 50 | 300 | 8 |

34 calls per model for 1,400 questions (run 1: 1,400 calls). A prompt is about 8,000 to 10,000 tokens, of which the
report prefix (about 7,500) is cached on first-party hosts after the first prompt of a world.

## B. Natural-conditional arm, one question per prompt, per model

| arm | items | calls | 22 unchanged models | DeepSeek V3, DeepSeek V4 Flash (re-pinned) |
|---|---|---|---|---|
| turn 1 of the cells' questions (arm t1nc) | 355 questions: 231 bank questions (276 cells) and 124 extra value questions (124 cells) | 355 | run 1, rebuilt from score_items.csv | re-elicited (elicit_natcond_v1.py) |
| turn 2 | 400 cells | 400 | run 1 | re-elicited |
| no-news control | 100 cells on 99 questions | 99 | run 1 | re-elicited |
| total | | 854 | 0 new calls | 854 new calls each |

Run 1's cell forecasts are recoverable exactly: for all 9,591 parsed cells, p1 equals the bank or extra row's p of the
same question, and score_v2 on the rebuilt rows reproduces every model's natcond excess and gain to machine precision.
The raw response texts of 23 models stay in the Drive archive (not needed for scoring; needed for the release).

## C. Cost (list prices; Fabio's batched per-question output; run 1's own per-arm costs for the re-pinned rerun)

| | batched turn 1, 24 models | natcond rerun, 2 models | total |
|---|---|---|---|
| output per question as in Micropolis | $17 | $7.5 | about $24 |
| twice that output | $24 | $7.5 | about $31 |

Largest shares: Fable $3.5 to $5.3, Kimi K2 $2.8 (Novita's hidden reasoning), GPT-5.5 $2.0 to $3.1, Opus 5 $1.6 to $2.5;
DeepSeek V4 Flash's natcond rerun about $4.9, DeepSeek V3's about $2.6. Smoke first (2 prompts per model, 48 calls,
well under $2). Balance on the key's account about $340.

## D. Data management and scoring

| what | where (results/) | rows | used for |
|---|---|---|---|
| batched turn 1, all 24 models | `run2_batched/<model>/results.jsonl` + `calls.jsonl` | arm t1, one row per question, value from the answer block | bank, tails, mirrors, continuous |
| natcond of 22 models | `run2_natcond/from_run1/results.jsonl` | arms t1nc, t2, nonews, values from run 1 | natural conditionals; the extra set |
| natcond of the 2 re-pinned models | `run2_natcond/repinned/<model>/results.jsonl` | arms t1nc, t2, nonews, fresh calls | same |
| scoring | `score_v2.py run2_batched run2_natcond --out scores_v2` | one pass over all three | `SCORES.md`, `score_items.csv`, `score_summary.csv` |

The one collision is the 231 bank questions (276 cells) that also anchor natural-conditional cells: they have a batched turn-1
forecast (arm t1, run 2) and an unbatched one (arm t1nc). score_v2 scores the bank with t1 and takes the cell's p1 from
t1nc, so each forecast is used only under the protocol it was made in. The 124 extra questions exist only unbatched
(t1nc) and score_v2 falls back to that row for them. Cost per row in the batched files is the call's cost divided by
the questions asked (`cost_share`); the per-set cost columns come from summing it.

Paper: Section 2 and Appendix A state that the natural-conditional arm was asked one question per prompt because
turn 2 continues the conversation about one question, and that the other four sets were batched at up to 50.

## E. Smoke test, 2026-09-19/20 (2 batched prompts per model, 48 calls; natcond: 3 questions with cells for the 2 re-pinned models)

- First attempt, non-streamed calls fired at once from 24 processes: 13 calls returned within 15 s, the other 35 sat on
  established connections for over 20 minutes with nothing coming back, while the same request bodies through curl
  completed in 22 to 29 s. Stopped after $1.88. Fix: streamed completions with a 180 s idle timeout per read, one process
  per model started 2 s apart, a progress line per call (`elicit_v2.py`, `launch_v2.sh`).
- Second attempt (resumed): every returned call parsed completely (50/50 or 34/34), finish `stop`, no retries; the answer
  block was omitted twice (GPT-5.4 Nano, Qwen3.5 Flash) and the lenient parser read the Q lines correctly. Calls took
  3 to 82 s except Kimi K2 (71 s binary) and DeepSeek V4 Flash on StreamLake (about 26 tokens/s: a single natural-
  conditional question took 345 s with 9,000 reasoning tokens).
- Cost per question is far below the pre-run estimate: the projection from the smoke is a few dollars for the whole
  batched arm, plus the DeepSeek natcond reruns.

## F. Matched rerun (2026-09-20, Jaeho's decision after the first batched arm)

The first batched arm (results/run2_2026-09-20/batched, 34 prompts per model, seeded shuffle, cap 50 for both kinds) is
complete for all 24 models and is kept as an ablation of order and cap. It showed o4-mini refusing 4 of its 8 continuous
prompts of 38 to 48 questions. The run that the paper uses matches Micropolis's batching exactly where batching is
concerned (results/run2b_2026-09-20/batched, 44 prompts per model): binary cap 50 with horizon-major order (all questions
of one horizon, families in name order, then the next horizon), continuous cap 20 with metric-major order, Fabio's
`1. question` numbering with an indented resolution-criteria line, each prompt sent once with no re-ask (a missing
answer stays missing and is reported in the parse rate), the same answer blocks and parsing rules. Kept from FreeCiv:
the run-1 wording (no persona, scoring rule unnamed), resolution criteria, percentiles p5 to p95, sorting of
non-monotone percentiles (flagged), the 32,768-token output cap, cached report prefixes on first-party hosts,
streamed transport. The natural-conditional arm is unchanged (results/run2_2026-09-20/natcond).
