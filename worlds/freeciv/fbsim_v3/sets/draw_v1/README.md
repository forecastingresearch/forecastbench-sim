# FreeCiv question sets — draw v1.5 (2026-09-03)

Corpus: 8 anchors × 1,000 continuations (rerun of 2026-09-02, savegame truth, all channels exact).
Anchors 7001 7003 7005 7008 7010 7011 7014 7022. Truth for every item = the frequency/distribution over ALL
1,000 replays (no half split). Selection is structural (horizon, probability band, family, reveal type,
relation); no item was chosen by its measured effect.

| file | items | what |
|---|---|---|
| bank_750.json | 750 | binary bank: 150/horizon = 5 bands over (0.05,0.95) × 30, family-balanced per band-cell |
| tails_300.json | 300 | binary, 0 < q ≤ 0.05, 60/horizon, family-balanced |
| mirrors_50.json | 50 | binary, 0.95 ≤ q < 1, 10/horizon (our own top-end probe) |
| continuous_300.json | 300 | 60/horizon: 8 count families × 6 + 4 value-at-T metrics × 3; `values` = the 1,000 truth values |
| natcond_600.json | 400 | turn-2 reveal cells at T120–T210, 100/horizon in blocks A9 B9 C1 15 C2 42 D25 (strong-first) |
| natcond_extra_turn1.json | 124 | value-series questions used only by natcond cells; need turn-1 elicitation too |
| COMPOSITION.md, QC_SAMPLES.md, report.html | | tables and random QC samples (report.html = the published page) |

Item fields: `id` (world:index), `world`, `family`, `T`, `text`, `criteria`, `subj` (civ ids), family params
(`k`, `metric`, `x`, `wonder`, `gov`, `tech`, `kind`, `f`), truth `qAll` (binary) or `medAll/iqrAll/p05/p95/values`
(continuous), plus `qA/qB` for reference. Natcond cells add `qid`, `question`, `reveal`, `rev_kind`, `rev_id`,
`block`, `p`, `p_given`, `delta`, `nx`, `from_bank`, `control_no_news`, `control_single_prompt`, `turn2_preamble`.

Elicitation (tmp/fbsim_v3_run/elicit_v1.py, models in models_v1.csv, reasoning effort low where supported): per model
1,524 turn-1 calls (1,400 items + 124 extra questions), 400 turn-2 reveal calls, 276 no-news controls (one per question,
shared by its cells), 300 single-prompt controls = 2,500 calls. Scorer: tmp/fbsim_v3_run/score_v1.py (self-test:
score_selftest.py). Cost: cost_estimate.py.

Scoring, one rule per set: binary → Brier vs qAll and excess Brier (p−q)²; tails additionally reported in bits;
continuous → CRPS of the elicited percentiles vs the empirical distribution in `values` (excess over the ensemble
floor); natcond → turn-2 excess Brier vs p_given, compared with the no-news and single-prompt controls.
No family needs a special case.

Turn-2 prompt: `turn2_preamble` is the one uniform reveal sentence; the event's exact turn is never disclosed.
World reports for the 8 anchors: ../reports/seed<N>/turn_060_report.txt (savegame truth, exact turn-1..60 events).
Generator/draw scripts: tmp/fbsim_v3_run/{bank_v1.py, draw_v1.py, criteria_v1.py, describe_draw.py, report_draw.py}.

2026-09-08 v1.8: controls trimmed per owner — single-prompt control dropped (all cells false); no-news control on 100 cells stratified by block (seed 2026; previous flags kept in natcond_600_v1.7_controls300.json). Prompt order switched to report-first for prefix caching.
