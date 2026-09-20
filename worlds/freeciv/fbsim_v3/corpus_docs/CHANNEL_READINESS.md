# Channel readiness for the fbsim v3 corpus (2026-09-02 rerun)

Consolidator: `consolidate_v4.py` (this dir). Input = pulled corpus (`*_data.json.gz` + `*_sgtables.json.gz` +
optional `TRUNCATED` per fork). Output = per-world pickles in the `mine_v3.py` schema plus exact extras.
Validated 2026-09-02/03 on the full corpus (6,080,000 cells, 0 mismatches; first on 20 real forks of seed7001: 15,200 (turn,civ) cells cross-checked between the
serializer series and an independent parse of every per-turn savegame — 0 mismatches on cities, techs,
government, alive, and pairwise diplomacy. `mine_v3.py` runs unchanged on the output.

## Channel per family (final roster, handoff §2)

| family | v4 channel | exactness | semantic notes |
|---|---|---|---|
| NW1 war_at, W6 peace_at, NW2 state reached, S3 wars-at-T, S2 war end | `dstate` (savegame diplomacy per turn) | exact, cross-checked | unchanged |
| EX_government_at, S4 gov-change count | `government` array + `government_change` events from savegame diffs | exact | S4 excludes `to == Anarchy` (spec); Anarchy counts as a state for gov_at |
| EX_tech_discovered, W5 tech lead, NB1 tech threshold, P5/P6/P9/NC3 | `techs_known` series + `tech_discovered` events (savegame tech-set diffs, names via savefile technology_vector) | exact | turn-60 initial-knowledge entries are NOT events (they were in the old ledger) |
| NW5 wonder any, W1 wonder race, S5 civ wonders, NC14 world wonders, P8 first wonder | `wonder_completed` events from per-city improvement diffs, filtered to ruleset genus 0 | exact | Palace (small wonder) excluded at source; builder = city owner on the completion turn. `wonders_count` is rebuilt from the tables (great wonders only); the serializer's series counted the Palace for every civ |
| S6 city founding, P3 civ founds | `city_founded` (new city id appears) | exact | includes cities founded by splinter civs under their own ids (not main civs) |
| W2 directed conquest, P1 civ captures, P7 first capture | `city_conquered` with `new_owner == civ` | exact | captures OF barbarian-held cities count as captures |
| P2 civ losses | `city_conquered` with `prev_owner == civ` | exact | losses TO barbarians count (kind = barbarian) |
| NB6 event counts, NC5 world captures | `city_conquered` (all) | exact | DECISION NEEDED: barbarian captures are included (kind='barbarian'); civil-war transfers are a separate type `civil_war_transfer` and are NOT counted |
| S7 world razings | `city_destroyed` (city id disappears) | exact | — |
| staged: EX_comparative / NB1 on population, cities, territory; NB4 drawdown; NC9 range; NC4 margins; NC11 totals | metrics series | exact for all civs now | can be re-admitted |
| staged: NW4 survival | `is_alive` series (savegame alive flag) + `player_died` events | exact | — |

## What changed vs. the old ledger (why v4 events must be used)
Smoke fork seed1000/rng4002: captures 21 (ledger) vs 109 (savegame), government changes 98 vs 150,
foundings 154 vs 138 (ledger over-counted via "first seen"). Techs and diplomacy were already exact.

## Corpus-level facts to carry into mining
- `end_turn[replay]`, `truncated[replay]`: forks whose observer died before T210 publish with a TRUNCATED marker;
  arrays are NaN after `end_turn`. Final 8 anchors: none so far (7016 was dropped for 3.1% truncation).
- `player_kind_counts`: main / barbarian / splinter players seen per world. Civil wars are common (3 splinters in one fork).
- Half split unchanged: numeric-aware tag sort, min tag reserved, even -> A, odd -> B.

## Still to build (unchanged from handoff §7 step 6)
The S/W/P family sweeps as scripts (their JSON outputs exist; the scripts must be located or rewritten from the
spec semantics), the balanced draw producing 750/300/300, the natcond mine over the new corpus, and the criteria
renderer wiring. All read the v4 pickles.

## World reports
`regen_reports.py` renders the 8 turn-60 reports from the anchors with savegame-truth series, exact turn-1..60 events (from the anchor savegames), great-wonder counts, map size; no territory maps; 0 masked cells. Output: `tmp/fbsim_v3_corpus/reports/seed<N>/turn_060_report.txt`.

## Decisions after the rerun (2026-09-03) — supersede the half-split and effect-size-strata language above and in family_specs
- Truth = ALL 1,000 replays for every set. Measured on this corpus: selecting tails/bands on one half and scoring on the
  other shifts q by ≤ 0.0016 and the tail log-loss by 0.001 bits — negligible — so the split is dropped for binary,
  tails, mirrors and continuous.
- Natural conditionals are NOT selected on effect size (certified cells overstated |Δ| by ~30% and 17% flipped sign on
  a holdout). Cells are drawn in structural blocks (A irrelevant news, B related weak news, C1 mechanical same-series,
  C2 persistence cross-series, D inferential), template-balanced, strong reveals first; effect sizes are reported
  post hoc. Admissibility: n_x ≥ 100, 0.02 < p(Y|X) < 0.98, ≤ 2 distinct reveals per question, ≤ 8 cells per event.
- Timing families (first capture, first wonder, time-to-K) removed: their censored "never" outcome needed a scoring
  special case. Replaced by value-at-T questions (score, population, cities, territory), plain CRPS.
- Government-change count = every recorded transition incl. into/out of anarchy (62% of exits from anarchy restore
  the previous government; the report's timeline shows all of them). War-end family dropped (identical to
  cease-fire-by on pairs at war at t60: 110/110 cells). Wonder counts and events = great wonders only.
- Anchor 7016 replaced by 7011 (3.1% observer deaths); two observer-death forks (7001 rng5482, 7014 rng5290) replaced by
  fresh seeds at the owner's direction; five deterministic hangs substituted (state/SUBSTITUTIONS.md).
- Criteria are rendered from family + params (criteria_v1.py), never from question text; they state only how we
  resolve (recorded quantity, window/turn semantics, ties, what counts), not game knowledge.
Sets and schema: tmp/fbsim_v3_corpus/draw_v1/README.md.
- 2026-09-04: "number of technologies" = real technologies (the savegame tech set carries FreeCiv's A_NONE placeholder for
  every civ; the report's Techs column/series and every tech-count resolver now subtract it). No question is asked about a
  civ (or pair, or world-level state) that is eliminated by its horizon in any replay (admissibility, structural).
  Report: RANKINGS section, Adjective/Nation-ID columns and the EVENTS Metadata column removed; exact diplomatic_change
  events added so the EVENT TYPES list is true; WONDERS and SCIENCE kept.
- 2026-09-04: report gains BORDERS (TURN 60): per pair, whether territories are adjacent on the tile-ownership grid
  (Chebyshev distance 1, x-wrap honoured) and otherwise the gap in tiles; and a header line stating barbarians run at
  default settings (raids begin around turn 60; none exist at the snapshot). Mid-wonder production stays hidden state.
