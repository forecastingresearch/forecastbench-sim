# Continuous Conditional Questions — Implementation Plan

**Goal:** Generate continuous conditional questions by wiring the 6 existing continuous templates into the conditional question generator. Enables testing H3 from the continuous questions experiment.

**Design decisions:**
- Affected civ only (not all civs) — cleanest causal signal
- All 6 metrics × all condition types — no curated pairings, analysis filters post-hoc
- Question text: "If {condition}, {continuous question}?" (e.g., "If Egypt receives 5000 gold, how much gold will Egypt have at turn 90?")

## Tasks

### Task 1: Add continuous templates to conditional generator

**File:** `src/civrealm/world_reports/questions/conditional_generator.py`

**Changes:**
1. Add 6 continuous template IDs to `ALL_TARGET_TEMPLATES` (line 31):
   ```python
   "techs_continuous",
   "treasury_continuous",
   "population_continuous",
   "cities_count_continuous",
   "territory_continuous",
   "scores_continuous",
   ```

2. Add `elif` branch in `_generate_for_template()` (after line 449) for continuous templates:
   - Match on the 6 continuous template IDs
   - Generate ONE question per condition, targeting the affected civ only (`condition.player_id`)
   - Parameters: `civ`, `player_id`, `metric`, `resolution_turn`, `checkpoint_turn`

**Expected:** `_generate_for_template()` produces `ConditionalQuestion` instances for continuous templates with affected-civ-only targeting.

---

### Task 2: Widen ConditionalResult answer types for numeric values

**File:** `src/civrealm/world_reports/questions/conditional_schema.py`

**Changes:**
1. Change `answer_control` type from `bool | None` to `bool | int | float | None` (line 103)
2. Change `answer_intervention` type from `bool | int | float | None` to match (line 107)
3. Update `conditional_effect` docstring (line 110) — for continuous answers, this will be the signed numeric difference instead of binary 1.0/0.0

**Expected:** Schema accepts both boolean answers (existing binary templates) and numeric answers (new continuous templates).

---

### Task 3: Add continuous resolution in conditional runner

**File:** `src/civrealm/world_reports/questions/conditional_runner.py`

**Changes:**
1. Add continuous branches in `_resolve_target_question()` (after line 576):
   ```
   treasury_continuous → player_states[player_id]["gold"]
   techs_continuous → player_states[player_id]["techs"]
   population_continuous → player_states[player_id]["population"]
   cities_count_continuous → player_states[player_id]["cities"]
   territory_continuous → player_states[player_id]["landarea"]
   scores_continuous → player_states[player_id]["score"] (with proxy fallback)
   ```
   Return numeric value (int/float) instead of bool.

2. Update conditional_effect computation in `run_conditional_question()` (line 220) and `run_batch()` (line 355):
   - If both answers are numeric (not bool): `conditional_effect = answer_intervention - answer_control`
   - If both answers are bool (existing behavior): `conditional_effect = 1.0 if different else 0.0`

**Expected:** `_resolve_target_question()` returns numeric values for continuous templates. Conditional effect captures the magnitude of the intervention's impact.

---

### Task 4: Add continuous question text construction in conditional I/O

**File:** `src/civrealm/world_reports/questions/conditional_io.py`

**Changes:**
1. Add continuous branches in `_build_target_question()` (after line 399):
   ```
   techs_continuous → "how many technologies would {civ} have by turn {resolution_turn}?"
   treasury_continuous → "how much gold would {civ} have at turn {resolution_turn}?"
   population_continuous → "what would {civ}'s population be at turn {resolution_turn}?"
   cities_count_continuous → "how many cities would {civ} have at turn {resolution_turn}?"
   territory_continuous → "how many tiles would {civ} control at turn {resolution_turn}?"
   scores_continuous → "what would {civ}'s score be at turn {resolution_turn}?"
   ```

2. Update `to_question_instances()` (line 232):
   - For continuous conditional questions, set `question_type="continuous"` on the `QuestionInstance`
   - Use `Resolution(answer=True, value_at_resolution=numeric_value)` for continuous resolutions (matching the pattern in `resolver.py:173`)

**Expected:** Continuous conditional questions produce properly formatted question text ("If Egypt receives 5000 gold, how much gold would Egypt have at turn 90?") and convert to `QuestionInstance` objects with `question_type="continuous"`.

---

### Task 5: Update generation script for continuous resolution

**File:** `scripts/generate_conditional_results.py`

**Context:** This is the actual generation path used in practice. It uses `QuestionResolver` (not `ConditionalQuestionRunner`) to resolve questions against game_data built from savegames. Two issues:

**Changes:**
1. Add `techs_known` and `cities_count` to `build_game_data_from_savegames()` time_series (line 152):
   ```python
   for metric in ["treasury", "population", "scores", "territory_size", "techs_known", "cities_count"]:
   ```
   And add the data mappings:
   ```python
   game_data["time_series"]["techs_known"][str(turn)][pid_str] = state.get("techs", 0)
   game_data["time_series"]["cities_count"][str(turn)][pid_str] = state.get("cities", 0)
   ```

2. Update resolution handling (line 323-341) to extract `value_at_resolution` for continuous templates:
   - For continuous templates, use `resolution.value_at_resolution` as the answer (not `resolution.answer`, which is just `True`/`False` indicating data presence)
   - Compute `conditional_effect` as signed numeric difference for continuous answers

**Expected:** Script correctly resolves continuous conditional questions with numeric values and computes meaningful conditional effects.

---

### Task 6: Run generation against existing fork data

**Script:** `scripts/generate_conditional_results.py`

**Steps:**
1. Run the script against each existing fork savegame directory (same forks already computed for binary conditional questions)
2. Since continuous templates are now in `ALL_TARGET_TEMPLATES`, the script will automatically generate and resolve continuous conditional questions alongside binary ones
3. Verify output: check that `conditional_results.json` now contains continuous conditional questions with numeric `answer_control`/`answer_intervention` values
4. Report counts and conditional effect statistics for continuous vs binary

**Expected:** `conditional_results.json` files updated with ~54 additional continuous conditional questions per game. No new fork simulations needed — reuses existing savegames.

---

## Files changed (5)

| File | Change type | Lines affected |
|------|-------------|---------------|
| `conditional_generator.py` | Add template IDs + new branch | ~20 lines added |
| `conditional_schema.py` | Widen type annotations | ~3 lines changed |
| `conditional_runner.py` | Add resolution branches + effect calc | ~30 lines added |
| `conditional_io.py` | Add text builders + question_type | ~25 lines added |
| `generate_conditional_results.py` | Add metrics to time_series + continuous resolution | ~15 lines changed |

## Files NOT changed

- `templates.py` — continuous templates already defined
- `schema.py` — `QuestionInstance.question_type` already supports `"continuous"`
- `resolver.py` — already handles `resolution_type="continuous"` via `_resolve_continuous()`
- `generator.py` — unconditional generator, not involved

## Volume estimate

~3 conditions/game × 6 metrics × 3 horizons (H1-H3) × 1 question/condition = ~54 continuous conditional questions per game. With 14 games: ~756 total.

Combined with existing: 2,910 continuous unconditional + 6,516 binary + ~756 continuous conditional.
