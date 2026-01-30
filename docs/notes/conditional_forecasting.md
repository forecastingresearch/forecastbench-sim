# Conditional Forecasting

Conditional forecasting tests whether models can reason about counterfactuals: **"If X happened, how would that change Y?"**

This is implemented via world forking — running parallel simulations where one world has an intervention applied.

## Core Concept

A conditional question has the form:

> Given that [condition], what is P(target)?

For example:
- "If Egypt receives 5000 gold at turn 50, will they have the highest treasury at turn 100?"
- "If Rome switches to Republic government, will their score exceed 200?"

We measure this by comparing outcomes between:
- **Baseline**: The original game trajectory (no intervention)
- **Fork**: A modified game where the condition is applied

## Architecture

### Fork Execution

```
┌─────────────────┐
│  Game Recording │  (turns 1-100, saved every turn)
└────────┬────────┘
         │
         ▼ load turn 50
┌─────────────────┐
│ SavegameModifier│  Apply intervention (gold/tech/government)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ForkManager   │  Run modified game turns 50→100
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compare Outcomes│  Baseline vs Fork for each target question
└─────────────────┘
```

### Resolution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QuestionResolver                          │
│  (single source of truth for ground truth resolution)       │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ game_data dict
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────┴─────────┐         ┌──────────┴──────────┐
    │   game_data.json  │         │  build_game_data()  │
    │   (baseline run)  │         │  (from savegames)   │
    └───────────────────┘         └─────────────────────┘
              │                               │
      Unconditional &                    Fork only
      Baseline conditional
```

Both unconditional and conditional questions use the same `QuestionResolver` for consistency. The only difference is where the data comes from:

- **Unconditional**: Uses `game_data.json` from the full simulation run
- **Conditional baseline**: Uses same `game_data.json`
- **Conditional fork**: Builds a `game_data`-like structure from fork savegames

## Key Optimizations

1. **Baseline from savegames**: Instead of running a control fork, we parse the existing end-turn savegame. This is instant vs. 50 turns of simulation.

2. **One fork per condition**: Questions sharing the same condition reuse cached fork results. Running 9 questions with 1 condition = 1 fork execution, not 9.

## Question Generation

### Condition Types

Currently supported interventions:

| Type | Value | Example |
|------|-------|---------|
| `gold` | Amount to set | `gold:0:5000` — Set player 0 gold to 5000 |
| `gold_add` | Amount to add | `gold_add:0:5000` — Add 5000 gold to player 0 |
| `government` | Government name | `government:1:Republic` — Switch player 1 to Republic |
| `tech` | Tech ID | `tech:0:23` — Grant player 0 Iron Working |

### Templates

Conditional questions use **all the same templates** as unconditional questions:

```python
ALL_TARGET_TEMPLATES = [
    "treasury_comparative",
    "score_comparative",
    "tech_comparative",
    "population_comparative",
    "city_count_comparative",
    "territory_comparative",
    "score_rank_1",
    "tech_discovered",
    "wonder_completed",
    "government_at",
]
```

Rather than filtering templates based on assumed causal relationships, we generate questions for all templates and let the data show what's affected by each intervention.

### Horizons

Questions are generated at multiple time horizons relative to the checkpoint turn:

| Horizon | Offset | Example (checkpoint=50) |
|---------|--------|-------------------------|
| H1 | +30 turns | Turn 80 |
| H2 | +60 turns | Turn 110 |
| H3 | +90 turns | Turn 140 |

## Resolution

Questions are resolved by comparing baseline vs fork outcomes:

```python
# For each question:
answer_control = resolver.resolve(question, baseline_game_data)
answer_intervention = resolver.resolve(question, fork_game_data)
conditional_effect = 1.0 if answer_control != answer_intervention else 0.0
```

### Building Fork game_data

Since forks don't have a full `game_data.json`, we build one from savegames:

```python
fork_game_data = build_game_data_from_savegames(
    recording_dir="logs/recordings/seed0forkgoldadd5000p0",
    turns=[80, 110, 140],  # Resolution turns
    base_game_data=baseline_game_data,  # For civilizations, metadata
)
```

This extracts from savegames:
- `time_series`: treasury, population, scores, territory at each turn
- `snapshots`: aggregated metrics at each turn
- `events`: tech discoveries (from `parse_player_technologies`)

## Comparison: Unconditional vs Conditional

| Aspect | Unconditional | Conditional |
|--------|---------------|-------------|
| Purpose | Forecast future state | Measure intervention effect |
| Data source | `game_data.json` | Baseline: `game_data.json`, Fork: savegames |
| Resolver | `QuestionResolver` | Same `QuestionResolver` |
| Templates | All 10 templates | Same 10 templates |
| Horizons | H1, H2, H3 (+30, +60, +90) | Same horizons |
| Scope | All civilizations | Condition target player |

Example question counts (1 condition, 5 civs, 3 horizons):
- Unconditional: 174 questions (all civs × templates × horizons)
- Conditional: 108 questions (1 condition player × templates × horizons)

## Usage

Conditional forecasting is a two-step process:

### Step 1: Run fork (simulation)

The `run_fork.py` script runs the fork simulation only:

```bash
# Run fork with gold modification
uv run python scripts/run_fork.py \
  --base-seed 0 \
  --checkpoint-turn 50 \
  --end-turn 300 \
  --modification "gold_add:0:5000"
```

This produces:
- `logs/recordings/s0forkgoldadd5000p0/savegames/` — Fork savegames

### Step 2: Generate conditional results (benchmark prep)

Use `generate_conditional_results.py` to create conditional questions:

```bash
uv run python scripts/generate_conditional_results.py \
  --baseline-dir logs/recordings/s0 \
  --fork-dir logs/recordings/s0forkgoldadd5000p0 \
  --condition "gold_add:0:5000" \
  --checkpoint-turn 50 \
  --end-turn 300
```

This produces:
- `logs/recordings/s0forkgoldadd5000p0/conditional_results.json` — Results

### Multiple modifications

```bash
# Step 1: Run fork
uv run python scripts/run_fork.py \
  --base-seed 0 \
  --checkpoint-turn 50 \
  --end-turn 100 \
  --modification "gold_add:0:5000" \
  --modification "tech:0:23"

# Step 2: Generate results
uv run python scripts/generate_conditional_results.py \
  --baseline-dir logs/recordings/s0 \
  --fork-dir logs/recordings/s0forkgoldadd5000p0_tech23p0 \
  --condition "gold_add:0:5000" \
  --condition "tech:0:23" \
  --checkpoint-turn 50 \
  --end-turn 100
```

## Output Format

The script produces two files:

### `conditional_results.json`

Full conditional data including baseline and fork values:

```json
{
  "metadata": {
    "seed": 100,
    "checkpoint_turn": 50,
    "end_turn": 100
  },
  "questions": [
    {
      "conditional_id": "cond_gold_5000_p0_treasury_comparative_p0_p1",
      "condition": {
        "condition_id": "gold_5000_p0",
        "condition_type": "gold",
        "player_id": 0,
        "value": 5000,
        "description": "Egyptian receives 5000 gold"
      },
      "question_text": "Given that Egyptian receives 5000 gold at turn 50: By turn 100, will Egyptian have more gold than Roman?",
      "baseline_answer": false,
      "fork_answer": true,
      "conditional_effect": true
    }
  ]
}
```

### `conditional_questions.json`

Evaluation-compatible format (matches unconditional question format):

```json
[
  {
    "id": "cond_gold_5000_p0_treasury_comparative_p0_p1",
    "question": "Given that Egyptian receives 5000 gold at turn 50: By turn 100, will Egyptian have more gold than Roman?",
    "answer": true,
    "metadata": {
      "condition_id": "gold_5000_p0",
      "baseline_answer": false,
      "conditional_effect": true
    }
  }
]
```

## Prerequisites

Before running conditional forks:

1. **Complete game recording** at `logs/recordings/seed{seed}/`
2. **Savegames downloaded** from Docker to `logs/recordings/seed{seed}/savegames/`

Game data is optional but improves civilization names in question text.

## Evaluation

Use the standard evaluation script with conditional questions:

```bash
uv run python scripts/evaluate_llm_forecasts_parallel.py \
  --questions data/questions/s100/conditional_questions.json \
  --output data/results/s100_conditional_eval.json
```

## Files

| File | Purpose |
|------|---------|
| `src/civrealm/world_reports/questions/conditional_generator.py` | Question generation |
| `src/civrealm/world_reports/questions/conditional_schema.py` | Data model (Condition, ConditionalQuestion, ConditionalQuestionBank) |
| `src/civrealm/world_reports/questions/resolver.py` | Resolution (shared with unconditional) |
| `conditional_runner.py` | Execute forks with optimizations |
| `conditional_io.py` | Serialization + eval-compatible output |
| `scripts/generate_conditional_results.py` | CLI for generating results |
| `scripts/run_fork.py` | Run fork simulations |
