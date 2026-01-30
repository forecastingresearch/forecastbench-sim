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

## Key Optimizations

1. **Baseline from savegames**: Instead of running a control fork, we parse the existing end-turn savegame. This is instant vs. 50 turns of simulation.

2. **One fork per condition**: Questions sharing the same condition reuse cached fork results. Running 9 questions with 1 condition = 1 fork execution, not 9.

## Modules

| Module | Purpose |
|--------|---------|
| `conditional_schema.py` | Data model (Condition, ConditionalQuestion, ConditionalQuestionBank) |
| `conditional_generator.py` | Generate condition/question pairs from game data |
| `conditional_runner.py` | Execute forks with optimizations |
| `conditional_io.py` | Serialization + eval-compatible output |

## Condition Types

Currently supported interventions:

| Type | Value | Example |
|------|-------|---------|
| `gold` | Amount to set | `gold:0:5000` — Set player 0 gold to 5000 |
| `gold_add` | Amount to add | `gold_add:0:5000` — Add 5000 gold to player 0 |
| `government` | Government name | `government:1:Republic` — Switch player 1 to Republic |
| `tech` | Tech ID | `tech:0:23` — Grant player 0 Iron Working |

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
