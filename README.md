# CivBench

A benchmark for evaluating LLM forecasting ability using procedurally-generated questions from AI-vs-AI FreeCiv games.

A fork of [CivRealm](https://github.com/bigai-ai/civrealm).

![Punic War](docs/assets/punic_war_base.jpg)


## Quickstart

```bash
# 1. Run games (generates recordings)
python scripts/run_worlds.py --seeds 1-100 --max_turns 300

# 2. Generate data + questions
python scripts/generate_data_batch.py && python scripts/regenerate_questions.py

# 3. Evaluate models
python scripts/evaluate_llm_forecasts_parallel.py --questions-per-difficulty 20
```

## What CivBench does

CivBench provides a benchmark that enables **immediate feedback** on LLM forecasting ability:

- **Immediate feedback** on the efficacy of prompts and scaffolding techniques
- **Low-probability and long-timescale events** can be evaluated more effectively than with real-world data
- **Unbounded question generation** - unlike fixed-question benchmarks (MMLU, FrontierMath, HLE), questions can be automatically generated without significant human input
- **Tunable difficulty** via information masking, prediction horizon, question types (binary, multiple choice, quantile, conditional), and low-probability events
- **Benchmark saturation resistance** - difficulty can be continuously increased as models improve


## How CivBench works

This fork introduces a complete pipeline for running AI games, generating detailed world reports, creating forecasting questions, and evaluating LLMs based on those questions.

### Running AI Games with run_worlds.py

```bash
python scripts/run_worlds.py --seed 42 --max_turns 300 --num_ai_players 5
```
- `--seed` (required): Random seed for deterministic gameplay and unique run identifier
- `--max_turns` (default: 50): Maximum number of turns to simulate
- `--num_ai_players` (default: 5): Total number of AI players in the game

This command:
1. Runs a complete AI game with deterministic behavior
2. Records all game state data
3. Generates comprehensive world reports
4. Outputs report location for viewing

The [run_worlds.py](scripts/run_world.py) script orchestrates fully-automated AI-vs-AI games and report generation:

**Game Setup:**
- Creates a competitive game with N AI players (default 5, all using Freeciv's built-in AI)
- Configures game settings: max turns (default 50), AI difficulty (hard), random starting positions

**Deterministic Runs:**
- Each run is uniquely identified by a **seed** (required argument)
- The seed controls both map generation (`mapseed`) and game logic (`gameseed`)
- Games with the same seed produce identical results, enabling reproducible experiments
- Run ID is derived from seed: `s{seed}` (e.g., seed 42 → run ID `s42`)

**Recording During Gameplay:**
- Records full game state files in `logs/recordings/s{seed}/`
- Each turn produces: `turn_N_step_0_state.json` containing complete world state
- Downloads and persists savegames from Docker container after completion

### Data Production Pipeline

The data production system captures complete game state at every turn through multiple sources:

**1. JSON State Recordings**
- **Location:** `logs/recordings/{username}/turn_N_step_M_state.json`
- **Content:** Full game state including:
  - All player information (gold, science rate, government, etc.)
  - Complete unit roster with positions and status
  - City data (population, production, improvements)
  - Map tiles and terrain
  - Technology tree progress
  - Diplomatic relationships
- **Collection:** Automatic during gameplay when `debug.record_action_and_observation = True`

**2. Savegame Files**
- **Location:** Preserved autosave files in savegame directory
- **Content:** Complete Freeciv game state in binary format
- **Usage:** Parsed by [savegame_parser.py](src/civrealm/world_reports/utils/savegame_parser.py) to extract:
  - Production queues and city improvements
  - Complete civilization names and nation data
  - Historical data not available in JSON snapshots
- **Collection:** Automatic autosaves every turn when `delete_save = False`

**3. Ruleset Metadata**
- **Location:** `logs/recordings/{username}/ruleset.json`
- **Content:** Game rules and nation definitions
- **Usage:** Maps nation IDs to civilization names

**Data Loader ([data_loader.py](src/civrealm/world_reports/data_loader.py)):**
- Indexes all state files by turn number
- Provides efficient access: `get_state(turn)`, `get_states_range(start, end)`
- Caches ruleset data for fast lookups
- Validates data availability before report generation

### World Report Generation

Reports are produced through a two-stage pipeline implemented in [report_generator.py](src/civrealm/world_reports/report_generator.py):

**Stage 1: Data Extraction (Python → JSON)**

The [MetricsCollector](src/civrealm/world_reports/extractors/metrics_collector.py) processes game recordings, extracting:

- **Overview Metrics:** Player rankings, territory control, victory conditions
- **Economic Data:** GDP, production, trade routes, treasury
- **Demographics:** Population, growth rates, city distribution
- **Technology:** Research progress, tech tree advancement
- **Historical Events:** Wars, alliances, city founding, tech discoveries

**Stage 2: Rendering (JSON → HTML)**

The [HTMLRenderer](src/civrealm/world_reports/renderers/html.py) transforms extracted data into reports.

**Report Components:**
- **Charts:** Population trends, economic growth, tech progress
  - Generated by [graph_generator.py](src/civrealm/world_reports/renderers/graph_generator.py)
  - Uses matplotlib with seaborn styling
- **Territory Maps:** Color-coded civilization territories
  - Created by [visualizations.py](src/civrealm/world_reports/utils/visualizations.py)
  - Shows borders and city locations
- **Statistical Tables:** Rankings, comparative metrics, detailed breakdowns
- **Event Timeline:** Chronological history of significant game events
  - Detected by [event_detector.py](src/civrealm/world_reports/utils/event_detector.py)

**Output:**
- `turn_50_data.json` - Complete extracted metrics (intermediate format)
- `turn_50_report.html` - HTML report
- Embedded PNG charts and visualizations

### Question Generation

The question generation system creates forecasting questions from game data.

**Difficulty Dimensions:**

Questions are classified along two orthogonal dimensions:

**Information Availability (I)** - Can the outcome be computed from observable state + known mechanics?

| Level | Definition | Examples |
|-------|------------|----------|
| I1 | Computable from observable state + known mechanics | Tech count, population, score comparatives |
| I2 | Observable trends, but hidden priorities add noise | War declarations, city founding, treasury |
| I3 | Depends on genuinely hidden state | Government changes, wonder completion |

**Time Horizon (H)** - How far ahead is the prediction?

| Level | Turns Ahead | Rationale |
|-------|-------------|-----------|
| H1 | 30 | Short extrapolation; trends likely continue |
| H2 | 60 | Medium-short; early second-order effects |
| H3 | 90 | Medium; requires reasoning about second-order effects |
| H4 | 120 | Medium-long; regime changes begin |
| H5 | 150 | Long; significant regime changes likely |
| H6 | 180 | Very long; compounding uncertainty |
| H7 | 210 | Extreme long; maximum uncertainty |

**Difficulty Matrix:** D = H + I (scores range from 2 to 6)

|  | H1 (short) | H2 (medium) | H3 (long) |
|--|------------|-------------|-----------|
| I1 (computable) | 2 (easiest) | 3 | 4 |
| I2 (noisy) | 3 | 4 | 5 |
| I3 (hidden) | 4 | 5 | 6 (hardest) |

**Question Templates:**

*I1 Questions (Computable):*
- **tech_comparative:** "Will [Civ A] have more technologies than [Civ B] at turn T?"
- **score_comparative:** "Will [Civ A] have a higher score than [Civ B] at turn T?"
- **population_comparative:** "Will [Civ A] have a larger population than [Civ B] at turn T?"

*I2 Questions (Observable with Noise):*
- **at_war_dyad:** "Will [Civ A] and [Civ B] be at war at turn T?"
- **alliance_dyad:** "Will [Civ A] and [Civ B] have an alliance at turn T?"
- **city_founding:** "Will [Civ] found a new city between now and turn T?"
- **conquest:** "Will any city change ownership between now and turn T?"

*I3 Questions (Hidden State):*
- **government_state:** "Will [Civ] be in [government type] at turn T?"
- **wonder_completion:** "Will [Wonder] be completed by any civilization by turn T?"

**Statistics-Based Threshold Calibration:**

For thresholds, we extract empirical percentiles from game data:

```bash
# Extract statistics from all games
python scripts/compute_signal_statistics.py --data-dir data/games --output signal_stats.json
```

This produces percentile distributions for each signal at key turns:

| Signal | Turn 70 (p50) | Turn 125 (p50) | Turn 200 (p50) |
|--------|---------------|----------------|----------------|
| techs_known | 11 | 33 | 46 |
| population | 2 | 31 | 43 |
| territory_size | 27 | 134 | 138 |
| cities_count | 2 | 15 | 15 |

**Threshold Selection Formula:**

For a question "Will signal ≥ threshold by resolution_turn?", we target ~40% True rate:
- Select the 60th percentile at the resolution turn
- 60% of outcomes fall below → 40% at or above → 40% True

```python
from civrealm.world_reports.questions import select_threshold_for_rate

# Get calibrated threshold for ~40% True rate at turn 125
threshold = select_threshold_for_rate(
    signal_name="techs_known",
    resolution_turn=125,
    target_rate=0.4
)
# Returns 34 (60th percentile of tech count at turn 125)
```

**Generating Question Banks:**

```python
from civrealm.world_reports.questions import QuestionGenerator, QuestionResolver

# Generate questions with calibrated thresholds
generator = QuestionGenerator()
question_bank = generator.generate_question_bank(
    game_id="s42",
    game_data=game_data,
    snapshot_turn=60,  # Forecaster sees data up to turn 60
)

# Resolve questions against actual outcomes
resolver = QuestionResolver()
resolved_bank = resolver.resolve_batch(question_bank, game_data)
```

**Computing Base Rates:**

To verify calibration across multiple games:

```bash
# Compute base rates across all games
python scripts/compute_base_rates.py --data-dir data/games --snapshot-turn 60

# Output shows True rate by template and horizon:
# Template                          H1         H2         H3
# tech_count_gte                 54.1%      60.5%      55.2%
# population_gte                 39.1%      57.6%      61.9%
# cities_gte                     41.2%      54.6%      71.4%
```

**Key Files:**
- [signal_statistics.py](src/civrealm/world_reports/questions/signal_statistics.py) - Embedded statistics and threshold functions
- [generator.py](src/civrealm/world_reports/questions/generator.py) - Question generation with calibrated thresholds

### LLM Evaluation

The evaluation system measures LLM forecasting performance using parallel model queries and stratified-batched sampling.

**Running Evaluations:**

```bash
# Standard evaluation: 100 questions (20 per difficulty level), batched by game
python scripts/evaluate_llm_forecasts_parallel.py --seed 42 --questions-per-difficulty 20

# Dry run to inspect sample without querying models
python scripts/evaluate_llm_forecasts_parallel.py --dry-run --questions-per-difficulty 5

# Evaluate only ForecastBench models (for rank correlation validation)
python scripts/evaluate_llm_forecasts_parallel.py --forecastbench-only --questions-per-difficulty 20

# Specify specific models
python scripts/evaluate_llm_forecasts_parallel.py --models claude-3-7-sonnet-20250219 gpt-4o-mini

# Resume from checkpoint after interruption
python scripts/evaluate_llm_forecasts_parallel.py --resume logs/eval_20251216_143000/checkpoint.json
```

**Stratified-Batched Sampling:**

The `--questions-per-difficulty` flag uses a hybrid two-phase sampling approach that provides both **guaranteed difficulty balance** and **token efficiency**:

1. **Phase 1 - Stratified sampling:** Sample exactly N questions from each difficulty level (2-6)
2. **Phase 2 - Game batching:** Regroup sampled questions by game for efficient prompts

| Difficulty | Sample Count | Description |
|------------|--------------|-------------|
| 2 | N questions | Easy: Short horizon (H1) + computable info (I1) |
| 3 | N questions | Medium-easy: H1+I2 or H2+I1 |
| 4 | N questions | Medium: H1+I3, H2+I2, or H3+I1 |
| 5 | N questions | Medium-hard: H2+I3 or H3+I2 |
| 6 | N questions | Hard: Long horizon (H3) + hidden state (I3) |

With `--questions-per-difficulty 20`, you get exactly 100 questions (20 from each difficulty level), grouped into batches by game. Questions from the same game share a single world report in the prompt, reducing token usage by ~80%.

**Parallel Execution:**

The evaluation queries all models in parallel for each batch, with per-provider rate limiting to avoid API throttling. This provides ~7x speedup over sequential evaluation.

**Metrics:**

The evaluation computes:

- **Brier Score:** Mean squared error between predicted probabilities and outcomes. Lower is better (0.0 = perfect, 0.25 = uninformed 50% baseline, 1.0 = maximally wrong).
- **Expected Calibration Error (ECE):** Measures how well predicted probabilities match actual frequencies.
- **Brier by Difficulty:** Breakdown of Brier score for each difficulty level.

```python
from civrealm.metrics import compute_brier_score, compute_calibration_error

predictions = [0.9, 0.1, 0.7]
outcomes = [True, False, True]

brier = compute_brier_score(predictions, outcomes)  # 0.03
ece = compute_calibration_error(predictions, outcomes)
```

**Output:**

Results are saved to `data/evaluations/` with:
- Per-question predictions from each model (probability, latency, errors)
- Per-model aggregate metrics (Brier score, ECE, by-difficulty breakdown)
- Metadata for reproducibility (seed, difficulty distribution, timestamps)

**ForecastBench Validation:**

The default model list includes models with published ForecastBench scores, enabling rank correlation analysis between CivBench and ForecastBench performance.

### Scripts

All utility scripts are in the `scripts/` directory:

**Game Simulation:**

| Script | Description |
|--------|-------------|
| [run_world.py](scripts/run_world.py) | Run a single AI-vs-AI game with deterministic seed |
| [run_worlds.py](scripts/run_worlds.py) | Run multiple games in parallel |

**Data Extraction & Analysis:**

| Script | Description |
|--------|-------------|
| [generate_data_batch.py](scripts/generate_data_batch.py) | Batch extract JSON data from game recordings |
| [compute_signal_statistics.py](scripts/compute_signal_statistics.py) | Extract percentile statistics from game data |
| [compute_base_rates.py](scripts/compute_base_rates.py) | Compute base rates across games |
| [calibrate_thresholds.py](scripts/calibrate_thresholds.py) | Calibrate threshold values |
| [test_world_report.py](scripts/test_world_report.py) | Generate world reports from existing recordings |

**Question Generation:**

| Script | Description |
|--------|-------------|
| [generate_questions.py](scripts/generate_questions.py) | Generate all question types from a single game data file |
| [generate_questions_batch.py](scripts/generate_questions_batch.py) | Generate all question types from all games (parallel processing) |

**LLM Evaluation:**

| Script | Description |
|--------|-------------|
| [evaluate_llm_forecasts_parallel.py](scripts/evaluate_llm_forecasts_parallel.py) | Parallel LLM evaluation with stratified sampling (primary) |
| [evaluate_llm_forecasts.py](scripts/evaluate_llm_forecasts.py) | Sequential evaluation (legacy) |

## Prerequisites

CivRealm requires Python `≥ 3.8` and docker. We have tested on Ubuntu 22.04, Mac OS X, and Windows. 

To test CivRealm on <http://localhost>, please follow the docker installation instructions on <https://bigai-ai.github.io/civrealm/getting_started/requirements.html>.

After starting the Freeciv-web service, you can connect to the Freeciv-web server via the host machine <a href="http://localhost:8080/">localhost:8080</a> using a standard browser.

## Installation

Clone this repository and install:

```bash
git clone <your-fork-url> && cd civrealm
pip install -e .
```

This installs CivRealm and all dependencies needed for world report generation.

## Testing the Installation

Before testing the installation, please make sure that the freeciv-web service is running. You can check the status of the freeciv-web service by running:

```bash
docker ps
```

You should see a docker container named `freeciv-web` running.

### Single player mode (against built-in AIs)

To test the installation, run the following command after installation. This will start a single player game against the built-in AIs with the default settings.

```bash
test_civrealm
```

!!! success
    If the installation is successful, the output should be similar to the following:

    ```bash
    Reset with port: 6300
    Step: 0, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 104, 'move NorthEast')
    Step: 1, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 117, 'move North')
    Step: 2, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 118, 'move North')
    Step: 3, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 119, 'move SouthEast')
    Step: 4, Turn: 1, Reward: 0, Terminated: False, Truncated: False, action: ('unit', 120, 'move SouthEast')
    ```

### Multiplayer mode

To test with multiple players, run the following command in a terminal to start the game with player `myagent`:

```bash
test_civrealm --minp=2 --username=myagent --client_port=6001
```

Then start another terminal and join the game with player `myagent1`:

```bash
test_civrealm --username=myagent1 --client_port=6001
```

<!-- ### Using a different freeciv version

As a standard, the official docker image from the [official repository](https://github.com/freeciv/freeciv-web) will be pulled. If you want to create a custom freeciv server (e.g., different rulesets, customizations, etc.) you can use `build_freeciv_server` to create a custom docker image or run a separate image in parallel. In this case, you might need to adapt src/init_server.py -->

## Original CivRealm Project

This fork builds upon [CivRealm](https://github.com/bigai-ai/civrealm), developed by BIGAI. CivRealm is based on [freeciv-bot](https://github.com/chris1869/freeciv-bot) and integrates with [freeciv-web](https://github.com/freeciv/freeciv-web) and [FCIV-NET](https://github.com/fciv-net/fciv-net).
