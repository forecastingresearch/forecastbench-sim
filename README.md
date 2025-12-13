# CivBench

A benchmark for evaluating LLM forecasting ability using procedurally-generated questions from AI-vs-AI FreeCiv games.

A fork of [CivRealm](https://github.com/bigai-ai/civrealm).

## About This Fork

This project builds on the excellent [CivRealm](https://github.com/bigai-ai/civrealm) framework developed by BIGAI, which provides a Gymnasium-compatible environment for the open-source strategy game [Freeciv-web](https://github.com/freeciv/freeciv-web).

This package extends CivRealm with:
- Configurable AI vs. AI gameplay with deterministic seeds
- Specialized logging of world state, including:
  - Economic metrics and trade analysis
  - Demographic trends and population statistics
  - Technology advancement tracking
  - Historical event timelines

We use these data to create world reports and forecasting questions.

## Research Contributions

CivBench provides a benchmark that enables **immediate feedback** on LLM forecasting ability:

- **Immediate feedback** on the efficacy of prompts and scaffolding techniques
- **Low-probability and long-timescale events** can be evaluated more effectively than with real-world data
- **Unbounded question generation** - unlike fixed-question benchmarks (MMLU, FrontierMath, HLE), questions can be automatically generated without significant human input
- **Tunable difficulty** via information masking, prediction horizon, question types (binary, multiple choice, quantile, conditional), and low-probability events
- **Benchmark saturation resistance** - difficulty can be continuously increased as models improve

### Research Questions

1. **Correlation with existing benchmarks**: Do model rankings on CivBench correlate with ForecastBench and other forecasting benchmarks?
2. **Human baselines**: How do models compare to superforecaster performance?
3. **Extensibility**: Can CivBench serve as a harder, unsaturated extension of existing benchmarks?
4. **Prompt/method A/B testing**: Which prompting strategies and scaffolding methods improve forecasting performance?

![Punic War](docs/assets/punic_war_base.jpg)

# Contents

- [About This Fork](#about-this-fork)
- [Research Contributions](#research-contributions)
- [How It Works](#how-it-works)
  - [Running AI Games with run_world.py](#running-ai-games-with-run_worldpy)
  - [Data Production Pipeline](#data-production-pipeline)
  - [World Report Generation](#world-report-generation)
  - [Question Generation](#question-generation)
  - [Scripts](#scripts)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Testing the Installation](#testing-the-installation)
  - [Single player mode (against built-in AIs)](#single-player-mode-against-built-in-ais)
  - [Multiplayer mode](#multiplayer-mode)
- [Trouble Shooting](#trouble-shooting)
- [Original CivRealm Project](#original-civrealm-project)

## How It Works

This fork introduces a complete pipeline for running AI games, generating detailed world reports, and creating forecasting questions. Here's how each component works:

### Running AI Games with run_world.py

The [run_world.py](scripts/run_world.py) script orchestrates fully-automated AI-vs-AI games and report generation:

**Game Setup:**
- Creates a competitive game with N AI players (default 5, all using Freeciv's built-in AI)
- Configures game settings: max turns (default 50), AI difficulty (hard), random starting positions
- Connects as a player and toggles to AI control via `/aitoggle`
- Uses a NoOpAgent that simply returns `None` each turn, letting Freeciv AI play

**Deterministic Runs:**
- Each run is uniquely identified by a **seed** (required argument)
- The seed controls both map generation (`mapseed`) and game logic (`gameseed`)
- Games with the same seed produce identical results, enabling reproducible experiments
- Run ID is derived from seed: `s{seed}` (e.g., seed 42 → run ID `s42`)

**Recording During Gameplay:**
- Enables `debug.record_action_and_observation` to capture game state every turn
- Preserves all autosave files for complete historical data extraction
- Records full game state as JSON files in `logs/recordings/s{seed}/`
- Each turn produces: `turn_N_step_0_state.json` containing complete world state
- Downloads and persists savegames from Docker container after completion

**Automatic Report Generation:**
- After game completion, automatically generates world reports
- Analyzes all recorded turns (0 to max_turns)
- Produces HTML reports with visualizations
- Saves to `reports/s{seed}/`

**Usage:**
```bash
python scripts/run_world.py --seed 42
python scripts/run_world.py --seed 42 --max_turns 100
python scripts/run_world.py --seed 42 --max_turns 100 --num_ai_players 7
```

**Arguments:**
- `--seed` (required): Random seed for deterministic gameplay and unique run identifier
- `--max_turns` (default: 50): Maximum number of turns to simulate
- `--num_ai_players` (default: 5): Total number of AI players in the game

This command:
1. Runs a complete AI game with deterministic behavior
2. Records all game state data
3. Generates comprehensive world reports
4. Outputs report location for viewing

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

The [MetricsCollector](src/civrealm/world_reports/extractors/metrics_collector.py) processes game recordings:

```python
# Load all states from turn 0 to target turn
states = data_loader.get_states_range(0, target_turn)

# Extract metrics across all categories
collector = MetricsCollector()
data = collector.collect_all(states, config, data_loader)

# Save intermediate JSON for reproducibility
write_world_data(data, 'turn_500_data.json')
```

**What gets extracted:**
- **Overview Metrics:** Player rankings, territory control, victory conditions
- **Economic Data:** GDP, production, trade routes, treasury
- **Demographics:** Population, growth rates, city distribution
- **Technology:** Research progress, tech tree advancement
- **Historical Events:** Wars, alliances, city founding, tech discoveries

**Stage 2: Rendering (JSON → HTML)**

The [HTMLRenderer](src/civrealm/world_reports/renderers/html.py) transforms extracted data into reports:

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

**Configuration ([config.py](src/civrealm/world_reports/config.py)):**

```python
report_config = ReportConfig(
    recording_dir='logs/recordings/s42/',  # s{seed} format
    output_dir='reports/s42/',
    report_turns=[30, 50],  # Generate reports at turn 30 and 50
    enabled_sections=['overview', 'historical_events', 'economics',
                     'demographics', 'technology'],
    formats=['html'],
    plot_style='seaborn',
    dpi=150
)

generator = ReportGenerator(report_config)
generator.generate_reports()
```

**Output:**
- `turn_50_data.json` - Complete extracted metrics (intermediate format)
- `turn_50_report.html` - HTML report
- Embedded PNG charts and visualizations

### Question Generation

The question generation system creates forecasting questions from game data, with thresholds calibrated from empirical statistics across 103 game simulations.

**Question Types:**

Questions are organized into three signal types based on predictability:
- **B1 (High base rate):** Tech count, population, score - signals that generally increase
- **B2 (Medium base rate):** Territory, treasury, cities - signals with more variance
- **B3 (Low base rate):** Events like wars, alliances, conquests - harder to predict

Each question has a time horizon:
- **H1 (Short):** ≤30 turns ahead - immediate predictions
- **H2 (Medium):** 30-100 turns ahead - medium-term forecasts
- **H3 (Long):** >100 turns ahead - long-range predictions

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
    snapshot_turn=50,  # Forecaster sees data up to turn 50
)

# Resolve questions against actual outcomes
resolver = QuestionResolver()
resolved_bank = resolver.resolve_batch(question_bank, game_data)
```

**Computing Base Rates:**

To verify calibration across multiple games:

```bash
# Compute base rates across all games
python scripts/compute_base_rates.py --data-dir data/games --snapshot-turn 50

# Output shows True rate by template and horizon:
# Template                          H1         H2         H3
# tech_count_gte                 54.1%      60.5%      55.2%
# population_gte                 39.1%      57.6%      61.9%
# cities_gte                     41.2%      54.6%      71.4%
```

**Key Files:**
- [signal_statistics.py](src/civrealm/world_reports/questions/signal_statistics.py) - Embedded statistics and threshold functions
- [generator.py](src/civrealm/world_reports/questions/generator.py) - Question generation with calibrated thresholds

### Scripts

All utility scripts are in the `scripts/` directory:

| Script | Description |
|--------|-------------|
| [run_world.py](scripts/run_world.py) | Run a single AI-vs-AI game with deterministic seed |
| [run_worlds.py](scripts/run_worlds.py) | Run multiple games in parallel |
| [generate_data_batch.py](scripts/generate_data_batch.py) | Batch extract JSON data from game recordings |
| [compute_signal_statistics.py](scripts/compute_signal_statistics.py) | Extract percentile statistics from game data |
| [compute_base_rates.py](scripts/compute_base_rates.py) | Compute base rates across games |
| [generate_questions.py](scripts/generate_questions.py) | Generate question banks from game data |
| [calibrate_thresholds.py](scripts/calibrate_thresholds.py) | Calibrate threshold values |

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
