# CivBench

A package to automatically create forecasting questions from AI-vs-AI FreeCiv games.

A fork of [CivRealm](https://github.com/bigai-ai/civrealm).

## About This Fork

This project builds on the excellent [CivRealm](https://github.com/bigai-ai/civrealm) framework developed by BIGAI, which provides a Gymnasium-compatible environment for the open-source strategy game [Freeciv-web](https://github.com/freeciv/freeciv-web).

This package CivRealm with:
- Configurable AI vs. AI gameplay
- Specialized logging of world state, including:
  - Economic metrics and trade analysis
  - Demographic trends and population statistics
  - Technology advancement tracking
  - Historical event timelines

We use these data to create world reports and forecasting questions.

![Punic War](docs/assets/punic_war_base.jpg)

# Contents

- [About This Fork](#about-this-fork)
- [How It Works](#how-it-works)
  - [Running AI Games with run_world.py](#running-ai-games-with-run_worldpy)
  - [Data Production Pipeline](#data-production-pipeline)
  - [World Report Generation](#world-report-generation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Testing the Installation](#testing-the-installation)
  - [Single player mode (against built-in AIs)](#single-player-mode-against-built-in-ais)
  - [Multiplayer mode](#multiplayer-mode)
- [Trouble Shooting](#trouble-shooting)
- [Original CivRealm Project](#original-civrealm-project)

## How It Works

This fork introduces a complete pipeline for running AI games, generating detailed world reports, and creating forecastingq uestions. Here's how each component works:

### Running AI Games with run_world.py

The [run_world.py](run_world.py) script orchestrates fully-automated AI-vs-AI games and report generation:

**Game Setup:**
- Creates a competitive game with 5 AI players (all using Freeciv's built-in AI)
- Configures game settings: max turns (default 500), AI difficulty, random starting positions
- Connects as a player and toggles to AI control via `/aitoggle`
- Uses a NoOpAgent that simply returns `None` each turn, letting Freeciv AI play

**Recording During Gameplay:**
- Enables `debug.record_action_and_observation` to capture game state every turn
- Preserves all autosave files for complete historical data extraction
- Records full game state as JSON files in `logs/recordings/username/`
- Each turn produces: `turn_N_step_0_state.json` containing complete world state

**Automatic Report Generation:**
- After game completion, automatically generates world reports
- Analyzes all recorded turns (0 to MAX_TURNS)
- Produces HTML reports with visualizations
- Saves to `reports/latest_game/`

**Usage:**
```bash
python run_world.py
```

This single command:
1. Runs a complete 500-turn AI game (takes ~30-60 minutes)
2. Records all game state data
3. Generates comprehensive world reports
4. Opens HTML report in your browser

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
    recording_dir='logs/recordings/myagent2/',
    output_dir='reports/latest_game/',
    report_turns=[30, 100, 500],  # Generate reports at turn 30, 100, and 500
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
- `turn_500_data.json` - Complete extracted metrics (intermediate format)
- `turn_500_report.html` - HTML report
- Embedded PNG charts and visualizations

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
