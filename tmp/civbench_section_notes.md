# CivBench section notes

## Recommended main-text footprint

Target `0.75-1.25` NeurIPS pages in two-column format.

- If you include a schematic figure (`world -> report -> questions -> outcomes`): aim for about `1.0-1.25` pages.
- If you do not include a figure: aim for about `0.75-1.0` page.
- Keep exact template counts, seed counts, prompt variants, and conditional/forking details in the appendix unless a later section depends on them directly.

This is shorter than benchmark-introduction papers because the paper's main contribution is the capability/overconfidence result, not the benchmark itself.

## Recommended subsection structure

### 1. CivBench overview

Purpose: explain what the benchmark is, what the model sees, and what it predicts.

### 2. Why Freeciv?

Purpose: justify the simulated world. This is where to explain the lineage:

- `Freeciv` is the open-source turn-based empire-building game.
- `Freeciv-web` is the browser/server implementation.
- `CivRealm` turns that game into an agent environment.
- `CivBench` repurposes those rollouts for forecasting rather than control.

Keep this to one short paragraph.

### 3. Forecasting tasks and metrics

Purpose: explain binary vs. continuous forecasting, the multi-horizon setup, and why the continuous variables matter for the paper's later claims.

This subsection should explicitly name the variables that later drive the paper:

- `treasury`
- `territory`
- `population`
- `cities`

Call these the "disruptable" variables, and contrast them with more stable targets such as technologies and composite score.

## What to emphasize for this paper

The slides suggest the section should support four later claims:

1. Binary CivBench performance tracks general capability in the expected direction.
2. Continuous CivBench performance becomes anti-capability at longer horizons.
3. The reversal is concentrated in disruptable targets whose trajectories can crash because of war, conquest, or other endogenous shocks.
4. CivBench is useful because it creates these shocks in a controlled but nontrivial environment.

Accordingly, the section should emphasize:

- multi-agent interaction
- long horizons
- endogenous shocks
- natural-language world reports
- quantile forecasts for continuous outcomes

It should de-emphasize:

- leaderboard details
- exhaustive template inventories
- repo-specific counts that may change

## Comparable-paper standard

Two useful anchors:

- `ForecastBench` spends roughly `3.5` pages on benchmark construction, leaderboard, and datasets because the benchmark itself is the paper's central contribution.
- `SWE-bench` spends roughly `1.5-2.0` pages on benchmark construction, task formulation, and benchmark features because the benchmark is again central.

For *Is Capability a Liability?*, CivBench is closer to enabling infrastructure than to the whole claim, so the right target is about half of that: around `1` page in the main text, with extra mechanics moved to the appendix.

## Drop-in LaTeX draft

This draft deliberately uses a compact, procedural tone similar to benchmark papers such as ForecastBench and SWE-bench.

```tex
\section{CivBench}

\subsection{Overview}

We study model forecasting in a simulated strategic world rather than on static text alone. CivBench is a forecasting benchmark built on rollouts from Freeciv, an open-source turn-based empire-building game, via the CivRealm agent interface. The benchmark first generates AI-vs.-AI games, then freezes each game at a snapshot turn and converts the observed state into a natural-language world report. Given this report, a model is asked to forecast future properties of the same world at multiple horizons. The resulting setup preserves many features that matter for forecasting under uncertainty: long time horizons, strategic interaction among multiple players, partially predictable growth dynamics, and occasional discontinuous shocks such as war, conquest, and regime change.

This construction is useful for the present paper because it separates two problems that are often entangled in real-world forecasting. On the one hand, the model receives rich state information about the current world. On the other hand, the future remains genuinely uncertain because later outcomes depend on endogenous interactions among civilizations rather than on a fixed continuation of a known time series. CivBench therefore provides a controlled setting in which models can succeed by extrapolating genuine regularities, but can also fail when they are too confident in smooth growth trajectories.

\subsection{Why Freeciv?}

Freeciv is a long-running open-source strategy game in the Civilization tradition, and Freeciv-web provides a browser-accessible server/client implementation of the same game. CivRealm builds on this stack to expose Freeciv as an environment for decision-making agents, including language-based agents. We build on CivRealm in a different way: instead of evaluating action selection, we use its simulated worlds as a source of forecasting problems. This choice is attractive because Freeciv combines several properties that are difficult to obtain simultaneously in simpler benchmarks: a long horizon, many interacting state variables, strategic multi-agent feedback, and shocks that arise endogenously from the game rather than from exogenous noise injected by the benchmark designer.

\subsection{Forecasting tasks and metrics}

From each snapshot, CivBench generates both binary and continuous forecasting questions over multiple future horizons. Binary questions ask whether an event or relation will hold at a later turn, such as whether one civilization will exceed another on territory or technology. These questions are evaluated with standard probabilistic scoring rules such as Brier score, and we also track calibration. Continuous questions ask for future values of state variables such as treasury, territory, population, cities, technology count, or overall score. Models answer these by providing quantiles of a predictive distribution, which allows us to evaluate both point accuracy and distributional calibration using metrics such as CRPS and median error.

For this paper, the key distinction is between relatively stable targets and what we call \emph{disruptable} targets. Technologies often follow a comparatively smooth curriculum, and score is a composite measure that averages over multiple components. By contrast, treasury, territory, population, and city count can all collapse when war, conquest, or other strategic disruptions intervene. This distinction matters because the central empirical pattern in the paper appears most clearly on the disruptable continuous variables and at longer horizons. CivBench is therefore not only a source of many forecasting questions; it is a way of testing whether models represent downside risk appropriately in worlds where growth is common but reversals are structurally possible.
```

## Suggested citations

Main text:

- `Qi et al. (2024)` for CivRealm.
- `The Freeciv Project` for Freeciv.
- `The Freeciv-web Project` for Freeciv-web.

Tone/benchmark-comparison references:

- `Karger et al. (2025)` for ForecastBench.
- `Jimenez et al. (2023)` for SWE-bench.

## Sources consulted

- Repo mechanics:
  - `/Users/jaeholee0404/civbench/README.md`
  - `/Users/jaeholee0404/civbench/scripts/run_world.py`
  - `/Users/jaeholee0404/civbench/scripts/generate_data_batch.py`
  - `/Users/jaeholee0404/civbench/scripts/generate_questions_batch.py`
  - `/Users/jaeholee0404/civbench/src/civrealm/world_reports/questions/templates.py`
  - `/Users/jaeholee0404/civbench/src/civrealm/world_reports/questions/schema.py`
  - `/Users/jaeholee0404/civbench/src/civrealm/world_reports/txt_report.py`
  - `/Users/jaeholee0404/civbench/data/conditional/README.md`
- Slides:
  - `/Users/jaeholee0404/Downloads/[FRI] Is Capability a Liability_.pptx`
- Web:
  - `https://www.freeciv.org/`
  - `https://github.com/freeciv/freeciv-web`
  - `https://openreview.net/forum?id=Dco5HKrZLk`
  - `https://openreview.net/pdf/2438053006dce815a53229bcb3995810f4b0fda5.pdf`
  - `https://ar5iv.labs.arxiv.org/html/2310.06770`
