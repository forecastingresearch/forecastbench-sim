"""Model-facing world report (situation report) + prompt builder."""

from collections import Counter

from . import module_globals as g
from .city_sim import CitySimulation, turn_of

_difficulty_labels = {0: "Easy", 1: "Medium", 2: "Hard"}


def _report_metrics(censor_city_funds: bool) -> list[str]:
    """The g.METRICS the report shows, minus city funds when censoring.

    Shared by CURRENT STATE and HISTORY so the snapshot and the table always
    carry the same metrics as each other.
    """
    if not censor_city_funds:
        return g.METRICS
    return [m for m in g.METRICS if m != g.FUNDS_METRIC]


def _snapshot_section(
    row: dict,
    report_effectiveness: bool = False,
    censor_city_funds: bool = True,
    report_census: bool = True,
) -> list[str]:
    """Current-state block from a single log_data row.

    report_effectiveness includes the three funding-effectiveness readouts
    (road, police, fire). They are omitted by default: they are derived from
    the budget rather than observed, so a variant can ask what the city looks
    like without them.

    censor_city_funds drops every line about the city's money — the funds
    reading, cash flow, and the auto-budget setting.

    report_census (the default) adds the ground-survey tile counts
    (g.SNAPSHOT_CENSUS) the binary tile-count questions resolve on. False
    reproduces the reports built before the flag existed.
    """
    lines = [f"CURRENT STATE (turn {turn_of(row)})", ""]
    difficulty = _difficulty_labels[row["gameLevel"]]
    lines.append(f"  Difficulty: {difficulty}")
    if not censor_city_funds:
        auto_budget_state = "On" if row["autoBudget"] else "Off"
        lines.append(f"  Auto budget: {auto_budget_state}")

    city_class = row.get("cityClass")
    if city_class is not None and 0 <= city_class < len(g.CITY_CLASSES):
        lines.append(f"  City class: {g.CITY_CLASSES[city_class]}")
    for metric in _report_metrics(censor_city_funds):
        label = g.METRIC_LABELS.get(metric, metric)
        lines.append(f"  {label}: {row[metric]}")
    if not censor_city_funds:
        lines.append(f"  Cash flow: {row['cashFlow']}")
    lines.append(f"  Tax rate: {row['cityTax']}%")
    if report_effectiveness:
        lines.append(f"  Road funding effectiveness: {row['roadEffect']}")
        lines.append(f"  Police funding effectiveness: {row['policeEffect']}")
        lines.append(f"  Fire department funding effectiveness: {row['fireEffect']}")
    lines.append("")
    lines.append(
        "  Infrastructure — "
        + ", ".join(f"{label} {row[key]}" for key, label in g.SNAPSHOT_INFRASTRUCTURE)
    )
    if report_census:
        lines.append(
            "  Ground survey — "
            + ", ".join(
                f"{label} {row['census'][key]}" for key, label in g.SNAPSHOT_CENSUS
            )
        )
    return lines


def _history_section(
    log_data: list[dict],
    turn: int,
    history_freq: int,
    history_length: int = -1,
    censor_city_funds: bool = True,
) -> list[str]:
    """Metric history sampled every history_freq turns, ending on the snapshot turn.

    Counted back from the snapshot so the last row is always the state the
    forecast is made from. Turn 0 is added when the stride steps over it, so an
    untruncated table always shows where the city started as well as where it
    stands.

    history_length caps the table at that many rows, keeping the most recent
    ones; -1 keeps every sampled turn. Truncating drops turn 0 along with the
    rest of the early history — the point of a capped window is to show only a
    recent slice, and pinning the start would leave a misleading gap — so the
    header says the table is a window rather than the whole run.

    censor_city_funds drops the city funds column, header included.
    """
    turns = sorted(set(range(turn, -1, -history_freq)) | {0})
    truncated = 0 < history_length < len(turns)
    if truncated:
        turns = turns[-history_length:]
    metrics = _report_metrics(censor_city_funds)
    header = ["Turn"] + [g.METRIC_LABELS.get(m, m) for m in metrics]
    rows = [[str(t)] + [str(log_data[t][m]) for m in metrics] for t in turns]
    if truncated:
        n = len(turns)
        title = (
            f"HISTORY (most recent {n} sample{'' if n == 1 else 's'}, "
            f"every {history_freq} turns)"
        )
    else:
        title = f"HISTORY (every {history_freq} turns)"
    return [title, "", ",".join(header)] + [",".join(row) for row in rows]


def _events_section(events_data: list[dict], cutoff_tick: int) -> list[str]:
    """Disasters (individually, with dates) and recurring advisories (aggregated)."""
    messages = [
        e
        for e in events_data
        if e.get("event") == "sendMessage" and e["tick"] <= cutoff_tick
    ]
    disasters = [e for e in messages if e["messageNum"] in g.DISASTER_MESSAGES]
    advisories = [e for e in messages if e["messageNum"] not in g.DISASTER_MESSAGES]

    lines = ["EVENTS", ""]

    if disasters:
        lines.append(f"  Disasters ({len(disasters)} total):")
        # One incident often fires several messages on the same turn (a plane
        # crash triggers a helicopter response plus explosions), so collapse
        # repeats of the same message within a turn into a count.
        counts = Counter((turn_of(e), e["messageText"]) for e in disasters)
        seen = set()
        for e in disasters:
            key = (turn_of(e), e["messageText"])
            if key in seen:
                continue
            seen.add(key)
            suffix = f" (x{counts[key]})" if counts[key] > 1 else ""
            lines.append(f"    turn {key[0]}: {key[1]}{suffix}")
    else:
        lines.append("  Disasters: none reported.")

    if advisories:
        lines.append("")
        lines.append("  Standing advisories (first and last reported):")
        counts = Counter(e["messageText"] for e in advisories)
        first = {}
        last = {}
        for e in advisories:
            first.setdefault(e["messageText"], turn_of(e))
            last[e["messageText"]] = turn_of(e)
        for text, n in counts.most_common():
            span = (
                f"turn {first[text]}"
                if first[text] == last[text]
                else f"turns {first[text]}–{last[text]}"
            )
            lines.append(f"    {text} (x{n}, {span})")

    return lines


def gen_world_report(
    sim: CitySimulation,
    turn: int,
    history_freq: int,
    label: str,
    snapshot_only: bool = False,
    history_length: int = -1,
    report_effectiveness: bool = False,
    censor_city_funds: bool = True,
    report_census: bool = True,
) -> str:
    """Build the model-facing situation report for `sim` as of `turn`.

    Args:
        sim: A CitySimulation with load_from_disk() already called.
        turn: Index into sim.log_data for the snapshot. Everything after it
            (later log rows, later events) is excluded to avoid leakage.
        history_freq: Sample the metric history every this many turns.
        label: The config's label, which names the file this is saved to. Every
            config gets its own report file, so two prompt variants — or any
            other pair of configs that build the report differently — sit side
            by side instead of one overwriting the other.
        snapshot_only: Omit the HISTORY table, leaving the model the current
            state and nothing about how the city got there. CURRENT STATE
            already carries every HISTORY metric at `turn` — both sections
            iterate _report_metrics() — so this drops the past turns only,
            which is the point: it isolates what the time series is worth to a
            forecaster. history_freq is then unused but still validated, so a
            config that sets it nonsensically fails the same either way.
        history_length: Cap the HISTORY table at this many rows, keeping the
            most recent ones. -1 (the default) keeps every sampled turn.
        report_effectiveness: Include the road, police and fire funding
            effectiveness lines in CURRENT STATE. False (the default) drops
            all three.
        censor_city_funds: Omit everything about the city's money — the city
            funds reading, cash flow and auto-budget lines in CURRENT STATE,
            and the city funds column in HISTORY. True (the default) omits all
            of it, and scenarios.build_corpus then drops the city funds
            question too, so the metric is neither reported nor asked about.
            False reports all of it.
        report_census: Include the ground-survey tile counts (rubble, fire,
            road) in CURRENT STATE. True (the default) includes them; False
            reproduces the reports built before the flag existed. Part of the
            prompt, so flipping it misses the response cache rather than
            mixing variants.

    As a side effect it saves the world report to disk and prints out the path to it.
    """
    if sim.log_data is None or sim.events_data is None:
        raise ValueError("Simulation data not loaded; call sim.load_from_disk() first")
    if not 0 <= turn < len(sim.log_data):
        raise IndexError(
            f"turn {turn} out of range for {len(sim.log_data)} logged turns"
        )
    if history_freq < 1:
        raise ValueError(f"history_freq must be >= 1, got {history_freq}")
    if history_length == 0 or history_length < -1:
        raise ValueError(
            f"history_length must be -1 (unlimited) or >= 1, got {history_length}"
        )

    row = sim.log_data[turn]
    sections = [
        [
            f"City: {sim.city_name}",
            f"Disasters: {'enabled' if sim.disasters else 'disabled'}",
        ],
        _snapshot_section(row, report_effectiveness, censor_city_funds, report_census),
    ]
    if not snapshot_only:
        sections.append(
            _history_section(
                sim.log_data, turn, history_freq, history_length, censor_city_funds
            )
        )
    sections.append(_events_section(sim.events_data, row["tick"]))
    report_text = "\n\n".join("\n".join(section) for section in sections)

    # Labelled rather than suffixed by variant: what goes into a report depends
    # on more than snapshot_only (history_freq, history_length,
    # report_effectiveness, censorCityFunds, and whatever a later variant
    # adds), so keying the file by the config that produced it keeps any two
    # configs' reports side by side instead of one overwriting the other.
    report_path = sim.get_data_file_path(f"worldreport-{label}-T{turn}", ext="txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Read before writing so the message can say whether this run actually
    # changed the report — the interesting case when re-running after a config
    # or engine change, which a bare "Saved" would hide.
    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    if previous is None:
        status = "(new)"
    elif previous == report_text:
        status = "(unchanged)"
    else:
        status = "(CHANGED)"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    if status != "(unchanged)":
        print(f"gen_report: Saved world report to {report_path} {status}")
    return report_text
