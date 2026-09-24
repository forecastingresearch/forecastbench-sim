"""Plot a Micropolis run's metrics over time, one subplot each.

The panels are listed in PANEL_METRICS: population, traffic, pollution, crime,
land value and funds.

Works off an already-completed simulation's log/events files, loading them from
disk when the CitySimulation doesn't have them in memory yet. Also overlays a
vertical line for every disaster event found in the events log, color-coded by
disaster type, with a legend listing only the disaster types that actually
occurred in this run.
"""

import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from . import module_globals as g
from .city_sim import CitySimulation

# Line color per disaster type; the names come from g.DISASTER_MESSAGES.
DISASTER_COLORS = {
    "Fire": "tab:red",
    "Monster": "tab:purple",
    "Tornado": "tab:gray",
    "Earthquake": "tab:brown",
    "Plane crash": "tab:orange",
    "Shipwreck": "tab:cyan",
    "Train crash": "tab:olive",
    "Helicopter crash": "gold",
    "Firebombing": "darkred",
    "Explosion": "magenta",
    "Flooding": "tab:blue",
    "Nuclear meltdown": "lime",
    "Riots": "black",
}

# cityTime advances 4 times per in-game month (see update.cpp: cityMonth =
# (cityTime % 48) >> 2), so cityYear/cityMonth alone can't distinguish the 4
# sub-month ticks. Spread them across 4 fixed, roughly-evenly-spaced days
# instead of collapsing them onto a single date per month.
SUB_MONTH_DAYS = [1, 8, 15, 22]

# The metrics to panel, in subplot order (row-major), each with its line color.
# Keys are log fields from g.METRICS; the titles come from g.METRIC_LABELS so a
# panel cannot disagree with how the same metric is named in a report.
PANEL_METRICS = [
    ("cityPop", "tab:blue"),
    ("trafficAverage", "tab:purple"),
    ("pollutionAverage", "tab:orange"),
    ("crimeAverage", "tab:red"),
    ("landValueAverage", "tab:brown"),
    ("totalFunds", "tab:green"),
]

# Rows x columns for the panel grid, wide enough for the six metrics above.
PANEL_GRID = (3, 2)


def date_for(city_time, city_year, city_month):
    # cityMonth is 0-indexed (Jan=0..Dec=11) per the engine's update.cpp.
    day = SUB_MONTH_DAYS[city_time % 4]
    return datetime.date(city_year, city_month + 1, day)


def plot_run(sim: CitySimulation, output: str | None = None) -> None:
    """Build the per-metric subplot figure for a CitySimulation.

    One panel per entry of PANEL_METRICS, sharing a time axis.

    Writes the figure to `output` if given, otherwise shows it interactively.
    """
    if sim.log_data is None or sim.events_data is None:
        sim.load_from_disk()
    rows = sim.log_data or []
    events_data = sim.events_data or []

    dates = [date_for(r["cityTime"], r["cityYear"], r["cityMonth"]) for r in rows]

    disaster_events = []
    for event in events_data:
        if event.get("event") != "sendMessage":
            continue
        label = g.DISASTER_MESSAGES.get(event.get("messageNum"))
        if label is None:
            continue
        date = date_for(event["cityTime"], event["cityYear"], event["cityMonth"])
        disaster_events.append((date, (label, DISASTER_COLORS.get(label, "black"))))

    nrows, ncols = PANEL_GRID
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 3.2 * nrows), sharex=True, squeeze=False
    )
    fig.suptitle(rows[0].get("cityName", sim.city_name))

    flat_axes = [ax for row in axes for ax in row]
    disaster_lines = []  # (axes, Line2D, label, date) for hover lookups
    for ax, (metric, color) in zip(flat_axes, PANEL_METRICS):
        ax.plot(dates, [r[metric] for r in rows], color=color)
        # Capitalized here rather than in METRIC_LABELS, which reads mid-sentence
        # in the reports.
        ax.set_title(g.METRIC_LABELS[metric].capitalize())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, alpha=0.3)
        for date, (label, disaster_color) in disaster_events:
            line = ax.axvline(
                date, color=disaster_color, linestyle="--", linewidth=1, alpha=0.7
            )
            disaster_lines.append((ax, line, label, date))

    # A grid wider than the metric list would leave blank panels showing only
    # their axes, which read as a missing plot rather than an empty slot.
    for ax in flat_axes[len(PANEL_METRICS) :]:
        ax.set_visible(False)

    # Only legend the disaster types that actually occurred in this run.
    legend_labels = {label: color for _, (label, color) in disaster_events}
    if legend_labels:
        handles = [
            plt.Line2D([0], [0], color=color, linestyle="--", linewidth=1)
            for label, color in legend_labels.items()
        ]
        fig.legend(
            handles,
            legend_labels.keys(),
            loc="lower center",
            ncol=len(legend_labels),
            fontsize="small",
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    if legend_labels:
        # The strip reserved for the legend is a fraction of the figure, so it
        # has to shrink as rows are added or it leaves a gap the height of a
        # whole panel.
        fig.subplots_adjust(bottom=0.3 / nrows)

    # Hover tooltip: show the disaster name when the cursor is near one of its
    # vertical lines. Only meaningful in the interactive window (no-op on save).
    if disaster_lines:
        annotations = {}
        for ax, _line, _label, _date in disaster_lines:
            if ax not in annotations:
                annotations[ax] = ax.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox={"boxstyle": "round", "fc": "w", "ec": "0.3"},
                    visible=False,
                    zorder=100,
                )

        def on_move(event):
            for ax, annotation in annotations.items():
                if event.inaxes != ax:
                    if annotation.get_visible():
                        annotation.set_visible(False)
                        event.canvas.draw_idle()
                    continue
                # ~6 pixels of hover tolerance around each line, in display coords.
                # Multiple disasters often land on the same month (e.g. a plane
                # crash triggering a helicopter response + explosion), so collect
                # every line within tolerance rather than stopping at the first
                # match — otherwise the label shown can silently disagree with
                # whichever line color is actually on top at the cursor.
                hits = []
                for line_ax, _line, label, date in disaster_lines:
                    if line_ax is not ax:
                        continue
                    x_display = ax.transData.transform((mdates.date2num(date), 0))[0]
                    distance = abs(event.x - x_display)
                    if distance <= 6:
                        hits.append((distance, date, label))
                if hits:
                    hits.sort()
                    # Same (date, label) pair can repeat if the same disaster type
                    # fires more than once in a month; keep first-seen order.
                    seen = set()
                    lines_text = []
                    for _distance, date, label in hits:
                        key = (date, label)
                        if key in seen:
                            continue
                        seen.add(key)
                        lines_text.append(f"{label} ({date:%b %Y})")
                    annotation.xy = (event.xdata, event.ydata)
                    annotation.set_text("\n".join(lines_text))
                    annotation.set_visible(True)
                else:
                    annotation.set_visible(False)
                event.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    if output:
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")
    else:
        plt.show()
    plt.close(fig)


def save_run_plot(sim: CitySimulation) -> str:
    """Plot `sim` into data/micropolis/runs/<city>/, returning the path written."""
    out = sim.get_plot_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_run(sim, output=str(out))
    return str(out)


def plot_repeats(
    runs: list[list[dict]],
    title: str,
    output: str | None = None,
) -> None:
    """Overlay several runs of one scenario on the PANEL_METRICS panels.

    Same panels and time axis as plot_run, so the two figures can be read
    against each other; the difference is that every run gets its own line per
    panel. That makes the spread between supposedly identical runs the thing
    the figure shows.

    `runs` holds one run's log rows per entry, in run order. Disasters are not
    marked: the vertical lines of plot_run would confound a strike with the
    run-to-run spread this figure is about, so callers pass nodisasters runs.
    """
    if not runs:
        raise ValueError("plot_repeats needs at least one run")

    # Each run gets a color from a qualitative map, so no run reads as the
    # "real" one — unlike PANEL_METRICS' per-metric colors, which would make
    # every line in a panel identical and unattributable.
    colors = plt.get_cmap("tab10").colors

    nrows, ncols = PANEL_GRID
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 3.2 * nrows), sharex=True, squeeze=False
    )
    fig.suptitle(title)

    flat_axes = [ax for row in axes for ax in row]
    for ax, (metric, _color) in zip(flat_axes, PANEL_METRICS):
        for i, rows in enumerate(runs):
            dates = [
                date_for(r["cityTime"], r["cityYear"], r["cityMonth"]) for r in rows
            ]
            ax.plot(
                dates,
                [r[metric] for r in rows],
                color=colors[i % len(colors)],
                linewidth=1,
                alpha=0.8,
            )
        ax.set_title(g.METRIC_LABELS[metric].capitalize())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, alpha=0.3)

    for ax in flat_axes[len(PANEL_METRICS) :]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=1)
        for i in range(len(runs))
    ]
    fig.legend(
        handles,
        [f"run {i}" for i in range(1, len(runs) + 1)],
        loc="lower center",
        ncol=min(len(runs), 8),
        fontsize="small",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    # Same fraction-of-figure reservation as plot_run's disaster legend.
    fig.subplots_adjust(bottom=0.3 / nrows)

    if output:
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")
    else:
        plt.show()
    plt.close(fig)
