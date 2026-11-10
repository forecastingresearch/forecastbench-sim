#!/usr/bin/env python3
"""Plot population, funds, crime, and pollution over time from run_sim.js JSONL output.

Also overlays a vertical line for every disaster event found in the companion
event log written by run_sim.js (same directory, "log-" filename prefix
swapped for "events-"), color-coded by disaster type, with a legend listing
only the disaster types that actually occurred in this run.

Usage:
    python3 cli/plot_sim_run.py haight.jsonl
    python3 cli/plot_sim_run.py haight.jsonl --output haight.png
"""

import argparse
import datetime
import json
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# sendMessage messageNum -> (label, color) for messages that represent a
# disaster actually being triggered (see packages/micropolis-engine/src/
# text.h for the enum and message.cpp's sound-effect switch for the set of
# messages tied to a disaster occurring, as opposed to a status/advisory
# message like "Pollution very high").
DISASTER_MESSAGES = {
    20: ('Fire', 'tab:red'),
    21: ('Monster', 'tab:purple'),
    22: ('Tornado', 'tab:gray'),
    23: ('Earthquake', 'tab:brown'),
    24: ('Plane crash', 'tab:orange'),
    25: ('Shipwreck', 'tab:cyan'),
    26: ('Train crash', 'tab:olive'),
    27: ('Helicopter crash', 'gold'),
    30: ('Firebombing', 'darkred'),
    32: ('Explosion', 'magenta'),
    42: ('Flooding', 'tab:blue'),
    43: ('Nuclear meltdown', 'lime'),
    44: ('Riots', 'black'),
}

# cityTime advances 4 times per in-game month (see update.cpp: cityMonth =
# (cityTime % 48) >> 2), so cityYear/cityMonth alone can't distinguish the 4
# sub-month ticks. Spread them across 4 fixed, roughly-evenly-spaced days
# instead of collapsing them onto a single date per month.
SUB_MONTH_DAYS = [1, 8, 15, 22]


def date_for(city_time, city_year, city_month):
    # cityMonth is 0-indexed (Jan=0..Dec=11) per the engine's update.cpp.
    day = SUB_MONTH_DAYS[city_time % 4]
    return datetime.date(city_year, city_month + 1, day)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('jsonl_file', help='Path to the JSONL file written by run_sim.js')
parser.add_argument('-o', '--output', help='Save the plot to this PNG file instead of showing it interactively')
args = parser.parse_args()

rows = []
with open(args.jsonl_file) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

dates = [date_for(r['cityTime'], r['cityYear'], r['cityMonth']) for r in rows]
pop = [r['cityPop'] for r in rows]
funds = [r['totalFunds'] for r in rows]
crime = [r['crimeAverage'] for r in rows]
pollution = [r['pollutionAverage'] for r in rows]

# run_sim.js writes stats to <dir>/log-seed<seed>[-nodisasters].jsonl and the
# companion event log next to it as <dir>/events-seed<seed>[-nodisasters].jsonl.
stats_dir, stats_name = os.path.split(args.jsonl_file)
events_path = os.path.join(stats_dir, 'events-' + stats_name[len('log-'):])
disaster_events = []
if os.path.exists(events_path):
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get('event') != 'sendMessage':
                continue
            disaster = DISASTER_MESSAGES.get(event.get('messageNum'))
            if disaster is None:
                continue
            date = date_for(event['cityTime'], event['cityYear'], event['cityMonth'])
            disaster_events.append((date, disaster))
else:
    print(f'No event log found at {events_path}, skipping disaster overlay.')

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
fig.suptitle(rows[0].get('cityName', args.jsonl_file))

plots = [
    (axes[0][0], pop, 'Population', 'tab:blue'),
    (axes[0][1], funds, 'Funds ($)', 'tab:green'),
    (axes[1][0], crime, 'Crime Average', 'tab:red'),
    (axes[1][1], pollution, 'Pollution Average', 'tab:orange'),
]
disaster_lines = []  # (axes, Line2D, label, date) for hover lookups
for ax, values, title, color in plots:
    ax.plot(dates, values, color=color)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True, alpha=0.3)
    for date, (label, disaster_color) in disaster_events:
        line = ax.axvline(date, color=disaster_color, linestyle='--', linewidth=1, alpha=0.7)
        disaster_lines.append((ax, line, label, date))

# Only legend the disaster types that actually occurred in this run.
legend_labels = {label: color for _, (label, color) in disaster_events}
if legend_labels:
    handles = [
        plt.Line2D([0], [0], color=color, linestyle='--', linewidth=1)
        for label, color in legend_labels.items()
    ]
    fig.legend(handles, legend_labels.keys(), loc='lower center', ncol=len(legend_labels), fontsize='small')

fig.autofmt_xdate()
fig.tight_layout()
if legend_labels:
    fig.subplots_adjust(bottom=0.15)

# Hover tooltip: show the disaster name when the cursor is near one of its
# vertical lines. Only meaningful in the interactive window (no-op on save).
if disaster_lines:
    annotations = {}
    for ax, _line, _label, _date in disaster_lines:
        if ax not in annotations:
            annotations[ax] = ax.annotate(
                '', xy=(0, 0), xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', fc='w', ec='0.3'),
                visible=False, zorder=100,
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
                    lines_text.append(f'{label} ({date:%b %Y})')
                annotation.xy = (event.xdata, event.ydata)
                annotation.set_text('\n'.join(lines_text))
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)
            event.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

if args.output:
    fig.savefig(args.output, dpi=150)
    print(f'Wrote {args.output}')
else:
    plt.show()
