"""Unit tests for the multi-run overlay figure.

Checks the panel/legend structure and that the file is written, using the Agg
backend so nothing tries to open a window. Does not run the simulation.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from micropolis_world.plot_sim import (
    PANEL_GRID,
    PANEL_METRICS,
    plot_repeats,
)


def fake_run(offset: int, nrows: int = 6) -> list[dict]:
    """One run's log rows, with every panelled metric present."""
    return [
        {
            "cityTime": 1071 + i,
            "cityYear": 1922 + i // 12,
            "cityMonth": i % 12,
            **{metric: 100 + offset + i for metric, _color in PANEL_METRICS},
        }
        for i in range(nrows)
    ]


def test_writes_the_figure(tmp_path):
    out = tmp_path / "repeats.png"
    plot_repeats([fake_run(0), fake_run(5)], title="t", output=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_one_line_per_run_in_every_panel():
    import matplotlib.pyplot as plt

    runs = [fake_run(i) for i in range(5)]
    # plot_repeats closes its figure before returning, so the assertions have to
    # run while it is still open: stand in for plt.show to inspect it there.
    captured = {}
    original_show = plt.show

    def capture():
        captured["axes"] = [ax for ax in plt.gcf().get_axes() if ax.get_visible()]
        captured["lines"] = [len(ax.get_lines()) for ax in captured["axes"]]

    plt.show = capture
    try:
        plot_repeats(runs, title="t", output=None)
    finally:
        plt.show = original_show

    assert len(captured["axes"]) == len(PANEL_METRICS)
    assert captured["lines"] == [len(runs)] * len(PANEL_METRICS)


def test_hides_the_unused_grid_slots():
    import matplotlib.pyplot as plt

    nrows, ncols = PANEL_GRID
    spare = nrows * ncols - len(PANEL_METRICS)
    captured = {}
    original_show = plt.show

    def capture():
        axes = plt.gcf().get_axes()
        captured["hidden"] = sum(not ax.get_visible() for ax in axes)

    plt.show = capture
    try:
        plot_repeats([fake_run(0)], title="t", output=None)
    finally:
        plt.show = original_show

    assert captured["hidden"] == spare


def test_legend_names_every_run(tmp_path):
    import matplotlib.pyplot as plt

    captured = {}
    original_show = plt.show

    def capture():
        legend = plt.gcf().legends[0]
        captured["labels"] = [t.get_text() for t in legend.get_texts()]

    plt.show = capture
    try:
        plot_repeats([fake_run(0), fake_run(1), fake_run(2)], title="t", output=None)
    finally:
        plt.show = original_show

    assert captured["labels"] == ["run 1", "run 2", "run 3"]


def test_rejects_an_empty_run_list():
    with pytest.raises(ValueError):
        plot_repeats([], title="t", output=None)
