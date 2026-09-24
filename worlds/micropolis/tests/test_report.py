"""Tests for the world report's censorCityFunds variant.

Censoring is the default, so the tests that want the balance reported pass
censor_city_funds=False explicitly. The two section builders are exercised
directly rather than through gen_world_report, which needs a run on disk and
writes a report file.
"""

import micropolis_world.module_globals as g
from micropolis_world.report import _history_section, _snapshot_section

FUNDS_LABEL = g.METRIC_LABELS[g.FUNDS_METRIC]
OTHER_METRICS = [m for m in g.METRICS if m != g.FUNDS_METRIC]


def make_row(turn: int, funds: int = 4242) -> dict:
    row = {
        "tick": turn * g.TICKS_PER_TURN,
        "gameLevel": 1,
        "autoBudget": 1,
        "cityClass": 2,
        "cashFlow": -536,
        "cityTax": 7,
        "roadEffect": 32,
        "policeEffect": 1000,
        "fireEffect": 1000,
    }
    row.update({m: 100 + i for i, m in enumerate(g.METRICS)})
    row[g.FUNDS_METRIC] = funds
    row.update({key: 3 for key, _ in g.SNAPSHOT_INFRASTRUCTURE})
    row["census"] = {"rubble": 53, "fire": 0, "road": 127}
    return row


# Funds rise by one per turn, so a leaked value is recognizable by turn.
LOG = [make_row(t, funds=1000 + t) for t in range(49)]


class TestSnapshotSection:
    def test_reports_the_money_lines_when_not_censoring(self):
        text = "\n".join(_snapshot_section(make_row(48), censor_city_funds=False))
        assert f"{FUNDS_LABEL}: 4242" in text
        assert "Cash flow: -536" in text
        assert "Auto budget: On" in text

    def test_censoring_is_the_default_and_drops_funds_cash_flow_and_budget(self):
        text = "\n".join(_snapshot_section(make_row(48)))
        assert FUNDS_LABEL not in text
        assert "4242" not in text
        assert "Cash flow" not in text
        assert "-536" not in text
        assert "budget" not in text.lower()

    def test_censoring_keeps_the_other_metrics(self):
        row = make_row(48)
        text = "\n".join(_snapshot_section(row))
        for metric in OTHER_METRICS:
            assert f"{g.METRIC_LABELS[metric]}: {row[metric]}" in text
        assert "Tax rate: 7%" in text
        assert "Difficulty: Medium" in text

    def test_census_shown_by_default(self):
        text = "\n".join(_snapshot_section(make_row(48)))
        assert (
            "Ground survey — rubble tiles 53, tiles on fire 0, road tiles 127" in text
        )

    def test_report_census_false_omits_the_tile_counts(self):
        text = "\n".join(_snapshot_section(make_row(48), report_census=False))
        assert "Ground survey" not in text
        assert "rubble" not in text


class TestHistorySection:
    def test_reports_a_funds_column_when_not_censoring(self):
        lines = _history_section(LOG, turn=48, history_freq=24, censor_city_funds=False)
        expected = ["Turn"] + [g.METRIC_LABELS[m] for m in g.METRICS]
        assert lines[2].split(",") == expected
        assert all(len(line.split(",")) == len(expected) for line in lines[3:])

    def test_censoring_is_the_default_and_drops_the_funds_column(self):
        lines = _history_section(LOG, turn=48, history_freq=24)
        expected = ["Turn"] + [g.METRIC_LABELS[m] for m in OTHER_METRICS]
        assert lines[2].split(",") == expected
        assert all(len(line.split(",")) == len(expected) for line in lines[3:])
        table = "\n".join(lines)
        assert FUNDS_LABEL not in table
        # The sampled turns are 0, 24 and 48, whose funds are 1000, 1024, 1048.
        assert "1024" not in table
