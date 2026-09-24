"""A percentile set whose median cannot be a value of its metric is not a
forecast of it, and the dataset builder discards it like an unparseable one."""

from micropolis_world.continuous_eval import METRIC_RANGES, out_of_range


def pct(p50: float) -> dict[str, float]:
    return {"p10": p50 - 2, "p25": p50 - 1, "p50": p50, "p75": p50 + 1, "p90": p50 + 2}


def test_averages_are_bounded_by_the_engine_maps():
    assert out_of_range("pollutionAverage", pct(53)) is None
    assert out_of_range("pollutionAverage", pct(51_000_000)) is not None
    assert out_of_range("trafficAverage", pct(2200)) is not None
    assert out_of_range("trafficAverage", pct(600)) is None  # 255 * 2.4 = 612
    assert out_of_range("landValueAverage", pct(21_400)) is not None


def test_population_has_no_upper_bound_but_is_not_negative():
    assert out_of_range("cityPop", pct(5_000_000)) is None
    assert out_of_range("cityPop", pct(-10)) is not None


def test_only_the_median_is_checked():
    wide = {"p10": 0, "p25": 40, "p50": 100, "p75": 200, "p90": 300}
    assert out_of_range("crimeAverage", wide) is None


def test_unknown_metric_is_never_rejected():
    assert out_of_range("totalFunds", pct(1e12)) is None
    assert "totalFunds" not in METRIC_RANGES
