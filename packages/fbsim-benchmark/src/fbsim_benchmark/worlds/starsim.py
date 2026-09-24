"""Fabio: cached Starsim trajectory conversion."""
METRICS = ["cumulative_cases", "active_infections", "cumulative_deaths"]

def to_world(region_series: dict[int, dict], names: dict[int, str]) -> dict:
    """Assemble game_data in the core TURN-MAJOR schema:
    time_series[metric][day][region_id] = value.
    """
    ts = {m: {} for m in METRICS}
    for rid, series in region_series.items():
        for m in METRICS:
            for day, val in enumerate(series[m]):
                ts[m].setdefault(str(day), {})[str(rid)] = val
    return {
        "time_series": ts,
        "civilizations": {str(rid): {"name": names[rid], "nation_id": str(rid)}
                          for rid in region_series},
    }
