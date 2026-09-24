"""Fabio: cached Micropolis trajectory conversion, explicit schema constants."""
from __future__ import annotations
from collections import defaultdict
from typing import Protocol

class CitySimulation(Protocol):
    log_data: list[dict] | None

def turn_of(row: dict, ticks_per_turn: int) -> int:
    """The turn a log or event row falls on."""
    return row["tick"] // ticks_per_turn

def to_world(sims: dict[str, CitySimulation], *, metrics: list[str], ticks_per_turn: int) -> dict:
    """Assemble game_data in the core TURN-MAJOR schema:
    time_series[metric][turn][region_id] = value.
    """
    ts = {m: defaultdict(dict) for m in metrics}
    for sim_id, (name, sim) in enumerate(sims.items()):
        if sim.log_data is None:
            raise ValueError(f"Simulation {name} has no log_data")
        for line in sim.log_data:
            turn = turn_of(line, ticks_per_turn)
            for metric in metrics:
                ts[metric][str(turn)][str(sim_id)] = line[metric]
    return {
        "time_series": ts,
        "civilizations": {
            str(sim_id): {"name": name, "nation_id": str(sim_id)}
            for sim_id, (name, _) in enumerate(sims.items())
        },
    }
