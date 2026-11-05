"""Unit tests for the interventional-conditional draw on invented ground-truth lines."""

from collections import Counter

import pytest

from micropolis_world.binary_questions import QUESTION_IDS
from micropolis_world.intcond import (
    MAX_CELLS_PER_DOFILE,
    MAX_INTERVENTIONS_PER_A,
    block_of,
    candidate_cells,
    draw,
    weak_pairs,
)

N = 100
H = [144, 240]


def lines(yes: dict[str, int], scenario="kobe_disasters_seed42", S=960) -> list[dict]:
    """One line per horizon; question q resolves Yes in the first yes[q] continuations."""
    return [{
        "scenario_id": scenario, "city": scenario.split("_")[0], "snapshot_turn": S,
        "horizon": h, "n_continuations": N, "branch_seeds": [1, N],
        "yes_idx": {q: list(range(yes.get(q, 0))) for q in QUESTION_IDS},
    } for h in H]


def test_blocks():
    assert block_of("makeEarthquake", "A1") == "same-family"  # at least one earthquake
    assert block_of("makeEarthquake", "A2") == "null"  # tornado
    assert block_of("makeEarthquake", "A13") == "damage"  # rubble
    assert block_of("makeEarthquake", "A5") == "damage"  # plane crash: airport damage
    assert block_of("makeFire", "B4") == "same-family"
    assert block_of("makeFire", "A14") == "damage"  # burning tile at the horizon
    assert block_of("demolishAirport", "A5") == "infrastructure"
    assert block_of("demolishPowerPlants", "A6") == "infrastructure"  # unpowered seaport
    assert block_of("demolishPowerPlants", "A4") == "infrastructure"  # pollution -> monster
    assert block_of("demolishAirport", "A4") == "null"
    assert block_of("demolishAirport", "A1") == "null"
    with pytest.raises(KeyError):
        block_of("makeMonster", "A1")


def test_candidate_filters_and_delta():
    base = lines({"A13": 30, "A2": 50, "A11": 3, "A7": 50})
    do = lines({"A13": 90, "A2": 52, "A11": 60, "A7": 99})
    cells = {(c["qid"], c["horizon"]): c for c in candidate_cells(base, do, "makeEarthquake", H)}
    assert cells["A13", 144]["delta"] == pytest.approx(0.6)
    assert cells["A13", 144]["n_both"] == 30
    assert cells["A2", 240]["block"] == "null"
    assert ("A11", 144) not in cells  # P(A) = 0.03 is below the mid-range
    assert ("A7", 144) not in cells  # P(A|do X) = 0.99 is settled
    with pytest.raises(ValueError, match="seeds differ"):
        candidate_cells(base, [dict(l, n_continuations=50) for l in do], "makeEarthquake", H)


def test_weak_pairs_only_in_moving_blocks():
    base = lines({"A13": 50, "A2": 50})
    do = lines({"A13": 51, "A2": 50})
    cells = candidate_cells(base, do, "makeEarthquake", H)
    assert weak_pairs(cells) == {("damage", "makeEarthquake", "rubble")}


def test_draw_respects_caps_and_quota():
    cells = []
    for city in ("kobe", "kyoto", "bruce"):
        base = lines({q: 50 for q in QUESTION_IDS}, f"{city}_disasters_seed42")
        for action in ("makeEarthquake", "makeTornado", "makeFire", "demolishPowerPlants"):
            cells += candidate_cells(base, lines({q: 60 for q in QUESTION_IDS}, f"{city}_disasters_seed42"), action, H)
    quota = {"null": 10, "same-family": 6, "damage": 20, "infrastructure": 10}
    chosen, notes = draw(cells, 46, quota, seed=1)
    assert len(chosen) == 46, notes
    per_a = Counter((c["scenario_id"], c["horizon"], c["qid"]) for c in chosen)
    assert max(per_a.values()) <= MAX_INTERVENTIONS_PER_A
    per_do = Counter((c["scenario_id"], c["snapshot_turn"], c["action"]) for c in chosen)
    assert max(per_do.values()) <= MAX_CELLS_PER_DOFILE
    assert len({(c["scenario_id"], c["horizon"], c["qid"], c["action"]) for c in chosen}) == 46
    with pytest.raises(ValueError, match="sum"):
        draw(cells, 40, quota, seed=1)


def test_intcond_prompts():
    from micropolis_world.intcond_prompts import build_jobs, turn2_text

    cell = {"cell_id": 3, "scenario_id": "kobe_disasters_seed42", "snapshot_turn": 960, "horizon": 144, "qid": "A2",
            "action": "makeEarthquake", "block": "null", "question_id": "kobe_disasters_seed42_T960_H144_A2",
            "text": "Will at least one tornado be sighted between the current turn and turn 1104?"}
    t = turn2_text(cell)
    assert "an earthquake is set off" in t and "logged at turn 960" in t
    assert "between turn 960 and turn 1104? Only turns 961 through 1104 count." in t
    assert "logged" not in turn2_text(dict(cell, action="demolishAirport"))
    jobs = build_jobs([cell, dict(cell, cell_id=4, action="makeFire")], {("kobe_disasters_seed42", 960): "R"}, "P")
    assert len(jobs["t1"]) == 1 and [j["item"] for j in jobs["t2"]] == [
        "kobe_disasters_seed42_T960_H144_A2|do-makeEarthquake", "kobe_disasters_seed42_T960_H144_A2|do-makeFire"]
