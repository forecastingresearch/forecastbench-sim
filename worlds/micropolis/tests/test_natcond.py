"""Unit tests for the natural-conditional draw (micropolis_world/natcond.py)."""

import pytest

from micropolis_world.binary_questions import QUESTION_IDS
from micropolis_world.natcond import (
    FAMILY,
    block_of,
    candidate_cells,
    draw,
    finish,
)

N = 1000
H = [144, 240]


def lines(yes: dict[int, dict[str, range]], sid="kyoto_disasters_seed42", S=960) -> list[dict]:
    """Ground-truth lines with the given Yes continuations; other questions never Yes."""
    return [
        {
            "scenario_id": sid, "city": sid.split("_")[0], "snapshot_turn": S, "horizon": h,
            "n_continuations": N,
            "yes_idx": {q: list(yes.get(h, {}).get(q, ())) for q in QUESTION_IDS},
        }
        for h in H
    ]


def test_every_question_has_a_family():
    assert set(FAMILY) == set(QUESTION_IDS)


def test_blocks():
    assert block_of("A1", "B2") == "mechanical"  # one earthquake vs two or more
    assert block_of("A7", "A7") == "mechanical"  # same question, earlier horizon
    assert block_of("A3", "A5") == "irrelevant"  # flood vs airplane crash
    assert block_of("A13", "A1") == "inferential"  # earthquake revealed, rubble asked
    assert block_of("A10", "A7") == "inferential"  # population drop revealed, classification asked
    assert block_of("A1", "A13") is None  # rubble revealed, earthquake asked: reverse direction
    assert block_of("A12", "A3") is None  # pollution fits no block
    assert block_of("A14", "A1") is None  # burning: too weak to ask in the inferential block
    assert block_of("A14", "B4") == "mechanical"  # but still mechanical with its own family


def test_candidate_truth_and_requirements():
    # A13 at 240: Yes on 0..399 (P=.4). A1 at 144: Yes on 200..499 (n_b=300, n_ab=200).
    cells = candidate_cells(lines({240: {"A13": range(400)}, 144: {"A1": range(200, 500)}}), H)
    c = next(c for c in cells if (c["qid_a"], c["qid_b"]) == ("A13", "A1"))
    assert (c["horizon_a"], c["horizon_b"], c["block"]) == (240, 144, "inferential")
    assert (c["n_b"], c["n_ab"]) == (300, 200)
    assert c["p_a"] == pytest.approx(0.4) and c["p_a_given_b"] == pytest.approx(2 / 3)
    assert c["delta"] == pytest.approx(2 / 3 - 0.4)
    assert all(c["horizon_b"] < c["horizon_a"] for c in cells)


def test_max_gap_limits_horizon_pairs():
    h = [144, 240, 336]
    ls = [dict(line, horizon=x) for x, line in zip(h, lines({}) + lines({})[:1])]
    for line in ls:
        line["yes_idx"] = {q: list(range(250, 600)) if q in ("A13", "A1") else [] for q in QUESTION_IDS}
        line["yes_idx"]["A13"] = list(range(400))
    pairs = lambda cells: {(c["horizon_a"], c["horizon_b"]) for c in cells}
    assert pairs(candidate_cells(ls, h)) == {(240, 144), (336, 144), (336, 240)}
    assert pairs(candidate_cells(ls, h, max_gap=1)) == {(240, 144), (336, 240)}


@pytest.mark.parametrize(
    "a, b",
    [
        (range(40), range(300)),  # P(A) = .04: not mid-range
        (range(960), range(300)),  # P(A) = .96: not mid-range
        (range(400), range(500, 599)),  # B Yes in 99 < 100 replays
        (range(400), range(300)),  # P(A|B) = 1: the reveal settles A
        (range(400), range(400, 700)),  # P(A|B) = 0
    ],
)
def test_candidate_rejections(a, b):
    cells = candidate_cells(lines({240: {"A13": a}, 144: {"A1": b}}), H)
    assert not any((c["qid_a"], c["qid_b"]) == ("A13", "A1") for c in cells)


def pool(cities=6):
    """Admissible cells over several cities, many per A question and per revealed event."""
    cells = []
    for i in range(cities):
        yes = {240: {q: range(100 + 20 * k, 500 + 20 * k) for k, q in enumerate(["A13", "A15", "A7", "A1", "A3"])},
               144: {q: range(250, 600) for q in ["A1", "A2", "A3", "A5", "A13", "A15", "A7", "B4"]}}
        cells += candidate_cells(lines(yes, sid=f"city{i}_disasters_seed42"), H)
    return cells


def test_draw_caps_quota_and_determinism():
    cells = pool()
    quota = {"irrelevant": 6, "mechanical": 10, "inferential": 14}
    chosen, notes = draw(cells, 30, quota, seed=42)
    assert len(chosen) == 30 and not notes
    assert {b: sum(c["block"] == b for c in chosen) for b in quota} == quota
    per_a, per_b = {}, {}
    for c in chosen:
        per_a.setdefault((c["scenario_id"], c["horizon_a"], c["qid_a"]), []).append(c["qid_b"])
        k = (c["scenario_id"], c["horizon_b"], c["qid_b"])
        per_b[k] = per_b.get(k, 0) + 1
    assert all(len(v) <= 2 and len(set(v)) == len(v) for v in per_a.values())
    assert max(per_b.values()) <= 8
    again, _ = draw(cells, 30, quota, seed=42)
    assert again == chosen
    other, _ = draw(cells, 30, quota, seed=7)
    assert other != chosen


def test_draw_refills_and_reports_shortfall():
    cells = [c for c in pool(1) if c["block"] != "irrelevant"]
    chosen, notes = draw(cells, 6, {"irrelevant": 2, "mechanical": 2, "inferential": 2}, seed=1)
    assert len(chosen) == 6 and any("irrelevant" in n for n in notes)


def test_quota_must_sum_to_n():
    with pytest.raises(ValueError, match="sum"):
        draw(pool(1), 10, {"irrelevant": 1, "mechanical": 1, "inferential": 1}, seed=1)


def test_record():
    chosen, _ = draw(pool(), 20, {"irrelevant": 4, "mechanical": 8, "inferential": 8}, seed=3)
    r = finish(chosen[0], 0, 3, "abc")
    S, ha = r["snapshot_turn"], r["horizon_a"]
    assert r["question_id_a"] == f"{r['scenario_id']}_T{S}_H{ha}_{r['qid_a']}"
    assert str(S + ha) in r["text_a"] and "{HORIZON}" not in r["text_b"]
