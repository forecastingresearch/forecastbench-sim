"""Unit tests for the continuation stream reader behind extract_ground_truth.py.

Everything runs on a synthetic stream shaped like run_continuations.js's
output; nothing here launches the engine.
"""

import json

import pytest

import micropolis_world.module_globals as g
from micropolis_world.binary_questions import (
    MSG_EARTHQUAKE,
    MSG_TORNADO,
    QUESTION_IDS,
    Message,
)
from micropolis_world.city_sim import CitySimulation
from micropolis_world.ground_truth import (
    StreamError,
    consume_stream,
    covers,
    cross_check_trunk,
    ground_truth_lines,
    load_truths,
    merge_shards,
    producer_command,
    seed_shards,
    write_lines,
)

CITY = "kyoto"
S = 96  # snapshot turn; >= one year so B8's baseline window is non-empty
H = 96  # producer horizon; each tested horizon spans a yearly checkpoint (B8)
SHORT, LONG = 48, 96
HORIZONS = [SHORT, LONG]


def row(turn: int, seed: int | None, pop: int = 1000) -> dict:
    """A stats line: tick 16T+15 like run_sim.js, every field the resolver reads."""
    return {
        "kind": "stats",
        "seed": seed,
        "tick": turn * g.TICKS_PER_TURN + 15,
        "cityPop": pop,
        "cityClass": 2,
        "cityScore": 500,
        "totalFunds": 20000,
        "trafficAverage": 10,
        "pollutionAverage": 30,
        "crimeAverage": 20,
        "landValueAverage": 40,
        "census": {"rubble": 10, "fire": 0, "road": 100},
    }


def message(turn: int, seed: int | None, msg: Message) -> dict:
    return {
        "kind": "event",
        "seed": seed,
        "tick": turn * g.TICKS_PER_TURN + 3,
        "event": "sendMessage",
        "messageNum": msg.num,
        "messageText": msg.text,
    }


def header(**overrides) -> dict:
    rec = {
        "kind": "header",
        "city": CITY,
        "trunkSeed": 42,
        "branchAt": S + 1,
        "horizon": H,
        "seedLo": 1,
        "seedHi": 2,
        "disasters": True,
        "trunkOutput": True,
    }
    rec.update(overrides)
    return rec


def trunk(pop: int = 1000) -> list[dict]:
    return (
        [{"kind": "begin", "seed": None, "run": "trunk"}]
        + [row(t, None, pop) for t in range(S + 1)]
        + [{"kind": "end", "seed": None, "run": "trunk"}]
    )


def continuation(
    seed: int, events: list[dict], pops: dict[int, int] | None = None
) -> list[dict]:
    pops = pops or {}
    lines = [{"kind": "begin", "seed": seed, "run": "continuation"}]
    for t in range(S + 1, S + H + 1):
        lines.append(row(t, seed, pops.get(t, 1000)))
        lines.extend(e for e in events if e["tick"] // g.TICKS_PER_TURN == t)
    lines.append({"kind": "end", "seed": seed, "run": "continuation"})
    return lines


def encode(records: list[dict]) -> list[bytes]:
    return [json.dumps(r).encode() for r in records]


def make_stream() -> list[bytes]:
    """Seed 1: earthquake in turn S+1, tornado at turn S+H. Seed 2: quiet, pop drops."""
    return encode(
        [header()]
        + trunk()
        + continuation(
            1, [message(S + 1, 1, MSG_EARTHQUAKE), message(S + H, 1, MSG_TORNADO)]
        )
        + continuation(2, [], pops={S + H: 400})
    )


def test_counts_per_horizon():
    result = consume_stream(make_stream(), CITY, True, S, HORIZONS)
    assert result.seeds == [1, 2]
    assert result.n == 2
    # The earthquake is inside both windows; the tornado only the long one.
    assert result.yes[SHORT]["A1"] == 1 and result.yes[LONG]["A1"] == 1
    assert result.yes[SHORT]["A2"] == 0 and result.yes[LONG]["A2"] == 1
    # Seed 2's population halves by S+H: A7 and A8 at the long horizon only.
    assert result.yes[SHORT]["A7"] == 0 and result.yes[LONG]["A7"] == 1
    assert result.yes[LONG]["A8"] == 1
    assert result.yes[LONG]["B2"] == 0


def test_values_per_horizon_in_seed_order():
    result = consume_stream(make_stream(), CITY, True, S, HORIZONS)
    assert set(result.values[LONG]) == set(g.METRICS)
    assert result.values[LONG]["cityPop"] == [1000, 400]
    assert result.values[SHORT]["cityPop"] == [1000, 1000]
    assert result.trunk_log[S]["tick"] == S * g.TICKS_PER_TURN + 15
    assert "kind" not in result.trunk_log[0] and "seed" not in result.trunk_log[0]


def test_trunk_boundary_message_counts_in_window():
    # The trunk's last tick, 16(S+1), is turn S+1 under turn_of and so inside
    # (S, S+h]: a message there must reach every continuation.
    boundary = dict(
        message(S + 1, None, MSG_EARTHQUAKE), tick=(S + 1) * g.TICKS_PER_TURN
    )
    records = [header()] + trunk()
    records.insert(-1, boundary)
    records += continuation(1, []) + continuation(2, [])
    result = consume_stream(encode(records), CITY, True, S, HORIZONS)
    assert result.yes[SHORT]["A1"] == 2


def test_header_mismatch_is_an_error():
    with pytest.raises(StreamError, match="branchAt"):
        consume_stream(encode([header(branchAt=S)]), CITY, True, S, HORIZONS)
    with pytest.raises(StreamError, match="horizon"):
        consume_stream(encode([header(horizon=5)]), CITY, True, S, HORIZONS)
    with pytest.raises(StreamError, match="city"):
        consume_stream(encode([header(city="bruce")]), CITY, True, S, HORIZONS)


def test_missing_trunk_and_short_streams_are_errors():
    records = [header(trunkOutput=False)]
    with pytest.raises(StreamError, match="trunkOutput"):
        consume_stream(encode(records), CITY, True, S, HORIZONS)
    records = [header()] + trunk() + continuation(1, [])
    with pytest.raises(StreamError, match="1 of 2"):
        consume_stream(encode(records), CITY, True, S, HORIZONS)
    short_trunk = [header()] + trunk()[:-2] + trunk()[-1:]
    with pytest.raises(StreamError, match="trunk"):
        consume_stream(encode(short_trunk), CITY, True, S, HORIZONS)


def test_message_drift_guard_applies_to_continuations():
    bad = dict(message(S + 2, 1, MSG_EARTHQUAKE), messageText="Tornado reported!")
    records = [header()] + trunk() + continuation(1, [bad]) + continuation(2, [])
    with pytest.raises(ValueError, match="drift"):
        consume_stream(encode(records), CITY, True, S, HORIZONS)


def test_seed_shards_cover_range_once():
    assert seed_shards(10, 1) == [(1, 10)]
    assert seed_shards(10, 3) == [(1, 4), (5, 7), (8, 10)]
    assert seed_shards(2, 8) == [(1, 1), (2, 2)]
    with pytest.raises(ValueError):
        seed_shards(0, 2)


def test_producer_command_branches_after_snapshot():
    cmd = producer_command("kyoto", 42, False, 960, 480, 1, 100)
    assert cmd[cmd.index("--branch-at") + 1] == "961"
    assert cmd[cmd.index("--branch-seeds") + 1] == "1-100"
    assert "--no-disasters" in cmd


def test_merge_shards_adds_counts_and_concatenates_values():
    a = consume_stream(make_stream(), CITY, True, S, HORIZONS)
    b = consume_stream(
        encode([header(seedLo=3, seedHi=3)] + trunk() + continuation(3, [])),
        CITY,
        True,
        S,
        HORIZONS,
    )
    merged = merge_shards([a, b])
    assert merged.seeds == [1, 2, 3]
    assert merged.yes[LONG]["A1"] == 1
    assert merged.values[LONG]["cityPop"] == [1000, 400, 1000]
    other_trunk = consume_stream(
        encode([header(seedLo=3, seedHi=3)] + trunk(pop=5) + continuation(3, [])),
        CITY,
        True,
        S,
        HORIZONS,
    )
    with pytest.raises(StreamError, match="trunk"):
        merge_shards([a, other_trunk])


def test_ground_truth_lines_and_covers():
    sim = CitySimulation(CITY, 42, disasters=True)
    merged = consume_stream(make_stream(), CITY, True, S, HORIZONS)
    lines = ground_truth_lines(sim, S, HORIZONS, merged, "abc", "def")
    assert [line["horizon"] for line in lines] == HORIZONS
    assert lines[1]["resolution_turn"] == S + LONG
    assert lines[1]["n_continuations"] == 2
    assert lines[1]["branch_seeds"] == [1, 2]
    assert list(lines[1]["counts"]) == QUESTION_IDS
    assert lines[1]["counts"]["A2"] == 1
    assert lines[1]["values"]["cityPop"] == [1000, 400]
    assert covers(lines, HORIZONS, 2)
    assert not covers(lines, HORIZONS, 3)
    assert not covers(lines, HORIZONS + [144], 2)
    assert covers(lines[1:], [LONG], 1)


def test_cross_check_trunk(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "RUNS_DIR", tmp_path)
    sim = CitySimulation(CITY, 42, disasters=True)
    result = consume_stream(make_stream(), CITY, True, S, HORIZONS)
    assert "no cached run" in cross_check_trunk(result.trunk_log, sim)

    path = sim.get_data_file_path("log")
    path.parent.mkdir(parents=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in result.trunk_log[: S // 2])
    assert "ends there" in cross_check_trunk(result.trunk_log, sim)

    with open(path, "w") as f:
        f.writelines(
            json.dumps(dict(r, cityPop=999) if t == 5 else r) + "\n"
            for t, r in enumerate(result.trunk_log)
        )
    with pytest.raises(RuntimeError, match="turn 5, field 'cityPop'"):
        cross_check_trunk(result.trunk_log, sim)


def test_load_truths(tmp_path, monkeypatch):
    import micropolis_world.ground_truth as gt

    monkeypatch.setattr(gt, "out_dir", lambda: tmp_path)
    sim = CitySimulation(CITY, 42, disasters=True)
    merged = consume_stream(make_stream(), CITY, True, S, HORIZONS)
    write_lines(
        gt.output_path(sim, S), ground_truth_lines(sim, S, HORIZONS, merged, "x", "y")
    )
    corpus = [
        {
            "question_id": f"q{qid}{h}",
            "qid": qid,
            "scenario_id": sim.get_id_str(),
            "snapshot_turn": S,
            "horizon": h,
        }
        for qid in ("A1", "B1")
        for h in HORIZONS
    ]
    truths = load_truths(corpus)
    assert set(truths) == {c["question_id"] for c in corpus}
    for c in corpus:
        truth = truths[c["question_id"]]
        assert truth.n == merged.n
        assert truth.p == merged.yes[c["horizon"]][c["qid"]] / merged.n

    with pytest.raises(FileNotFoundError, match="no horizon 7"):
        load_truths([dict(corpus[0], horizon=7)])
    with pytest.raises(FileNotFoundError, match="extract_ground_truth"):
        load_truths([dict(corpus[0], snapshot_turn=S + 1)])
