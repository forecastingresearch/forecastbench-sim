"""Unit tests for the binary question resolver and corpus builder.

Everything runs on synthetic log/event rows except TestBuildCorpusBinary,
which builds a one-city corpus from invented cached rows, like
test_continuous_batching's questions sort tests.
"""

import pytest

import micropolis_world.module_globals as g
from micropolis_world.binary_questions import (
    MSG_BLACKOUTS,
    MSG_EARTHQUAKE,
    MSG_FIRE,
    MSG_FLOOD,
    MSG_HELICOPTER_CRASH,
    MSG_MELTDOWN,
    MSG_MONSTER,
    MSG_PLANE_CRASH,
    MSG_SHIPWRECK,
    MSG_TORNADO,
    MSG_TRAIN_CRASH,
    QUESTION_IDS,
    QUESTIONS,
    Message,
    RunIndex,
    build_corpus_binary,
    check_horizons,
    check_structural_constraints,
    resolve_all,
    yearly_checkpoints,
)
from micropolis_world.city_sim import CitySimulation
from micropolis_world.scenarios import get_base_scenarios


def make_row(
    turn: int,
    pop: int = 1000,
    cls: int = 2,
    score: int = 500,
    poll: int = 30,
    rubble: int = 10,
    fire: int = 0,
    road: int = 100,
) -> dict:
    return {
        "tick": turn * g.TICKS_PER_TURN,
        "cityPop": pop,
        "cityClass": cls,
        "cityScore": score,
        "pollutionAverage": poll,
        "census": {"rubble": rubble, "fire": fire, "road": road},
    }


def make_log(nturns: int, **kwargs) -> list[dict]:
    return [make_row(t, **kwargs) for t in range(nturns)]


def event(turn: int, message: Message, text: str | None = None) -> dict:
    return {
        "tick": turn * g.TICKS_PER_TURN,
        "event": "sendMessage",
        "messageNum": message.num,
        "messageText": text if text is not None else message.text,
    }


def make_sim(log_data: list[dict], events_data: list[dict]) -> CitySimulation:
    sim = CitySimulation("bruce", seed=42, disasters=True)
    sim.log_data = log_data
    sim.events_data = events_data
    sim.nturns = len(log_data)
    return sim


def run_index(log_data: list[dict], msgs: list[tuple[int, Message]]) -> RunIndex:
    return RunIndex(log_data=log_data, msgs=[(t, m.num) for t, m in msgs])


NOW, H = 48, 96
LOG = make_log(H + 1)


class TestRunIndex:
    def test_events_indexed_by_turn(self):
        run = RunIndex.from_sim(
            make_sim(LOG, [event(60, MSG_EARTHQUAKE), event(70, MSG_TORNADO)])
        )
        assert run.msgs == [(60, MSG_EARTHQUAKE.num), (70, MSG_TORNADO.num)]

    def test_non_send_message_events_excluded(self):
        events = [
            {"tick": 60 * g.TICKS_PER_TURN, "event": "startEarthquake"},
            event(60, MSG_EARTHQUAKE),
        ]
        run = RunIndex.from_sim(make_sim(LOG, events))
        assert run.msgs == [(60, MSG_EARTHQUAKE.num)]

    def test_drift_guard_raises_on_wrong_text(self):
        with pytest.raises(ValueError, match="message table drift"):
            RunIndex.from_sim(
                make_sim(LOG, [event(60, MSG_EARTHQUAKE, text=MSG_FLOOD.text)])
            )

    def test_unknown_message_numbers_pass_the_guard(self):
        unknown = Message(35, "Anything at all")
        run = RunIndex.from_sim(make_sim(LOG, [event(60, unknown)]))
        assert run.msgs == [(60, unknown.num)]

    def test_msg_count_window_is_half_open(self):
        run = run_index(
            LOG,
            [
                (NOW, MSG_EARTHQUAKE),
                (NOW + 1, MSG_EARTHQUAKE),
                (H, MSG_EARTHQUAKE),
                (H + 1, MSG_EARTHQUAKE),
            ],
        )
        # (NOW, H]: the message at NOW is the report's, the one past H is
        # beyond the horizon; only the two inside count.
        assert run.msg_count(MSG_EARTHQUAKE, NOW, H) == 2


class TestYearlyCheckpoints:
    def test_multiples_of_48_in_half_open_window(self):
        assert list(yearly_checkpoints(48, 144)) == [96, 144]
        assert list(yearly_checkpoints(0, 48)) == [48]
        assert list(yearly_checkpoints(50, 95)) == []


def resolve(log=LOG, msgs=(), now=NOW, h=H):
    return resolve_all(run_index(log, list(msgs)), now, h)


class TestMessageQuestions:
    def test_no_messages_all_no(self):
        answers = resolve()
        for qid in [
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A11",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
        ]:
            assert answers[qid] is False

    def test_single_occurrence_questions(self):
        for message, qid in [
            (MSG_EARTHQUAKE, "A1"),
            (MSG_TORNADO, "A2"),
            (MSG_FLOOD, "A3"),
            (MSG_MONSTER, "A4"),
            (MSG_PLANE_CRASH, "A5"),
            (MSG_SHIPWRECK, "A6"),
            (MSG_BLACKOUTS, "A11"),
            (MSG_MELTDOWN, "B1"),
            (MSG_FIRE, "B4"),
            (MSG_TRAIN_CRASH, "B5"),
        ]:
            assert resolve(msgs=[(60, message)])[qid] is True, qid

    def test_two_or_more_questions(self):
        for message, qid in [
            (MSG_EARTHQUAKE, "B2"),
            (MSG_TORNADO, "B3"),
            (MSG_FLOOD, "B6"),
            (MSG_MONSTER, "B7"),
        ]:
            assert resolve(msgs=[(60, message)])[qid] is False, qid
            assert resolve(msgs=[(60, message), (70, message)])[qid] is True, qid

    def test_message_at_now_excluded_at_horizon_included(self):
        assert resolve(msgs=[(NOW, MSG_EARTHQUAKE)])["A1"] is False
        assert resolve(msgs=[(H, MSG_EARTHQUAKE)])["A1"] is True

    def test_helicopter_message_does_not_count_as_plane_crash(self):
        # Message 27 fires alongside 24 for the same collision; only 24 counts.
        crash = [(60, MSG_PLANE_CRASH), (60, MSG_HELICOPTER_CRASH)]
        assert resolve(msgs=crash)["A5"] is True
        assert resolve(msgs=[(60, MSG_HELICOPTER_CRASH)])["A5"] is False


def log_with(turn: int, **kwargs) -> list[dict]:
    log = make_log(H + 1)
    log[turn] = make_row(turn, **kwargs)
    return log


class TestStateQuestions:
    def test_a7_population_strictly_lower(self):
        assert resolve(log_with(H, pop=999))["A7"] is True
        assert resolve(log_with(H, pop=1000))["A7"] is False

    def test_a8_less_than_half(self):
        assert resolve(log_with(H, pop=499))["A8"] is True
        assert resolve(log_with(H, pop=500))["A8"] is False

    def test_a10_and_b9_class_ordinals(self):
        assert resolve(log_with(H, cls=1))["A10"] is True
        assert resolve(log_with(H, cls=3))["A10"] is False
        assert resolve(log_with(H, cls=3))["B9"] is True
        assert resolve(log_with(H, cls=1))["B9"] is False
        answers = resolve()
        assert answers["A10"] is False and answers["B9"] is False

    def test_a12_pollution_exceeds_60(self):
        assert resolve(log_with(H, poll=60))["A12"] is False
        assert resolve(log_with(H, poll=61))["A12"] is True

    def test_a13_rubble_growth_boundary(self):
        assert resolve(log_with(H, rubble=59))["A13"] is False
        assert resolve(log_with(H, rubble=60))["A13"] is True

    def test_a14_any_tile_burning(self):
        assert resolve(log_with(H, fire=1))["A14"] is True
        assert resolve()["A14"] is False

    def test_a15_fewer_road_tiles(self):
        assert resolve(log_with(H, road=99))["A15"] is True
        assert resolve(log_with(H, road=100))["A15"] is False

    def test_a16_score_strictly_higher(self):
        assert resolve(log_with(H, score=501))["A16"] is True
        assert resolve(log_with(H, score=500))["A16"] is False


class TestCheckpointQuestions:
    def test_a9_zero_at_a_checkpoint(self):
        assert resolve(log_with(H, pop=0))["A9"] is True

    def test_a9_ignores_mid_year_zero(self):
        # cityPop is only recomputed at yearly evaluations, so a mid-year zero
        # is not scanned; checkpoints are the multiples of 48.
        assert resolve(log_with(70, pop=0))["A9"] is False

    def test_b8_new_all_time_high(self):
        log = make_log(145)
        log[96] = make_row(96, pop=2000)
        assert resolve(log, now=48, h=144)["B8"] is True

    def test_b8_blocked_by_pre_snapshot_high(self):
        log = make_log(145)
        log[48] = make_row(48, pop=3000)
        log[96] = make_row(96, pop=2000)
        assert resolve(log, now=48, h=144)["B8"] is False


class TestResolveAllValidation:
    def test_horizon_beyond_the_log_raises(self):
        with pytest.raises(ValueError, match="logged turns"):
            resolve(h=len(LOG))

    def test_now_before_the_first_checkpoint_raises(self):
        with pytest.raises(ValueError, match="now must be"):
            resolve(now=47, h=96)

    def test_window_without_a_checkpoint_raises(self):
        # (48, 95] holds no multiple of 48, so B8 would have nothing to max over.
        with pytest.raises(ValueError, match="no yearly checkpoint"):
            resolve(now=48, h=95)
        resolve(now=48, h=96)

    def test_check_horizons(self):
        check_horizons([48, 240])
        with pytest.raises(ValueError, match=r"got \[47\]"):
            check_horizons([48, 47])


class TestStructuralConstraints:
    def test_impossible_yes_raises(self):
        answers = dict.fromkeys(QUESTION_IDS, False)
        check_structural_constraints("kowloon", answers)
        for city, qid in [
            ("kowloon", "A3"),
            ("kowloon", "B6"),
            ("kobe", "B1"),
            ("bruce", "A5"),
        ]:
            violated = {**answers, qid: True}
            with pytest.raises(ValueError, match=qid):
                check_structural_constraints(city, violated)

    def test_possible_yes_passes(self):
        answers = dict.fromkeys(QUESTION_IDS, False)
        # badnews has 8 nuclear plants and an airport; only floods are
        # impossible there.
        check_structural_constraints("badnews", {**answers, "B1": True, "A5": True})


class TestQuestions:
    def test_all_25_in_doc_order(self):
        assert QUESTION_IDS == [f"A{i}" for i in range(1, 17)] + [
            f"B{i}" for i in range(1, 10)
        ]

    def test_every_text_carries_the_horizon_placeholder(self):
        for q in QUESTIONS:
            assert "{HORIZON}" in q.text, q.qid

    def test_question_resolve_matches_resolve_all(self):
        run = run_index(log_with(H, pop=0), [(60, MSG_EARTHQUAKE)])
        answers = resolve_all(run, NOW, H)
        for q in QUESTIONS:
            assert q.resolve(run, NOW, H) is answers[q.qid], q.qid


@pytest.mark.usefixtures("synthetic_corpus")
class TestBuildCorpusBinary:
    def test_one_city_corpus(self):
        scenarios = get_base_scenarios(seed=42, cities=["bruce"], disasters=[True])
        corpus = build_corpus_binary(
            scenarios,
            snapshot_turns=[96],
            horizons=[48],
            history_freq=24,
            label="test-binary",
        )
        assert len(corpus) == len(QUESTION_IDS)
        assert [c["qid"] for c in corpus] == QUESTION_IDS
        for c in corpus:
            assert isinstance(c["answer"], bool)
            assert c["question_id"] == f"bruce_disasters_seed42_T96_H48_{c['qid']}"
            assert c["snapshot_turn"] == 96
            assert c["horizon"] == 48
            assert c["resolution_turn"] == 144
            assert "{HORIZON}" not in c["question_text"]
            assert c["scenario"] == {"name": "bruce", "seed": 42, "disasters": True}
        # {HORIZON} became the absolute resolution turn.
        assert "turn 144" in corpus[0]["question_text"]
        # All questions at one snapshot share one report.
        assert len({id(c["context"]) for c in corpus}) == 1
