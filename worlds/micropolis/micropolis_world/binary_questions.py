"""Binary yes/no question set + resolver, transcribing binary_forecasts.md.

Turn convention: this codebase's, not the doc's — state_at(T) = log_data[T]
and an event's turn is tick // 16 (city_sim.turn_of), a uniform one-turn
relabel of the doc's §4 reference resolver. Windows stay half-open (NOW, H].
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from . import module_globals as g
from .city_sim import CitySimulation, turn_of
from .report import gen_world_report


class Message(NamedTuple):
    """One sendMessage kind, pairing the engine's number with its text.

    Resolutions reference these constants rather than bare numbers; the text
    doubles as the §2.3 drift guard, checked in RunIndex.from_sim.
    """

    num: int
    text: str


MSG_BLACKOUTS = Message(15, "Blackouts reported. Check power map.")
MSG_FIRE = Message(20, "Fire reported!")
MSG_MONSTER = Message(21, "A monster has been sighted!")
MSG_TORNADO = Message(22, "Tornado reported!")
MSG_EARTHQUAKE = Message(23, "Major earthquake reported!")
MSG_PLANE_CRASH = Message(24, "A plane has crashed!")
MSG_SHIPWRECK = Message(25, "Shipwreck reported!")
MSG_TRAIN_CRASH = Message(26, "A train crashed!")
# Fires alongside MSG_PLANE_CRASH for the same collision; guarded but counted
# by no question, so the crash is never counted twice.
MSG_HELICOPTER_CRASH = Message(27, "A helicopter crashed!")
MSG_FLOOD = Message(42, "Flooding reported!")
MSG_MELTDOWN = Message(43, "A Nuclear Meltdown has occurred!")

MESSAGES = (
    MSG_BLACKOUTS,
    MSG_FIRE,
    MSG_MONSTER,
    MSG_TORNADO,
    MSG_EARTHQUAKE,
    MSG_PLANE_CRASH,
    MSG_SHIPWRECK,
    MSG_TRAIN_CRASH,
    MSG_HELICOPTER_CRASH,
    MSG_FLOOD,
    MSG_MELTDOWN,
)

_TEXT_BY_NUM = {m.num: m.text for m in MESSAGES}


@dataclass
class RunIndex:
    """One run's stats rows and (turn, messageNum) message pairs."""

    log_data: list[dict]
    msgs: list[tuple[int, int]]

    @classmethod
    def from_sim(cls, sim: CitySimulation) -> "RunIndex":
        if sim.log_data is None or sim.events_data is None:
            raise ValueError(
                "Simulation data not loaded; call sim.load_from_disk() first"
            )
        return cls(
            log_data=sim.log_data,
            msgs=messages_from_events(sim.events_data, sim.get_id_str()),
        )

    def state_at(self, t: int) -> dict:
        return self.log_data[t]

    def msg_count(self, msg: Message, a: int, b: int) -> int:
        """Count of `msg` occurrences with turn in the half-open window (a, b]."""
        return sum(1 for t, num in self.msgs if num == msg.num and a < t <= b)


def messages_from_events(events_data: list[dict], run_id: str) -> list[tuple[int, int]]:
    """(turn, messageNum) for every sendMessage row, with the §2.3 drift guard.

    `run_id` names the run in the error. Shared by the file-based RunIndex and
    the continuation stream reader so both resolve against the same guard.
    """
    msgs = []
    for e in events_data:
        if e.get("event") != "sendMessage":
            continue
        num = e["messageNum"]
        expected = _TEXT_BY_NUM.get(num)
        if expected is not None and e.get("messageText") != expected:
            raise ValueError(
                f"{run_id}: messageNum {num} at tick {e['tick']} "
                f"reads {e.get('messageText')!r}, expected {expected!r} — "
                "message table drift, resolution would be wrong"
            )
        msgs.append((turn_of(e), num))
    return msgs


def yearly_checkpoints(a: int, b: int) -> range:
    """The yearly checkpoints (multiples of 48 turns) in (a, b]."""
    year = g.TURNS_PER_YEAR
    return range(year * (a // year + 1), b + 1, year)


@dataclass(frozen=True)
class Window:
    """One (now, h] resolution window over a run, the vocabulary resolutions use."""

    run: RunIndex
    now: int
    h: int

    def __post_init__(self):
        if not 0 <= self.now < self.h < len(self.run.log_data):
            raise ValueError(
                f"need 0 <= now < h < {len(self.run.log_data)} logged turns, "
                f"got now={self.now}, h={self.h}"
            )
        # B8's all-time-high baseline scans yc(0, now), which must be non-empty.
        if self.now < g.TURNS_PER_YEAR:
            raise ValueError(f"now must be >= {g.TURNS_PER_YEAR}, got {self.now}")
        # B8 also takes the max over yc(now, h); a window with no checkpoint
        # would crash there instead of resolving.
        if not yearly_checkpoints(self.now, self.h):
            raise ValueError(
                f"window ({self.now}, {self.h}] holds no yearly checkpoint; "
                f"horizons must be >= {g.TURNS_PER_YEAR} turns"
            )

    def n(self, msg: Message) -> int:
        """Count of `msg` occurrences in (now, h]."""
        return self.run.msg_count(msg, self.now, self.h)

    def at(self, t: int) -> dict:
        """The stats row at turn t."""
        return self.run.state_at(t)

    def pop(self, t: int) -> int:
        return self.at(t)["cityPop"]

    def yc(self, a: int, b: int) -> range:
        """The yearly checkpoints in (a, b]."""
        return yearly_checkpoints(a, b)


@dataclass(frozen=True)
class Question:
    """One binary question: its id, model-facing text and resolution criterion.

    `text` carries a "{HORIZON}" placeholder for the absolute resolution turn;
    `resolution` decides Yes/No over one Window.
    """

    qid: str
    text: str
    resolution: Callable[[Window], bool]

    def resolve(self, run: RunIndex, now: int, h: int) -> bool:
        return bool(self.resolution(Window(run, now, h)))


# binary_forecasts.md §3, in doc order: A1-A16 are mid-range questions
# (target P(Yes) ≈ 10-90%), B1-B9 are tail-probability questions (target
# P(Yes) ≈ 0.5-5%). A10 carries the example the doc's wording note asks for;
# the class ordering table lives in the binary preamble.
QUESTIONS = [
    Question(
        "A1",
        "Will at least one earthquake be reported between the current turn "
        "and turn {HORIZON}?",
        lambda w: w.n(MSG_EARTHQUAKE) >= 1,
    ),
    Question(
        "A2",
        "Will at least one tornado be sighted between the current turn and "
        "turn {HORIZON}?",
        lambda w: w.n(MSG_TORNADO) >= 1,
    ),
    Question(
        "A3",
        "Will at least one flood be reported between the current turn and "
        "turn {HORIZON}?",
        lambda w: w.n(MSG_FLOOD) >= 1,
    ),
    Question(
        "A4",
        "Will a monster be sighted between the current turn and turn {HORIZON}?",
        lambda w: w.n(MSG_MONSTER) >= 1,
    ),
    Question(
        "A5",
        "Will an airplane crash between the current turn and turn {HORIZON}?",
        lambda w: w.n(MSG_PLANE_CRASH) >= 1,
    ),
    Question(
        "A6",
        "Will a shipwreck be reported between the current turn and turn {HORIZON}?",
        lambda w: w.n(MSG_SHIPWRECK) >= 1,
    ),
    Question(
        "A7",
        "Will the city's population at turn {HORIZON} be lower than it is at "
        "the current turn?",
        lambda w: w.pop(w.h) < w.pop(w.now),
    ),
    Question(
        "A8",
        "Will the city's population at turn {HORIZON} be less than half of "
        "its current value?",
        lambda w: w.pop(w.h) < 0.5 * w.pop(w.now),
    ),
    Question(
        "A9",
        "Will the city's population read zero at any yearly checkpoint "
        "between the current turn and turn {HORIZON}?",
        lambda w: any(w.pop(t) == 0 for t in w.yc(w.now, w.h)),
    ),
    Question(
        "A10",
        "Will the city's classification at turn {HORIZON} be lower than it "
        "is now (e.g. Metropolis → Capital)?",
        lambda w: w.at(w.h)["cityClass"] < w.at(w.now)["cityClass"],
    ),
    Question(
        "A11",
        'Will a "Blackouts reported" advisory appear between the current '
        "turn and turn {HORIZON}?",
        lambda w: w.n(MSG_BLACKOUTS) >= 1,
    ),
    Question(
        "A12",
        "Will the citywide average pollution level exceed 60 at turn {HORIZON}?",
        lambda w: w.at(w.h)["pollutionAverage"] > 60,
    ),
    Question(
        "A13",
        "Will the map hold at least 50 more rubble tiles at turn {HORIZON} "
        "than it does now?",
        lambda w: w.at(w.h)["census"]["rubble"] - w.at(w.now)["census"]["rubble"] >= 50,
    ),
    Question(
        "A14",
        "Will at least one map tile be actively burning at turn {HORIZON}?",
        lambda w: w.at(w.h)["census"]["fire"] > 0,
    ),
    Question(
        "A15",
        "Will the map contain fewer road tiles at turn {HORIZON} than it does now?",
        lambda w: w.at(w.h)["census"]["road"] < w.at(w.now)["census"]["road"],
    ),
    Question(
        "A16",
        "Will the city's evaluation score at turn {HORIZON} be higher than it is now?",
        lambda w: w.at(w.h)["cityScore"] > w.at(w.now)["cityScore"],
    ),
    Question(
        "B1",
        "Will a nuclear meltdown occur between the current turn and turn {HORIZON}?",
        lambda w: w.n(MSG_MELTDOWN) >= 1,
    ),
    Question(
        "B2",
        "Will two or more earthquakes be reported between the current turn "
        "and turn {HORIZON}?",
        lambda w: w.n(MSG_EARTHQUAKE) >= 2,
    ),
    Question(
        "B3",
        "Will two or more tornadoes be sighted between the current turn and "
        "turn {HORIZON}?",
        lambda w: w.n(MSG_TORNADO) >= 2,
    ),
    Question(
        "B4",
        'Will a "Fire reported!" disaster strike between the current turn '
        "and turn {HORIZON}?",
        lambda w: w.n(MSG_FIRE) >= 1,
    ),
    Question(
        "B5",
        "Will a train crash between the current turn and turn {HORIZON}?",
        lambda w: w.n(MSG_TRAIN_CRASH) >= 1,
    ),
    Question(
        "B6",
        "Will two or more separate floods be reported between the current "
        "turn and turn {HORIZON}?",
        lambda w: w.n(MSG_FLOOD) >= 2,
    ),
    Question(
        "B7",
        "Will the monster be sighted two or more times between the current "
        "turn and turn {HORIZON}?",
        lambda w: w.n(MSG_MONSTER) >= 2,
    ),
    Question(
        "B8",
        "Will the city's population reach a new all-time high at any yearly "
        "checkpoint between the current turn and turn {HORIZON}?",
        lambda w: (
            max(w.pop(t) for t in w.yc(w.now, w.h))
            > max(w.pop(t) for t in w.yc(0, w.now))
        ),
    ),
    Question(
        "B9",
        "Will the city's classification at turn {HORIZON} be higher than it is now?",
        lambda w: w.at(w.h)["cityClass"] > w.at(w.now)["cityClass"],
    ),
]

QUESTION_IDS = [q.qid for q in QUESTIONS]

# binary_forecasts.md §5: (city, question) pairs the map makes impossible.
# Used as assertions on resolver output — a Yes here is a parsing bug, most
# likely a shifted message-number table.
NO_FLOOD_CITIES = {"badnews", "haight", "happisle", "kowloon", "linecity"}
NO_MELTDOWN_CITIES = {
    "kamakura",
    "kobe",
    "kowloon",
    "kyoto",
    "linecity",
    "ndulls",
    "radial",
    "southpac",
    "wetcity",
}
NO_AIRPORT_CITIES = {
    "bruce",
    "freds",
    "linecity",
    "med_isle",
    "ndulls",
    "radial",
    "senri",
    "southpac",
}


def check_horizons(horizons: list[int]) -> None:
    """Raise if a horizon is too short to hold a yearly checkpoint (B8).

    For callers to run before simulating anything, so a bad config fails at
    once rather than at the first Window built from it.
    """
    short = [h for h in horizons if h < g.TURNS_PER_YEAR]
    if short:
        raise ValueError(
            f"horizons must be >= {g.TURNS_PER_YEAR} turns so every window holds "
            f"a yearly checkpoint (B8); got {short}"
        )


def resolve_all(run: RunIndex, now: int, h: int) -> dict[str, bool]:
    """All 25 answers for the window (now, h] — binary_forecasts.md §4."""
    return {q.qid: q.resolve(run, now, h) for q in QUESTIONS}


def check_structural_constraints(city: str, answers: dict[str, bool]) -> None:
    """Raise if a §5-impossible question resolved Yes for `city`."""
    impossible = []
    if city in NO_FLOOD_CITIES:
        impossible += ["A3", "B6"]
    if city in NO_MELTDOWN_CITIES:
        impossible += ["B1"]
    if city in NO_AIRPORT_CITIES:
        impossible += ["A5"]
    violated = [qid for qid in impossible if answers[qid]]
    if violated:
        raise ValueError(
            f"{city}: structurally impossible question(s) resolved Yes: "
            f"{', '.join(violated)} — message table drift or an off-by-one "
            "in the resolver (binary_forecasts.md §5)"
        )


def build_corpus_binary(
    scenarios: list[CitySimulation],
    snapshot_turns: list[int],
    horizons: list[int],
    history_freq: int,
    label: str,
    snapshot_only_report: bool = False,
    history_length: int = -1,
    report_effectiveness: bool = False,
    censor_city_funds: bool = True,
    report_census: bool = False,
) -> list[dict]:
    """One question per (scenario, snapshot turn, horizon, question id).

    The binary counterpart of scenarios.build_corpus: same sim/report loop,
    with the local resolve_all in place of the fbsim-core resolver and a bool
    "answer" in place of the continuous "value". Entries are ordered all
    questions for one horizon before the next, so a batch prompt walks the
    question list once per horizon.
    """
    check_horizons(horizons)
    corpus = []
    nturns = max(snapshot_turns) + max(horizons) + 1
    for sim in scenarios:
        sim.run_if_needed_and_load(nturns=nturns, quiet=True)
        run = RunIndex.from_sim(sim)
        scenario_id = sim.get_id_str()
        for snapshot_turn in snapshot_turns:
            report_text = gen_world_report(
                sim,
                turn=snapshot_turn,
                history_freq=history_freq,
                label=label,
                snapshot_only=snapshot_only_report,
                history_length=history_length,
                report_effectiveness=report_effectiveness,
                censor_city_funds=censor_city_funds,
                report_census=report_census,
            )
            for horizon in horizons:
                resolution_turn = snapshot_turn + horizon
                answers = resolve_all(run, snapshot_turn, resolution_turn)
                check_structural_constraints(sim.city_name, answers)
                for q in QUESTIONS:
                    corpus.append(
                        {
                            "question_id": (
                                f"{scenario_id}_T{snapshot_turn}_H{horizon}_{q.qid}"
                            ),
                            "qid": q.qid,
                            "question_text": q.text.replace(
                                "{HORIZON}", str(resolution_turn)
                            ),
                            "snapshot_turn": snapshot_turn,
                            "horizon": horizon,
                            "resolution_turn": resolution_turn,
                            "scenario_id": scenario_id,
                            "scenario": sim.describe(),
                            "answer": answers[q.qid],
                            "context": report_text,
                        }
                    )
    print(
        f"\nbuild_corpus_binary: {len(corpus)} questions for "
        f"{len(scenarios)} scenarios "
        f"({len(corpus) / len(scenarios):g} questions per scenario)."
    )
    return corpus
