"""Prompts for the natural-conditional cells, after FreeCiv's elicit_natcond_v1.py.

Turn 1 asks question A alone (one question per prompt). Turn 2 continues that
conversation (turn-1 prompt, the model's own reply, then one message): fact B as
the one extra thing known, and A restated with explicit turns. Unlike FreeCiv's
"One fact about turns ... has been revealed", this cannot be read as a full
account of B's window. A question A in two cells is asked once at turn 1; both cells
continue from that reply. FreeCiv's no-news control is not repeated here.
The world report is the one the binary eval showed for that city and snapshot.
"""

from pathlib import Path

from .config import DATAFILES_DIR

PREAMBLE_PATH = DATAFILES_DIR / "preamble-natcond.txt"

TAIL = """## Question
{question}

Think step by step about the information provided, reason about uncertainty, and put your final confidence that the question resolves YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

Your final answer should be the probability that the event resolves YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

# Turn 2 states the fact as the one extra thing known and restates question A
# with explicit turns, so a model cannot read the fact as a full account of its
# window or lose track of A's own window.
TURN2 = """Suppose you now learn one more thing about the game after turn {S}: {fact} You learn nothing else about what happened after turn {S}.

Given this, what is the probability that this question resolves YES: {question}{window}

Reason about what, if anything, this changes, and end your response with your updated probability in <probability> </probability> tags."""
WINDOW = " Anything from turn {start} through turn {end} counts toward it, including turns {start}–{fact_end}."

# What each question's Yes outcome says once it has happened, past tense, to
# follow a colon. {S1} is the first turn after the snapshot S, {T} the fact's
# resolution turn.
STATEMENT = {
    "A1": "at least one earthquake was reported at some point between turn {S1} and turn {T}.",
    "A2": "at least one tornado was sighted at some point between turn {S1} and turn {T}.",
    "A3": "at least one flood was reported at some point between turn {S1} and turn {T}.",
    "A4": "a monster was sighted at some point between turn {S1} and turn {T}.",
    "A5": "an airplane crashed at some point between turn {S1} and turn {T}.",
    "A6": "a shipwreck was reported at some point between turn {S1} and turn {T}.",
    "A7": "the city's population at turn {T} was lower than at turn {S}.",
    "A8": "the city's population at turn {T} was less than half of its value at turn {S}.",
    "A9": "the city's population read zero at a yearly checkpoint between turn {S1} and turn {T}.",
    "A10": "the city's classification at turn {T} was lower than at turn {S}.",
    "A11": 'a "Blackouts reported" advisory appeared at some point between turn {S1} and turn {T}.',
    "A12": "the citywide average pollution level exceeded 60 at turn {T}.",
    "A13": "the map held at least 50 more rubble tiles at turn {T} than at turn {S}.",
    "A14": "at least one map tile was actively burning at turn {T}.",
    "A15": "the map contained fewer road tiles at turn {T} than at turn {S}.",
    "A16": "the city's evaluation score at turn {T} was higher than at turn {S}.",
    "B1": "a nuclear meltdown occurred at some point between turn {S1} and turn {T}.",
    "B2": "two or more earthquakes were reported between turn {S1} and turn {T}.",
    "B3": "two or more tornadoes were sighted between turn {S1} and turn {T}.",
    "B4": 'a "Fire reported!" disaster struck at some point between turn {S1} and turn {T}.',
    "B5": "a train crashed at some point between turn {S1} and turn {T}.",
    "B6": "two or more separate floods were reported between turn {S1} and turn {T}.",
    "B7": "the monster was sighted two or more times between turn {S1} and turn {T}.",
    "B8": "the city's population reached a new all-time high at a yearly checkpoint between turn {S1} and turn {T}.",
    "B9": "the city's classification at turn {T} was higher than at turn {S}.",
}

# Question text -> the same question with explicit turns ({S} the snapshot).
EXPLICIT = [
    ("between the current turn and turn", "between turn {S} and turn"),
    ("than it is at the current turn", "than it was at turn {S}"),
    ("less than half of its current value", "less than half of its value at turn {S}"),
    ("than it is now", "than it was at turn {S}"),
    ("than it does now", "than it did at turn {S}"),
]


def explicit_question(text: str, S: int) -> str:
    """A question text with "the current turn" and "now" replaced by turn S."""
    for old, new in EXPLICIT:
        text = text.replace(old, new.format(S=S))
    if "current" in text or " now" in text:
        raise ValueError(f"question still refers to the present: {text!r}")
    return text

REPORT_START, REPORT_END = "## Game report\n", "## Questions\n"


def report_from_prompt(text: str) -> str:
    """The "## Game report" section of a binary-eval prompt, heading included."""
    i, j = text.find(REPORT_START), text.find(REPORT_END)
    if i < 0 or j < i:
        raise ValueError("prompt has no '## Game report' ... '## Questions' section")
    return text[i:j]


def binary_prompt_path(cache_dir: Path, scenario_id: str, snapshot_turn: int) -> Path:
    """The binary eval's single cached prompt for one (scenario, snapshot)."""
    found = sorted((cache_dir / f"{scenario_id}_T{snapshot_turn}").glob("prompt-*.txt"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one prompt in {cache_dir / f'{scenario_id}_T{snapshot_turn}'}, found {len(found)}")
    return found[0]


def fact_text(cell: dict) -> str:
    S, T = cell["snapshot_turn"], cell["snapshot_turn"] + cell["horizon_b"]
    return STATEMENT[cell["qid_b"]].format(S=S, S1=S + 1, T=T)


def turn2_text(cell: dict) -> str:
    """The turn-2 message: the fact, then question A restated with explicit turns."""
    S = cell["snapshot_turn"]
    question = explicit_question(cell["text_a"], S)
    window = ""
    if f"between turn {S} and turn" in question:
        window = WINDOW.format(start=S + 1, end=S + cell["horizon_a"], fact_end=S + cell["horizon_b"])
    return TURN2.format(S=S, fact=fact_text(cell), question=question, window=window)


def build_jobs(cells: list[dict], reports: dict[tuple[str, int], str], preamble: str) -> dict[str, list[dict]]:
    """Turn-1 prompts (one per question A) and turn-2 follow-ups (one per cell).
    A prompt is split into prefix (preamble and report, shared by a snapshot's
    prompts) and tail."""
    t1, t2 = [], []
    seen = set()
    for c in cells:
        prefix = preamble.rstrip("\n") + "\n\n\n" + reports[c["scenario_id"], c["snapshot_turn"]]
        qa = c["question_id_a"]
        base = {"question_id": qa, "scenario_id": c["scenario_id"], "snapshot_turn": c["snapshot_turn"]}
        if qa not in seen:
            seen.add(qa)
            t1.append({**base, "arm": "t1", "item": qa, "prefix": prefix, "tail": TAIL.format(question=c["text_a"])})
        t2.append({**base, "arm": "t2", "item": f"{qa}|{c['question_id_b']}", "cell_id": c["cell_id"],
                   "block": c["block"], "followup": turn2_text(c)})
    return {"t1": t1, "t2": t2}


def conversation(t1_job: dict, reply: str | None = None, followup: dict | None = None) -> list[dict]:
    """The messages of turn 1, or of turn 2 given the model's turn-1 reply."""
    msgs = [{"role": "user", "content": t1_job["prefix"] + t1_job["tail"]}]
    if followup is not None:
        msgs += [{"role": "assistant", "content": reply or ""}, {"role": "user", "content": followup["followup"]}]
    return msgs
