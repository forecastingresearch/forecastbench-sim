"""Prompts for the interventional-conditional cells (intcond.py), after natcond_prompts.py.

Turn 1 is the natural-conditional turn 1: question A alone, same preamble and
report, so its replies can be reused. Turn 2 continues that conversation with
intervention X done once by hand right after the current turn (how
extract_ground_truth.py --intervene applies it), without saying where it
strikes, and A restated with explicit turns.
"""

from .natcond_prompts import TAIL, explicit_question

# What is done by hand, to follow a colon. Each matches the engine action:
# disasters start at a random spot; demolitions take every matching building.
DO_TEXT = {
    "makeEarthquake": "an earthquake is set off in the city.",
    "makeFire": "a fire is started at a random spot in the city.",
    "makeFlood": "a flood is started at a random spot along the city's waterfront.",
    "makeTornado": "a tornado is set loose at a random spot on the map.",
    "makeMeltdown": "a nuclear power plant in the city is made to melt down.",
    "demolishPowerPlants": "every power plant in the city, coal and nuclear, is demolished.",
    "demolishAirport": "every airport in the city is demolished.",
}
DISASTERS = {"makeEarthquake", "makeFire", "makeFlood", "makeTornado", "makeMeltdown"}

TURN2 = """Now suppose that one thing is done to the city by hand, once, right after the current turn (turn {S}) and before turn {S1} begins: {do} Nothing else is done by hand; afterwards the simulation again advances on its own.{logged}

Given this, what is the probability that this question resolves YES: {question}{window}

Reason about what, if anything, this changes, and end your response with your updated probability in <probability> </probability> tags."""
# The disaster's own report is logged at turn S, outside A's window.
LOGGED = (
    " The disaster is reported once, when it starts, at turn {S}; that report does not count toward the question"
    " below, and the disaster is not reported again while it lasts. Anything it leads to from turn {S1} on does count."
)
WINDOW = " Only turns {S1} through {T} count."


def turn2_text(cell: dict) -> str:
    """The turn-2 message: the intervention, then question A with explicit turns."""
    S, T = cell["snapshot_turn"], cell["snapshot_turn"] + cell["horizon"]
    question = explicit_question(cell["text"], S)
    window = WINDOW.format(S1=S + 1, T=T) if f"between turn {S} and turn" in question else ""
    logged = LOGGED.format(S=S, S1=S + 1) if cell["action"] in DISASTERS else ""
    return TURN2.format(S=S, S1=S + 1, do=DO_TEXT[cell["action"]], logged=logged, question=question, window=window)


def build_jobs(cells: list[dict], reports: dict[tuple[str, int], str], preamble: str) -> dict[str, list[dict]]:
    """Turn-1 prompts (one per question A, as natcond_prompts.build_jobs writes them)
    and turn-2 follow-ups (one per cell)."""
    t1, t2 = [], []
    seen = set()
    for c in cells:
        prefix = preamble.rstrip("\n") + "\n\n\n" + reports[c["scenario_id"], c["snapshot_turn"]]
        qa = c["question_id"]
        base = {"question_id": qa, "scenario_id": c["scenario_id"], "snapshot_turn": c["snapshot_turn"]}
        if qa not in seen:
            seen.add(qa)
            t1.append({**base, "arm": "t1", "item": qa, "prefix": prefix, "tail": TAIL.format(question=c["text"])})
        t2.append({**base, "arm": "t2", "item": f"{qa}|do-{c['action']}", "cell_id": c["cell_id"],
                   "block": c["block"], "action": c["action"], "followup": turn2_text(c)})
    return {"t1": t1, "t2": t2}
