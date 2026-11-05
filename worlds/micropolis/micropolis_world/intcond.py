"""Interventional-conditional draw: P(A | do X) against P(A), after natcond.py.

A cell pairs question A (resolving over (S, S+h]) with intervention X applied at
snapshot S in every continuation, each with its own realisation
(extract_ground_truth.py --intervene). P(A) comes from the base ground truth and
P(A|do X) from the do-file; the two are effectively independent samples. Cells
are picked by blocks, template balance and caps, never by their own effect size,
but weak_pairs drops (block, X, A family) pairs by their median realised |delta|.
"""

import random
import statistics
from collections import Counter, defaultdict

from .natcond import CONSEQUENCES, FAMILY, P_A_RANGE, P_GIVEN_RANGE, TEXT, _spread

# Interventions in the study and their families. makeMonster is left out: the
# engine spawns the monster on a river tile, where it disappears without effect.
X_FAMILY = {
    "makeEarthquake": "earthquake",
    "makeFire": "fire",
    "makeFlood": "flood",
    "makeTornado": "tornado",
    "makeMeltdown": "meltdown",
    "demolishPowerPlants": "power",
    "demolishAirport": "airport",
}
DEMOLITIONS = {"power", "airport"}
# Disasters the engine draws at random, with no dependence on infrastructure.
RANDOM_DISASTERS = {"earthquake", "tornado", "flood", "monster", "meltdown"}
# Crashes need a powered airport, seaport or rail line, so damage reaches them.
CRASHES = {"air_crash", "shipwreck", "train_crash"}
OUTCOMES = CONSEQUENCES | {"pollution"} | CRASHES
# Same family as X but a state the disaster leaves behind, not a count of it:
# a tile still burning at the horizon after a forced fire is damage.
DAMAGE_NOT_SAME = {"A14"}

BLOCKS = ("null", "same-family", "damage", "infrastructure")
DEFAULT_QUOTA = {"null": 50, "same-family": 50, "damage": 200, "infrastructure": 100}
# (block, action, A family) pairs whose median |delta| over the pool is at most
# this are dropped, in the blocks meant to move only.
WEAK_DELTA = 0.013
WEAK_BLOCKS = {"damage", "infrastructure"}
MAX_INTERVENTIONS_PER_A = 2
MAX_CELLS_PER_DOFILE = 8


def block_of(action: str, qid: str) -> str | None:
    """The (X, A) pair's block, or None when it fits none."""
    xf, af = X_FAMILY[action], FAMILY[qid]
    if xf in DEMOLITIONS:
        # Power plants also feed pollution, hence the monster, and hold the nuclear plant.
        if af in OUTCOMES or (xf == "power" and af in {"meltdown", "monster"}):
            return "infrastructure"
        return "null" if af in RANDOM_DISASTERS else None
    if af == xf and qid not in DAMAGE_NOT_SAME:
        return "same-family"
    if af in RANDOM_DISASTERS:
        return "null"
    if af in OUTCOMES:
        return "damage"
    return None


def candidate_cells(base: list[dict], do: list[dict], action: str, horizons: list[int]) -> list[dict]:
    """Admissible cells of one (scenario, snapshot, X): base and do-file lines."""
    by_h = {line["horizon"]: line for line in base}
    do_by_h = {line["horizon"]: line for line in do}
    first = do[0]
    n = first["n_continuations"]
    if by_h[horizons[0]]["n_continuations"] != n or first["branch_seeds"] != by_h[horizons[0]]["branch_seeds"]:
        raise ValueError(f"{first['scenario_id']} T{first['snapshot_turn']}: base and do-{action} seeds differ")
    cells = []
    for h in horizons:
        for qid in TEXT:
            block = block_of(action, qid)
            if block is None:
                continue
            ya, yd = set(by_h[h]["yes_idx"][qid]), set(do_by_h[h]["yes_idx"][qid])
            p_a, p_do = len(ya) / n, len(yd) / n
            if not (P_A_RANGE[0] < p_a <= P_A_RANGE[1] and P_GIVEN_RANGE[0] < p_do < P_GIVEN_RANGE[1]):
                continue
            cells.append({
                "scenario_id": first["scenario_id"], "city": first["city"],
                "snapshot_turn": first["snapshot_turn"], "horizon": h, "qid": qid,
                "action": action, "block": block,
                "n": n, "n_a": len(ya), "n_do": len(yd), "n_both": len(ya & yd),
                "p_a": p_a, "p_a_do": p_do, "delta": p_do - p_a,
            })
    return cells


def weak_pairs(cells: list[dict]) -> set[tuple[str, str, str]]:
    """(block, action, A family) pairs in WEAK_BLOCKS with median |delta| <= WEAK_DELTA."""
    by = defaultdict(list)
    for c in cells:
        by[c["block"], c["action"], FAMILY[c["qid"]]].append(abs(c["delta"]))
    return {k for k, v in by.items() if k[0] in WEAK_BLOCKS and statistics.median(v) <= WEAK_DELTA}


def draw(cells: list[dict], n: int, quota: dict[str, int], seed: int) -> tuple[list[dict], list[str]]:
    """Each block's quota spread over (snapshot, horizon) strata, template-balanced over
    qid inside each stratum and block, least-used action then city first, with the caps.
    Shortfalls are refilled from the other blocks of the same stratum."""
    if sum(quota.values()) != n:
        raise ValueError(f"block quotas sum to {sum(quota.values())}, not {n}")
    rng = random.Random(seed)
    cells = sorted(cells, key=lambda c: (c["scenario_id"], c["snapshot_turn"], c["horizon"], c["qid"], c["action"]))
    strata = sorted({(c["snapshot_turn"], c["horizon"]) for c in cells})
    pools = defaultdict(lambda: defaultdict(list))  # (stratum, block) -> qid -> cells
    for c in cells:
        pools[(c["snapshot_turn"], c["horizon"]), c["block"]][c["qid"]].append(c)
    for by_q in pools.values():
        for lst in by_q.values():
            rng.shuffle(lst)

    chosen: list[dict] = []
    actions_of_a = defaultdict(set)  # A question -> actions already applied to it
    cells_of_do = Counter()  # (scenario, snapshot, action) -> cells
    action_count = defaultdict(Counter)  # block -> action -> cells
    city_count = Counter()

    def a_key(c):
        return (c["scenario_id"], c["snapshot_turn"], c["horizon"], c["qid"])

    def do_key(c):
        return (c["scenario_id"], c["snapshot_turn"], c["action"])

    def ok(c):
        r = actions_of_a[a_key(c)]
        return len(r) < MAX_INTERVENTIONS_PER_A and c["action"] not in r and cells_of_do[do_key(c)] < MAX_CELLS_PER_DOFILE

    def take(stratum, block, want):
        by_q = pools[stratum, block]
        templates = sorted(by_q)
        rng.shuffle(templates)
        got, i, stall = 0, 0, 0
        while got < want and templates and stall < len(templates):
            q = templates[i % len(templates)]
            i += 1
            lst = by_q[q]
            fits = [k for k, c in enumerate(lst) if ok(c)]
            if not fits:
                stall += 1
                continue
            k = min(fits, key=lambda k: (action_count[block][lst[k]["action"]], city_count[lst[k]["city"]]))
            c = lst.pop(k)
            chosen.append(c)
            actions_of_a[a_key(c)].add(c["action"])
            cells_of_do[do_key(c)] += 1
            action_count[block][c["action"]] += 1
            city_count[c["city"]] += 1
            got += 1
            stall = 0
        return got

    notes = []
    for block in BLOCKS:
        per = _spread(quota.get(block, 0), strata, rng)
        for stratum in strata:
            got = take(stratum, block, per[stratum])
            if got < per[stratum]:
                notes.append(f"S{stratum[0]} H{stratum[1]} {block}: {got}/{per[stratum]}")
                for other in BLOCKS:
                    if other != block and got < per[stratum]:
                        extra = take(stratum, other, per[stratum] - got)
                        got += extra
                        if extra:
                            notes.append(f"  refilled {extra} from {other}")
    if len(chosen) < n:
        notes.append(f"drew {len(chosen)} of {n} cells")
    return chosen, notes


def finish(c: dict, cell_id: int, seed: int, fbsim_commit: str) -> dict:
    """Output record: question id and text with the resolution turn filled in."""
    S, h = c["snapshot_turn"], c["horizon"]
    return {
        "cell_id": cell_id, **c,
        "question_id": f"{c['scenario_id']}_T{S}_H{h}_{c['qid']}",
        "text": TEXT[c["qid"]].replace("{HORIZON}", str(S + h)),
        "seed": seed,
        "ground_truth_fbsim_commit": fbsim_commit,
    }
