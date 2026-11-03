"""Natural-conditional draw from ground-truth replays, after FreeCiv's draw_v1.py.

A cell pairs question A (resolving at snapshot S + horizon_A) with a revealed
fact B (question B at S + horizon_B, horizon_B < horizon_A) in the same city;
its truth is P(A | B) over the continuations where B resolved Yes. The draw is
structural: blocks, template balance and caps, never the effect size.
"""

import random
from collections import Counter, defaultdict

from .binary_questions import QUESTIONS

# Event families: same family -> "mechanical".
FAMILY = {
    "A1": "earthquake", "B2": "earthquake",
    "A2": "tornado", "B3": "tornado",
    "A3": "flood", "B6": "flood",
    "A4": "monster", "B7": "monster",
    "B1": "meltdown",
    "A5": "air_crash",
    "A6": "shipwreck",
    "B5": "train_crash",
    "B4": "fire", "A14": "fire",
    "A7": "population", "A8": "population", "A9": "population", "B8": "population",
    "A10": "classification", "B9": "classification",
    "A11": "blackouts",
    "A12": "pollution",
    "A13": "rubble",
    "A15": "roads",
    "A16": "evaluation",
}
# Random disasters the engine triggers independently; two different ones -> "irrelevant".
INDEPENDENT_DISASTERS = {
    "earthquake", "tornado", "flood", "monster", "meltdown", "air_crash", "shipwreck", "train_crash",
}
# What disasters damage. A disaster revealed, then one of these asked, or two of these
# -> "inferential". The reverse (damage revealed, a disaster asked) carries no more
# signal than replay noise and fits no block.
CONSEQUENCES = {"fire", "rubble", "roads", "population", "classification", "evaluation", "blackouts"}
# Questions whose inferential pairings barely move (median |delta| <= 0.013 over the
# candidate pool): not asked in the inferential block.
WEAK_INFERENTIAL_A = {"B4", "A9", "B9", "A14"}
BLOCKS = ("irrelevant", "mechanical", "inferential")
DEFAULT_QUOTA = {"irrelevant": 50, "mechanical": 200, "inferential": 150}
# The reveal's horizon is the one just before the question's: closer reveals move P(A) more.
DEFAULT_MAX_GAP = 1

# Truth requirements, as FreeCiv: A mid-range (the bank's band), B Yes in >= 1/10 of
# the replays, and the reveal must not settle A.
P_A_RANGE = (0.05, 0.95)  # lo < P(A) <= hi
MIN_B_SHARE = 0.1
P_GIVEN_RANGE = (0.02, 0.98)  # lo < P(A|B) < hi
MAX_REVEALS_PER_A = 2
MAX_CELLS_PER_REVEAL = 8

TEXT = {q.qid: q.text for q in QUESTIONS}


def block_of(qid_a: str, qid_b: str) -> str | None:
    """The pair's block, or None when it fits none (e.g. pollution and a flood)."""
    fa, fb = FAMILY[qid_a], FAMILY[qid_b]
    if fa == fb:
        return "mechanical"
    if fa in INDEPENDENT_DISASTERS and fb in INDEPENDENT_DISASTERS:
        return "irrelevant"
    if fa in CONSEQUENCES and fb in INDEPENDENT_DISASTERS | CONSEQUENCES and qid_a not in WEAK_INFERENTIAL_A:
        return "inferential"
    return None


def candidate_cells(lines: list[dict], horizons: list[int], max_gap: int | None = None) -> list[dict]:
    """Admissible cells of one (scenario, snapshot) ground-truth file.

    horizon_B is earlier than horizon_A, and at most max_gap steps earlier in
    the sorted horizons when max_gap is set (1 = consecutive).
    """
    by_h = {line["horizon"]: line for line in lines}
    first = lines[0]
    n = first["n_continuations"]
    S = first["snapshot_turn"]
    horizons = sorted(horizons)
    yes = {h: {q: set(by_h[h]["yes_idx"][q]) for q in TEXT} for h in horizons}
    cells = []
    for i_a, h_a in enumerate(horizons):
        for i_b, h_b in enumerate(horizons):
            if i_b >= i_a or (max_gap is not None and i_a - i_b > max_gap):
                continue
            for qa in TEXT:
                ya = yes[h_a][qa]
                p_a = len(ya) / n
                if not P_A_RANGE[0] < p_a <= P_A_RANGE[1]:
                    continue
                for qb in TEXT:
                    block = block_of(qa, qb)
                    if block is None:
                        continue
                    yb = yes[h_b][qb]
                    if len(yb) < MIN_B_SHARE * n:
                        continue
                    n_ab = len(ya & yb)
                    p_given = n_ab / len(yb)
                    if not P_GIVEN_RANGE[0] < p_given < P_GIVEN_RANGE[1]:
                        continue
                    cells.append({
                        "scenario_id": first["scenario_id"], "city": first["city"], "snapshot_turn": S,
                        "horizon_a": h_a, "horizon_b": h_b, "qid_a": qa, "qid_b": qb,
                        "block": block,
                        "n": n, "n_a": len(ya), "n_b": len(yb), "n_ab": n_ab,
                        "p_a": p_a, "p_a_given_b": p_given, "delta": p_given - p_a,
                    })
    return cells


def _spread(total: int, keys: list, rng: random.Random) -> dict:
    """Split total over keys as evenly as possible; the remainder goes to random keys."""
    base, extra = divmod(total, len(keys))
    lucky = set(rng.sample(keys, extra))
    return {k: base + (k in lucky) for k in keys}


def draw(cells: list[dict], n: int, quota: dict[str, int], seed: int) -> tuple[list[dict], list[str]]:
    """Stratified draw: each block's quota spread over (snapshot, horizon_A) strata,
    template-balanced over qid_A inside each stratum and block, with FreeCiv's caps.
    Shortfalls are refilled from the other blocks of the same stratum. Returns the
    chosen cells and notes on any shortfall."""
    if sum(quota.values()) != n:
        raise ValueError(f"block quotas sum to {sum(quota.values())}, not {n}")
    rng = random.Random(seed)
    cells = sorted(cells, key=lambda c: (c["scenario_id"], c["snapshot_turn"], c["horizon_a"], c["horizon_b"], c["qid_a"], c["qid_b"]))
    strata = sorted({(c["snapshot_turn"], c["horizon_a"]) for c in cells})
    pools = defaultdict(lambda: defaultdict(list))  # (stratum, block) -> qid_a -> cells
    for c in cells:
        pools[(c["snapshot_turn"], c["horizon_a"]), c["block"]][c["qid_a"]].append(c)
    for by_q in pools.values():
        for lst in by_q.values():
            rng.shuffle(lst)

    chosen: list[dict] = []
    reveals_of_a = defaultdict(set)  # A question -> qid_b already revealed on it
    cells_of_b = Counter()  # revealed event -> cells using it
    city_count = Counter()

    def a_key(c):
        return (c["scenario_id"], c["snapshot_turn"], c["horizon_a"], c["qid_a"])

    def b_key(c):
        return (c["scenario_id"], c["snapshot_turn"], c["horizon_b"], c["qid_b"])

    def ok(c):
        r = reveals_of_a[a_key(c)]
        return len(r) < MAX_REVEALS_PER_A and c["qid_b"] not in r and cells_of_b[b_key(c)] < MAX_CELLS_PER_REVEAL

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
            # least-used city first, so no city dominates; ties keep the shuffled order
            k = min(fits, key=lambda k: city_count[lst[k]["city"]])
            c = lst.pop(k)
            chosen.append(c)
            reveals_of_a[a_key(c)].add(c["qid_b"])
            cells_of_b[b_key(c)] += 1
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
    """Output record: ids and question texts with the resolution turn filled in."""
    S = c["snapshot_turn"]
    out = {"cell_id": cell_id, **c}
    for side in ("a", "b"):
        h, q = c[f"horizon_{side}"], c[f"qid_{side}"]
        out[f"question_id_{side}"] = f"{c['scenario_id']}_T{S}_H{h}_{q}"
        out[f"text_{side}"] = TEXT[q].replace("{HORIZON}", str(S + h))
    out["seed"] = seed
    out["ground_truth_fbsim_commit"] = fbsim_commit
    return out
