#!/usr/bin/env python3
"""Curate the natural-conditional benchmark: horizon-balanced, effect-weighted.

Design (approved 2026-08-25):
  - Horizon balance is PRIMARY: equal n per horizon present (H1..H5).
  - Within a horizon, sample with a mild preference for larger |delta| —
    weight proportional to |delta| — so the full range stays represented but
    the median sits above the pool's, avoiding a set dominated by
    near-threshold cells that mostly measure noise.
  - Diversity caps still apply (per world, per world-event, per template) so no
    single world or cascade dominates.
  - Identity pairs (reveal names the question's own subject+target) excluded.
  - Matched placebos: same count, stratified by world x horizon.

Deterministic: selection uses a seeded RNG over the weighted order.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import Counter, defaultdict

# caps apply WITHIN each horizon stratum (not globally), so long horizons
# are not starved by budget consumed at H1.
MAX_PER_WORLD = 26
MAX_PER_EVENT = 8
MAX_PER_TEMPLATE = 40

def is_quasi(c):
    q, e = c["question"].lower(), c["event_desc"].lower()
    m = re.search(r"(discovered|changed government.*to|adopt\w*)\s+([a-z' ]+)$", e)
    if not m: return False
    return (m.group(2).strip() in q and e.split()[0] in q
            and c.get("half_b", {}).get("p_yx") in (0.0, 1.0))

def malformed(c):
    """Reject cells whose revealed event text is not human-meaningful (e.g. a
    raw tech id that never got a name, or barbarian/pirate pseudo-actors)."""
    e = c.get("event_desc", "")
    if re.search(r"#\d+|None|null|Unknown", e, re.I): return True
    return e.split()[0] in ("Barbarian", "Pirate") if e else True


def tfam(q):
    ql = q.lower()
    for k in ("technolog","wonder","government","rank","score","cit","population","territor","tile","treasur"):
        if k in ql: return k
    return "other"

def dA(c):
    """Selection weight uses half A ONLY. Half B is the scoring truth; selecting
    (or banding) on it lets half-B noise into the target and can bias which
    forecasters look good — band membership shifted up to 2x in testing."""
    return abs(c.get("half_a", {}).get("delta", 0.0))


def dB(c):
    """Half-B |delta| — REPORTING ONLY, never selection."""
    return abs(c.get("half_b", {}).get("delta", 0.0))

def pick(pool, per_horizon, seed=11):
    rng = random.Random(seed)
    chosen = []
    by_h = defaultdict(list)
    for c in pool: by_h[c.get("horizon","H1")].append(c)
    for hz in sorted(by_h):
        cand = by_h[hz]
        w_ct, e_ct, t_ct = Counter(), Counter(), Counter()   # reset per horizon
        # weighted shuffle: weight ~ |delta| (floored so small effects still appear)
        keyed = sorted(cand, key=lambda c: -(dA(c) + 0.02) * rng.random())
        n = 0
        for c in keyed:
            if n >= per_horizon: break
            w, e, t = c["game_id"], (c["game_id"], c["event_id"]), tfam(c["question"])
            if w_ct[w] >= MAX_PER_WORLD or e_ct[e] >= MAX_PER_EVENT or t_ct[t] >= MAX_PER_TEMPLATE:
                continue
            chosen.append(c); w_ct[w]+=1; e_ct[e]+=1; t_ct[t]+=1; n+=1
    return chosen

def matched_placebos(cells, chosen, target=None):
    """Placebos stratified to mirror the effect set's world x horizon shape.
    `target` optionally downsamples (proportionally) — selectivity needs fewer
    cells than the dose-response curve does."""
    strata = Counter((c["game_id"], c.get("horizon","H1")) for c in chosen)
    if target and target < sum(strata.values()):
        scale = target / sum(strata.values())
        scaled = {k: max(1, round(v*scale)) for k, v in strata.items()}
        # trim/pad to hit target exactly, largest strata first
        while sum(scaled.values()) > target:
            k = max(scaled, key=lambda k: scaled[k]); scaled[k] -= 1
            if scaled[k] == 0: del scaled[k]
        strata = Counter(scaled)
    pool = defaultdict(list)
    for c in cells:
        if c.get("cls")=="placebo" and c.get("half_b"): pool[(c["game_id"], c.get("horizon","H1"))].append(c)
    out=[]
    for key,n in strata.items():
        cand=sorted(pool.get(key,[]), key=lambda c: hashlib.sha256(f"{c['game_id']}:{c['qid']}:{c['event_id']}".encode()).hexdigest())
        out.extend(cand[:n])
    return out

def table(rows):
    ds = sorted(dB(c) for c in rows)
    return {"n":len(rows),
            "by_world":dict(sorted(Counter(c["game_id"] for c in rows).items())),
            "by_horizon":dict(sorted(Counter(c.get("horizon","H1") for c in rows).items())),
            "by_template":dict(sorted(Counter(tfam(c["question"]) for c in rows).items(), key=lambda kv:-kv[1])),
            "abs_delta":{"p10":round(ds[len(ds)//10],4),"p50":round(ds[len(ds)//2],4),
                          "p90":round(ds[9*len(ds)//10],4),"max":round(ds[-1],4)} if ds else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cells",required=True); ap.add_argument("--per-horizon",type=int,default=90)
    ap.add_argument("--placebo-n",type=int,default=None)
    ap.add_argument("--seed",type=int,default=11); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    payload=json.load(open(a.cells))
    cells=payload["cells"] if isinstance(payload,dict) and "cells" in payload else payload
    eff_all=[c for c in cells if c.get("cls")=="effect" and c.get("half_b")]
    quasi=[c for c in eff_all if is_quasi(c)]
    eff=[c for c in eff_all if not is_quasi(c) and not malformed(c)]
    malformed_n=sum(1 for c in eff_all if malformed(c))
    chosen=pick(eff,a.per_horizon,a.seed)
    plac=[c for c in matched_placebos([x for x in cells if not malformed(x)],chosen,a.placebo_n)]
    out={"meta":{"design":"horizon-balanced, |delta|-weighted within horizon",
                 "per_horizon_target":a.per_horizon,"seed":a.seed,
                 "certified_pool":len(eff_all),"quasi_excluded":len(quasi),"malformed_excluded":malformed_n,
                 "caps":{"world":MAX_PER_WORLD,"world_event":MAX_PER_EVENT,"template":MAX_PER_TEMPLATE}},
         "summary":{"effect":table(chosen),"placebo":table(plac)},
         "effect_cells":chosen,"placebo_cells":plac,"quasi_cells":quasi}
    json.dump(out,open(a.out,"w"),indent=1)
    print(f"pool {len(eff_all)} ({len(quasi)} quasi excluded) -> {len(chosen)} effect + {len(plac)} placebo")
    print(json.dumps(out["summary"],indent=1))

if __name__=="__main__": main()
