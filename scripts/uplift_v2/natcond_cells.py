#!/usr/bin/env python3
"""Build natural-conditional benchmark cells from archived baseline rollouts.

A conditioning event X is a predicate over a rollout's event log within a turn
window. Two abstraction levels, chosen a priori:
  specific: an exact event ("Benin discovered Feudalism", turns 75-90)
  vague:    an event class ("Benin loses a city", "any civ falls into Anarchy")

For each X with window frequency in [0.15, 0.65], and each question Y in the
world's H1 bank, emit a cell with ground truth:
  p_y        = P(Y) over all rollouts
  p_yx       = P(Y | X) over the conditioning subset (n_x rollouts)
  delta      = p_yx - p_y, se_delta (binomial, subset-dominated)
  cell class = 'effect' if |delta| > 2*se_delta else 'placebo-ish'

Usage:
  uv run python scripts/uplift_v2/natcond_cells.py --game-id seed2 \
      --arm-dir tmp/causal/seed2/baseline --out tmp/natcond/seed2_cells.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path


def vague_predicates(civs: list[str]):
    """Event-class predicates: (name, human description template, fn(events, a, b))."""
    preds = []
    for civ in civs:
        preds.append((
            f"{civ}_loses_city",
            f"{civ} loses at least one city (conquered or destroyed)",
            lambda evs, a, b, civ=civ: any(
                (e["type"] == "city_conquered" and f"from {civ}" in e["description"])
                or (e["type"] == "city_destroyed" and f"({civ})" in e["description"])
                for e in evs if a < e["turn"] <= b)))
        preds.append((
            f"{civ}_anarchy",
            f"{civ} falls into Anarchy (any government collapse)",
            lambda evs, a, b, civ=civ: any(
                e["type"] == "government_change" and e["description"].startswith(civ)
                and "to Anarchy" in e["description"]
                for e in evs if a < e["turn"] <= b)))
    preds.append((
        "any_wonder",
        "any civilization completes a Wonder of the World",
        lambda evs, a, b: any(e["type"] == "wonder_completed"
                              for e in evs if a < e["turn"] <= b)))
    preds.append((
        "pirate_conquest",
        "Pirates conquer at least one city from any civilization",
        lambda evs, a, b: any(e["type"] == "city_conquered"
                              and "by Pirate" in e["description"]
                              for e in evs if a < e["turn"] <= b)))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--windows", default="60-75,75-90")
    ap.add_argument("--freq-band", default="0.15,0.65")
    ap.add_argument("--max-specific", type=int, default=6)
    ap.add_argument("--max-vague", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lo, hi = map(float, args.freq_band.split(","))
    windows = [tuple(map(int, w.split("-"))) for w in args.windows.split(",")]

    manifest = json.loads((Path(args.arm_dir) / "manifest.json").read_text())
    answers = {qid: {r["tag"]: r["answer"] for r in recs if r["answer"] is not None}
               for qid, recs in manifest["answers"].items()}
    qbank = json.loads(Path(
        f"data/questions_mc/{args.game_id}/questions.json").read_text())
    qtext = {q["question_id"]: q["question_text"] for q in qbank["questions"]
             if q.get("horizon") == "H1"}

    rollouts = {}
    civs = set()
    for fp in sorted(glob.glob(f"{args.arm_dir}/rollouts/*.json.gz")):
        tag = Path(fp).name.split(".")[0]
        try:
            gd = json.load(gzip.open(fp, "rt"))
        except Exception:
            print(f"  !! skipping unreadable rollout {tag}")
            continue
        rollouts[tag] = gd["events"]
        for e in gd["events"]:
            if e["type"] == "tech_discovered" and " discovered " in e["description"]:
                civs.add(e["description"].split(" discovered ")[0])
    civs = sorted(c for c in civs if c not in ("Pirate",))
    N = len(rollouts)
    min_nx = max(12, int(0.15 * N))
    lo = max(lo, min_nx / N)  # keep the freq band consistent with the n_x floor
    print(f"{args.game_id}: {N} rollouts, civs={civs}, min_nx={min_nx}, band=[{lo:.2f},{hi}]")

    events = []  # (kind, name, description, window, member_tags)
    # specific events: exact description within window
    for a, b in windows:
        counts = Counter()
        members = {}
        for tag, evs in rollouts.items():
            seen = set()
            for e in evs:
                if a < e["turn"] <= b and e["type"] in (
                        "tech_discovered", "government_change",
                        "wonder_completed"):
                    d = e["description"]
                    if d not in seen:
                        seen.add(d)
                        counts[d] += 1
                        members.setdefault(d, []).append(tag)
        cands = [(d, c) for d, c in counts.items() if lo <= c / N <= hi]
        # spread across event types, prefer mid-band frequency
        cands.sort(key=lambda dc: abs(dc[1] / N - 0.4))
        used_types = Counter()
        for d, c in cands:
            ty = ("gov" if "government" in d else
                  "wonder" if "completed" in d else "tech")
            if used_types[ty] >= max(1, args.max_specific // 3):
                continue
            used_types[ty] += 1
            events.append(("specific", f"sp_{len(events)}", d, (a, b), members[d]))
            if sum(used_types.values()) >= args.max_specific // len(windows) + 1:
                break
    # vague events
    for a, b in windows:
        scored = []
        for name, desc, fn in vague_predicates(civs):
            mem = [t for t, evs in rollouts.items() if fn(evs, a, b)]
            f = len(mem) / N
            if lo <= f <= hi:
                scored.append((abs(f - 0.4), name, desc, mem))
        scored.sort()
        for _, name, desc, mem in scored[:args.max_vague // len(windows) + 1]:
            events.append(("vague", f"vg_{len(events)}", desc, (a, b), mem))

    # build cells
    cells = []
    for kind, eid, desc, (a, b), mem in events:
        mem = set(mem)
        n_x = len(mem)
        for qid, amap in answers.items():
            if qid not in qtext:
                continue
            tags = list(amap)
            p_y = sum(amap[t] for t in tags) / len(tags)
            sub = [amap[t] for t in tags if t in mem]
            if len(sub) < min_nx:
                continue
            p_yx = sum(sub) / len(sub)
            se = math.sqrt(max(p_yx * (1 - p_yx), 0.25 / len(sub)) / len(sub)
                           + p_y * (1 - p_y) / len(tags))
            delta = p_yx - p_y
            cells.append({
                "game_id": args.game_id, "qid": qid, "question": qtext[qid],
                "event_id": eid, "event_kind": kind, "event_desc": desc,
                "window": [a, b], "freq": round(n_x / N, 3), "n_x": len(sub),
                "p_y": round(p_y, 4), "p_yx": round(p_yx, 4),
                "delta": round(delta, 4), "se_delta": round(se, 4),
                "cls": "effect" if abs(delta) > 2 * se else "placebo",
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cells, open(args.out, "w"), indent=1)
    eff = sum(1 for c in cells if c["cls"] == "effect")
    print(f"{len(events)} events ({sum(1 for e in events if e[0]=='specific')} "
          f"specific, {sum(1 for e in events if e[0]=='vague')} vague) x "
          f"questions -> {len(cells)} cells ({eff} effect, "
          f"{len(cells)-eff} placebo) -> {args.out}")
    for kind, eid, desc, w, mem in events:
        print(f"  [{kind}] {w} f={len(mem)/N:.2f}  {desc[:60]}")


if __name__ == "__main__":
    main()
