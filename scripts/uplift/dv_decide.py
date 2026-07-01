"""Track B decision experiment (cheap, LLM-only).

Reuses the precomputed scenario bank (dv_build_bank.py). For each scenario the
LLM (player 0) is told its sole objective is to maximize gold by turn T+H and
must choose which government to adopt. Three arms:

  control   : state report only
  true      : state report + game-true p_mc forecast table
  scrambled : state report + the same probability values, shuffled across events
              (destroys the true signal but keeps the "there are numbers" framing)

The chosen government's *precomputed* mean gold (from the bank) is the outcome.
We measure whether forecasts raise mean gold / lower regret / raise best-pick rate.

Usage:
  PYTHONPATH=src uv run python scripts/uplift/dv_decide.py \
    --bank data/uplift/bank_pilot.json --repeats 6 --workers 12 \
    --output data/uplift/decisions_pilot.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CIVBENCH_ROOT, init_llm, llm_ask, find_state_json, parse_state_json

GOVERNMENTS = ["Despotism", "Monarchy", "Republic", "Democracy"]

EVENT_LABELS = {
    "at_war": "You are at war with a rival",
    "lose_city": "You lose at least one city",
    "gain_city": "You found/capture at least one new city",
    "army_shrinks": "Your military (unit count) shrinks",
    "rival_ahead": "A rival's score exceeds yours",
    "top_score": "You hold the top score",
}

GOV_NOTE = (
    "Government trade-offs (FreeCiv classic): Despotism/Monarchy are stable and "
    "war-tolerant but generate less trade income; Republic/Democracy boost "
    "trade (more gold) but suffer unhappiness during war and are prone to civil "
    "disorder. The best choice for maximizing gold depends on how safe and "
    "stable the next several turns will be."
)


def state_report(seed: int, ckpt: int) -> str:
    """Compact natural-language snapshot of player 0 at the checkpoint."""
    rec = CIVBENCH_ROOT / "logs" / "recordings" / f"seed{seed}"
    sj = find_state_json(str(rec), ckpt)
    if not sj:
        return f"(turn {ckpt}; detailed state unavailable)"
    ev = parse_state_json(sj, 0)
    rivals = {k: v for k, v in (ev.get("scores") or {}).items() if k != "0"}
    return (
        f"Turn: {ev.get('turn', ckpt)}\n"
        f"Your civilization (player 0):\n"
        f"  - Government: {ev.get('government_name')}\n"
        f"  - Treasury (gold): {ev.get('gold')}\n"
        f"  - Cities: {ev.get('n_cities')}\n"
        f"  - Military units: {ev.get('n_units')}\n"
        f"  - Score: {ev.get('score')}\n"
        f"  - Currently at war: {'yes' if ev.get('at_war') else 'no'}\n"
        f"  - Rival scores: {rivals}\n"
    )


def forecast_table(p_mc: dict, fc_horizons: list, order: list) -> str:
    """Render p_mc as a percentage table over horizons. `order` maps display
    event slots to underlying keys (identity for 'true', shuffled for 'scrambled')."""
    hs = [f"h{h}" for h in fc_horizons]
    header = "Event | " + " | ".join(f"next {h} turns" for h in fc_horizons)
    lines = [header, "-" * len(header)]
    for disp_evt, src_evt in order:
        cells = []
        for h in hs:
            v = p_mc.get(h, {}).get(src_evt)
            cells.append(f"{round(100*v)}%" if v is not None else "n/a")
        lines.append(f"{EVENT_LABELS[disp_evt]} | " + " | ".join(cells))
    return "\n".join(lines)


def build_prompt(scenario: dict, arm: str, rng: random.Random) -> str:
    seed, ckpt = scenario["seed"], scenario["checkpoint"]
    horizon = scenario["horizon"]
    report = state_report(seed, ckpt)
    objective = (
        f"You are player 0 in a game of FreeCiv. Your SOLE objective is to "
        f"maximize your treasury (gold) at turn {ckpt + horizon} "
        f"({horizon} turns from now). Nothing else matters for this task."
    )
    task = (
        "Decision: which government should you adopt right now to best achieve "
        f"that objective? Choose exactly one of: {', '.join(GOVERNMENTS)}.\n"
        'Respond in JSON: {"reasoning": "<brief>", "choice": "<Government>"}'
    )
    parts = [objective, "", "Current situation:", report, "", GOV_NOTE]
    if arm in ("true", "scrambled"):
        ev_keys = scenario["event_names"] if "event_names" in scenario else list(EVENT_LABELS)
        ev_keys = [e for e in EVENT_LABELS if e in (scenario.get("p_mc", {}).get(
            f"h{scenario['fc_horizons'][-1]}", {}))]
        if arm == "true":
            order = [(e, e) for e in ev_keys]
        else:  # scrambled: permute which probability attaches to which event
            shuffled = ev_keys[:]
            rng.shuffle(shuffled)
            order = list(zip(ev_keys, shuffled))
        table = forecast_table(scenario["p_mc"], scenario["fc_horizons"], order)
        parts += ["", "Monte-Carlo forecasts (game-true probabilities estimated "
                  "by simulating this position forward many times):", table]
    parts += ["", task]
    return "\n".join(parts)


def parse_choice(text: str):
    import re
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            c = str(obj.get("choice", "")).strip().capitalize()
            if c in GOVERNMENTS:
                return c, obj.get("reasoning", "")
        except Exception:  # noqa: BLE001
            pass
    # fallback: last government name mentioned
    found = [g for g in GOVERNMENTS if g.lower() in text.lower()]
    return (found[-1] if found else None), text[:200]


def score_choice(scenario: dict, choice: str):
    gp = scenario["gold_by_policy"]
    means = {g: gp[g]["mean"] for g in GOVERNMENTS if gp[g]["mean"] is not None}
    if not means or choice not in means:
        return None
    best = max(means.values())
    return {"gold": means[choice], "best_gold": best, "best_gov": scenario["best_gov"],
            "regret": best - means[choice], "chose_best": abs(means[choice] - best) < 1e-6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--repeats", type=int, default=6, help="LLM samples per scenario/arm")
    ap.add_argument("--arms", nargs="+", default=["control", "true", "scrambled"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bank = json.loads(Path(args.bank).read_text())
    scenarios = bank["scenarios"]
    for sc in scenarios:
        sc.setdefault("event_names", bank.get("event_names"))
    model = init_llm(args.model) if args.model else init_llm()

    # Build the full job list.
    jobs = []
    for si, sc in enumerate(scenarios):
        for arm in args.arms:
            for rep in range(args.repeats):
                jobs.append((si, arm, rep))

    from concurrent.futures import ThreadPoolExecutor
    rng_master = random.Random(12345)

    def run_job(job):
        si, arm, rep = job
        sc = scenarios[si]
        rng = random.Random((si, hash(arm), rep, 7))
        prompt = build_prompt(sc, arm, rng)
        try:
            text = llm_ask(model, prompt, temperature=0.7)
        except Exception as e:  # noqa: BLE001
            return {"si": si, "seed": sc["seed"], "ckpt": sc["checkpoint"],
                    "arm": arm, "rep": rep, "choice": None, "error": repr(e)[:200]}
        choice, reasoning = parse_choice(text)
        rec = {"si": si, "seed": sc["seed"], "ckpt": sc["checkpoint"], "arm": arm,
               "rep": rep, "choice": choice, "reasoning": reasoning[:400]}
        if choice:
            rec["outcome"] = score_choice(sc, choice)
        return rec

    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, j): i for i, j in enumerate(jobs)}
        done = 0
        for fut in futs:
            pass
        for fut, i in futs.items():
            results[i] = fut.result()
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} decisions")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"bank": args.bank, "arms": args.arms, "repeats": args.repeats,
         "model": args.model or "default", "decisions": results}, indent=2))
    print(f"Wrote {len(results)} decisions -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
