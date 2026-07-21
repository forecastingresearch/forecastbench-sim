#!/usr/bin/env python3
"""Goal B Stage 2: single-turn causal elicitation (phase 4 of the golden-question
pilot). For each (seed, question Y) pair, elicits five forecasts:

  base        p̂(Y)                       — plain world report
  do          p̂(Y | do(X))               — external grant of tech T at turn 60
  obs         p̂(Y | X observed)          — T discovered naturally by turn 70
  placebo_do  same do-framing with a tech the player ALREADY knows (Δ*=0)
  placebo_obs same obs-framing with the already-known tech (Δ*=0)

Prompt = the OF binary prompt (identical to RFT training/eval), with the framing
sentence appended to the background. Decoding: temp 1.0, samples averaged.

Usage:
  set -a; source .env; set +a
  uv run python scripts/uplift_v2/causal_elicit.py \
      --model 'accounts/fireworks/models/deepseek-v3' --tag deepseek \
      --seeds 0 1 2 3 --samples 3 --concurrency 12 \
      --output tmp/causal/elicit_deepseek.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_of_dataset import OF_BINARY_PROMPT  # noqa: E402
from eval_a1_gate import call_chat, parse_prob  # noqa: E402

CHOICES = json.loads(Path("tmp/causal/phase2_choices.json").read_text())


def civ_name(game_data: dict, player_id: int) -> str:
    for e in game_data["events"]:
        if e["type"] == "tech_discovered" and e.get("player_id") == player_id:
            return e["description"].split(" discovered ")[0]
    raise ValueError(f"no tech event for player {player_id}")


def placebo_tech(game_data: dict, player_id: int, exclude: str) -> str:
    known = [e["description"].split(" discovered ")[-1]
             for e in game_data["events"]
             if e["type"] == "tech_discovered" and e.get("player_id") == player_id
             and e["turn"] <= 55]
    known = [t for t in known if t != exclude and not t.startswith("Tech #")]
    if not known:
        raise ValueError("no placebo candidate")
    return known[-1]  # most recently learned pre-t55 tech


FRAMINGS = {
    "base": None,
    "do": ("IMPORTANT ADDITIONAL INFORMATION: at turn 60, an external "
           "intervention directly grants the technology \"{tech}\" to {civ}, "
           "regardless of their own research choices. {civ} is guaranteed to "
           "possess \"{tech}\" from turn 60 onward."),
    "obs": ("IMPORTANT ADDITIONAL INFORMATION: it is later observed that {civ} "
            "discovered the technology \"{tech}\" naturally by turn 70, through "
            "their own research."),
}


def build_prompt(report: str, question: str, resolution_turn: int,
                 framing: str | None) -> str:
    background = (
        "This is a partial report on a FreeCiv game simulation in progress, "
        "observed at turn 60. Five AI civilizations are competing.\n\n" + report)
    if framing:
        background += "\n\n" + framing
    resolution_criteria = (
        f"Resolves YES if the answer to the question is affirmative in the "
        f"simulation state at turn {resolution_turn}, as determined by the "
        f"game's recorded metrics.")
    return OF_BINARY_PROMPT.format(question_title=question,
                                   background=background,
                                   resolution_criteria=resolution_criteria)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    jobs = []
    for seed in args.seeds:
        gid = f"seed{seed}"
        ch = CHOICES[str(seed)]
        gd = json.load(open(f"data/games_mc/{gid}_data.json"))
        civ = civ_name(gd, ch["player"])
        pl_tech = placebo_tech(gd, ch["player"], ch["tech"])
        report = Path(
            f"data/questions_mc/{gid}/world_report/turn_060_report.txt"
        ).read_text()
        report = re.sub(r"^={10,} WORLD REPORT.*?\n|={10,} END REPORT.*$", "",
                        report, flags=re.DOTALL).strip()
        qbank = json.load(open(f"data/questions_mc/{gid}/questions.json"))
        qs = [q for q in qbank["questions"] if q.get("horizon") == "H1"]
        if args.limit_questions:
            qs = qs[:args.limit_questions]
        for q in qs:
            for fr in ("base", "do", "obs", "placebo_do", "placebo_obs"):
                base_fr = fr.replace("placebo_", "")
                tech = pl_tech if fr.startswith("placebo") else ch["tech"]
                text = (FRAMINGS[base_fr].format(tech=tech, civ=civ)
                        if FRAMINGS[base_fr] else None)
                prompt = build_prompt(report, q["question_text"],
                                      q["resolution_turn"], text)
                for s in range(args.samples):
                    jobs.append({"game_id": gid, "qid": q["question_id"],
                                 "framing": fr, "sample": s, "prompt": prompt,
                                 "civ": civ, "tech": tech})

    print(f"[{args.tag}] {len(jobs)} calls "
          f"({len(args.seeds)} seeds x 5 framings x {args.samples} samples)")

    def work(j):
        for attempt in range(3):
            try:
                out = call_chat(args.model, j["prompt"])
                return {**{k: j[k] for k in
                           ("game_id", "qid", "framing", "sample", "tech")},
                        "p": parse_prob(out["text"]),
                        "finish": out["finish"],
                        "completion_tokens":
                            out["usage"].get("completion_tokens")}
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                time.sleep(5 * (attempt + 1))
        return {**{k: j[k] for k in
                   ("game_id", "qid", "framing", "sample", "tech")},
                "p": None, "error": err}

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for k, f in enumerate(cf.as_completed(futs)):
            results.append(f.result())
            if (k + 1) % 100 == 0:
                bad = sum(1 for r in results if r.get("p") is None)
                print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s, "
                      f"{bad} unparsed)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "tag": args.tag, "results": results},
              open(args.output, "w"), indent=1)
    bad = sum(1 for r in results if r.get("p") is None)
    print(f"done: {len(results)} calls, {bad} unparsed -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
