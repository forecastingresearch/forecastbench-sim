#!/usr/bin/env -S uv run python3
"""Ask models the natural-conditional prompts (build_natcond_prompts.py output).

Turn 1 for every question, then turn 2 for every cell on top of the model's own
turn-1 reply. Rows go to <out>/results.jsonl; rerunning resumes. Every model
must have an entry in the model registry (provider pin, reasoning effort): the
paper's is forecastbench-sim-paper/production/micropolis/micropolis_world/
model_specs.json5. Needs OPENROUTER_API_KEY (credentials/.env).

Usage:
    scripts/elicit_natcond.py --model-specs SPECS --models deepseek/deepseek-chat --limit 2
    scripts/elicit_natcond.py --model-specs SPECS          # the 22-model panel, configs/models-22.json5
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import json5

from micropolis_world import module_globals as g


def load_models(arg: str) -> list[str]:
    p = Path(arg)
    return list(json5.loads(p.read_text())) if p.suffix in (".json", ".json5") and p.exists() else arg.split(",")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(Path(__file__).resolve().parents[1] / "configs" / "models-22.json5"),
                    help="comma-separated slugs, or a JSON5 list file (default: configs/models-22.json5, the 22-model panel)")
    ap.add_argument("--model-specs", type=Path, default=None,
                    help="model registry (default: $MICROPOLIS_MODEL_SPECS, else the package's)")
    ap.add_argument("--prompts", type=Path, default=None, help="default: data/micropolis/natcond/natcond_400_seed42_prompts.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default: <prompts stem>_elicit/ beside the prompts")
    ap.add_argument("--limit", type=int, default=0, help="only the first N turn-1 questions and their cells")
    ap.add_argument("--reask", type=int, default=1, help="re-ask an unparsed reply this many times (default 1)")
    ap.add_argument("--concurrency", type=int, default=None, help="calls in flight across all models (default 16)")
    ap.add_argument("--retries", type=int, default=8,
                    help="retries per call on rate limits and server errors (default 8: backoff up to ~2 min, "
                    "for providers' shared-pool 429s)")
    ap.add_argument("--dry-run", action="store_true", help="count the calls and stop")
    args = ap.parse_args()

    # The registry is read when the client is first imported, so set it before.
    if args.model_specs:
        os.environ["MICROPOLIS_MODEL_SPECS"] = str(args.model_specs.resolve())
    from micropolis_world.natcond_elicit import Results, elicit
    from micropolis_world.openrouter_completion import MODEL_SPECS_PATH, MODELS

    models = load_models(args.models)
    missing = [m for m in models if m not in MODELS]
    if missing:
        raise SystemExit(f"no entry in {MODEL_SPECS_PATH} for: {', '.join(missing)}")

    prompts = args.prompts or g.DATA_DIR / "natcond" / "natcond_400_seed42_prompts.jsonl"
    with open(prompts) as f:
        rows = [json.loads(line) for line in f]
    jobs = {"t1": [r for r in rows if r["arm"] == "t1"], "t2": [r for r in rows if r["arm"] == "t2"]}
    if args.limit:
        keep = {j["question_id"] for j in jobs["t1"][: args.limit]}
        jobs = {arm: [j for j in v if j["question_id"] in keep] for arm, v in jobs.items()}
    out = args.out or prompts.with_name(prompts.stem.removesuffix("_prompts") + "_elicit")
    out.mkdir(parents=True, exist_ok=True)
    results = Results(out / "results.jsonl")

    print(f"{len(models)} model(s), {len(jobs['t1'])} turn-1 and {len(jobs['t2'])} turn-2 prompts each -> {out}")
    if args.dry_run:
        return
    g.ensure_api_keys()
    answered, total = asyncio.run(
        elicit(jobs, models, results, reask=args.reask, concurrency=args.concurrency, retries=args.retries))

    cost = sum((r["usage"] or {}).get("cost_usd") or 0 for rs in results.rows.values() for r in rs)
    print(f"\n{answered}/{total} prompts answered; ${cost:.4f} spent in {results.path} so far")


if __name__ == "__main__":
    main()
