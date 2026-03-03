#!/usr/bin/env python3
"""Regenerate N=2/N=3 evaluation directories with question IDs matching N=1.

The original setup_multi_intervention_eval.py has two bugs:
1. Government question IDs are shuffled vs N=1 due to set() iteration order
   in conditional_generator.py (now fixed with sorted()).
2. Only H1-H3 horizons because fork runs used END_TURN=150.

This script fixes bug #1 by re-keying existing N=2/N=3 answers to match N=1
question IDs. It matches by full parameter tuples (template_id, civ_a, civ_b,
resolution_turn, government_type, tech_name, wonder_name, etc.).

Bug #2 (H4-H6 missing) requires re-running forks with END_TURN=270 — that's a
separate step. This script only regenerates H1-H3 for now.

Usage:
    uv run python scripts/regenerate_compound_eval.py
    uv run python scripts/regenerate_compound_eval.py --verify
    uv run python scripts/regenerate_compound_eval.py --dry-run
    uv run python scripts/regenerate_compound_eval.py --seeds 0-10
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


# === Text templates (from setup_multi_intervention_eval.py) ===

BASELINE_TEXT_TEMPLATES = {
    "treasury_comparative": "Will {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "Will {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "Will {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "Will {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "Will {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "Will {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "Will {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "Will {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "Will {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "Will {civ} be in {government_type} at turn {resolution_turn}?",
    "techs_continuous": "How many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "How much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "What will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "How many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "How many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "What will {civ}'s score be at turn {resolution_turn}?",
}

N2_CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} adopts Republic AND receives +500 gold next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} adopts Republic AND receives +500 gold next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization adopts Republic AND receives +500 gold next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} adopts Republic AND receives +500 gold next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
    "techs_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, what will {civ}'s score be at turn {resolution_turn}?",
}

N3_CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
    "techs_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Navigation next turn, what will {civ}'s score be at turn {resolution_turn}?",
}

# === Condition metadata ===

N2_CONDITION_METADATA = {
    "condition_type": "compound",
    "n_interventions": 2,
    "interventions": [
        {"type": "government", "value": "Republic"},
        {"type": "gold_add", "value": 500},
    ],
    "description": "Republic + 500 gold",
    "source": "regenerate_compound_eval.py",
}

N3_CONDITION_METADATA = {
    "condition_type": "compound",
    "n_interventions": 3,
    "interventions": [
        {"type": "government", "value": "Republic"},
        {"type": "gold_add", "value": 500},
        {"type": "tech", "value": 56, "name": "Navigation"},
    ],
    "description": "Republic + 500 gold + Navigation",
    "source": "regenerate_compound_eval.py",
}


def param_key(template_id: str, params: dict) -> tuple:
    """Build a hashable key from template_id + parameters for matching."""
    tid = template_id.replace("conditional_", "")
    return (
        tid,
        params.get("civ_a", ""),
        params.get("civ_b", ""),
        params.get("civ", ""),
        str(params.get("player_id", "")),
        params.get("resolution_turn"),
        params.get("government_type", ""),
        params.get("tech_name", ""),
        params.get("wonder_name", ""),
    )


def get_horizon(checkpoint_turn: int, resolution_turn: int) -> str:
    """Calculate horizon label from turn difference."""
    diff = resolution_turn - checkpoint_turn
    horizon_map = {30: "H1", 60: "H2", 90: "H3", 120: "H4", 150: "H5", 180: "H6"}
    return horizon_map.get(diff, f"H{diff // 30}")


def format_question_text(template_id: str, params: dict, text_templates: dict) -> str:
    """Format question text using the appropriate template."""
    text_template = text_templates.get(template_id)
    if text_template:
        fmt_params = dict(params)
        if "civ" not in fmt_params:
            fmt_params["civ"] = params.get("civ_a", "the civilization")
        return text_template.format(**fmt_params)
    return f"Question about {template_id}"


def regenerate_condition(
    condition_name: str,
    conditional_text_templates: dict,
    condition_metadata: dict,
    n1_dir: Path,
    compound_dir: Path,
    seeds: list[str],
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Regenerate eval dirs for one compound condition.

    Returns:
        (total_baseline, total_conditional, total_rekey_diffs) counts
    """
    baseline_out = compound_dir / "baseline"
    conditional_out = compound_dir / "conditional"

    total_baseline = 0
    total_conditional = 0
    total_rekey_diffs = 0

    for seed in seeds:
        # 1. Load N=1 questions (source of truth for question IDs)
        n1_path = n1_dir / "baseline" / seed / "questions.json"
        if not n1_path.exists():
            print(f"  {seed}: Skipping — no N=1 questions at {n1_path}")
            continue

        with open(n1_path) as f:
            n1_data = json.load(f)
        n1_questions = [q for q in n1_data["questions"] if q["horizon"] in ("H1", "H2", "H3")]

        # 2. Load existing N=2/N=3 baseline answers
        old_baseline_path = compound_dir / "baseline" / seed / "questions.json"
        if not old_baseline_path.exists():
            print(f"  {seed}: Skipping — no existing baseline at {old_baseline_path}")
            continue
        with open(old_baseline_path) as f:
            old_baseline = json.load(f)

        # 3. Load existing N=2/N=3 conditional answers
        old_cond_path = compound_dir / "conditional" / seed / "conditional_questions.json"
        if not old_cond_path.exists():
            print(f"  {seed}: Skipping — no existing conditional at {old_cond_path}")
            continue
        with open(old_cond_path) as f:
            old_cond = json.load(f)

        # 4. Build answer lookups keyed by param_key
        baseline_answers = {}
        for q in old_baseline["questions"]:
            pk = param_key(q["template_id"], q["parameters"])
            baseline_answers[pk] = q["resolution"]

        cond_answers = {}
        for q in old_cond["questions"]:
            pk = param_key(q["template_id"], q["parameters"])
            cond_answers[pk] = q["resolution"]

        # 5. Generate new questions using N=1 IDs + compound answers
        new_baseline_qs = []
        new_cond_qs = []
        matched = 0
        rekey_diffs = 0

        for n1q in n1_questions:
            pk = param_key(n1q["template_id"], n1q["parameters"])

            b_answer = baseline_answers.get(pk)
            c_answer = cond_answers.get(pk)

            if b_answer is None or c_answer is None:
                print(f"    {seed}: WARNING — no answer match for {n1q['question_id']} "
                      f"template={n1q['template_id']} pk={pk}")
                continue

            matched += 1
            template_id = n1q["template_id"]
            params = n1q["parameters"]
            resolution_turn = n1q["resolution_turn"]
            horizon = n1q["horizon"]
            is_continuous = template_id.endswith("_continuous")

            # Check if this would have had a different ID in old data
            old_baseline_by_pk = {
                param_key(q["template_id"], q["parameters"]): q["question_id"]
                for q in old_baseline["questions"]
            }
            old_id = old_baseline_by_pk.get(pk, "").replace("_control", "")
            if old_id and old_id != n1q["question_id"]:
                rekey_diffs += 1

            # Baseline question (unconditional framing, control answer)
            baseline_text = format_question_text(template_id, params, BASELINE_TEXT_TEMPLATES)
            new_baseline_qs.append({
                "question_id": f"{n1q['question_id']}_control",
                "template_id": template_id,
                "resolution_turn": resolution_turn,
                "horizon": horizon,
                "question_type": "continuous" if is_continuous else "binary",
                "parameters": {**params, "checkpoint_turn": 60},
                "question_text": baseline_text,
                "resolution": b_answer,
            })

            # Conditional question (compound framing, intervention answer)
            cond_text = format_question_text(template_id, params, conditional_text_templates)
            new_cond_qs.append({
                "question_id": f"{n1q['question_id']}_intervention",
                "template_id": f"conditional_{template_id}",
                "resolution_turn": resolution_turn,
                "horizon": horizon,
                "question_type": "continuous" if is_continuous else "binary",
                "parameters": {**params, "checkpoint_turn": 60},
                "question_text": cond_text,
                "resolution": c_answer,
            })

        if matched != len(n1_questions):
            print(f"  {seed}: WARNING — only matched {matched}/{len(n1_questions)} questions")

        total_baseline += len(new_baseline_qs)
        total_conditional += len(new_cond_qs)
        total_rekey_diffs += rekey_diffs

        rekey_str = f" ({rekey_diffs} re-keyed)" if rekey_diffs > 0 else ""
        print(f"  {seed}: {len(new_baseline_qs)} baseline + {len(new_cond_qs)} conditional{rekey_str}")

        if dry_run:
            continue

        # 6. Write output
        civilizations = n1_data.get("civilizations", old_baseline.get("civilizations", {}))

        baseline_bank = {
            "game_id": seed,
            "snapshot_turn": 60,
            "game_max_turn": 270,
            "generated_at": datetime.now().isoformat() + "Z",
            "civilizations": civilizations,
            "questions": new_baseline_qs,
            "condition_metadata": condition_metadata,
        }

        cond_bank = {
            "game_id": seed,
            "snapshot_turn": 60,
            "game_max_turn": 270,
            "generated_at": datetime.now().isoformat() + "Z",
            "civilizations": civilizations,
            "questions": new_cond_qs,
            "condition_metadata": condition_metadata,
        }

        # Write baseline
        seed_baseline_dir = baseline_out / seed
        seed_baseline_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_baseline_dir / "questions.json", "w") as f:
            json.dump(baseline_bank, f, indent=2)

        # Write conditional
        seed_cond_dir = conditional_out / seed
        seed_cond_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_cond_dir / "conditional_questions.json", "w") as f:
            json.dump(cond_bank, f, indent=2)

        # Copy world_report from N=1 if not already present
        n1_wr = n1_dir / "baseline" / seed / "world_report"
        if n1_wr.exists():
            for target_dir in [seed_baseline_dir, seed_cond_dir]:
                wr_dest = target_dir / "world_report"
                if wr_dest.exists():
                    shutil.rmtree(wr_dest)
                shutil.copytree(n1_wr, wr_dest)

    return total_baseline, total_conditional, total_rekey_diffs


def verify_condition(
    condition_name: str,
    n1_dir: Path,
    compound_dir: Path,
    seeds: list[str],
) -> bool:
    """Verify regenerated data matches N=1 question IDs."""
    ok = True

    for seed in seeds:
        n1_path = n1_dir / "baseline" / seed / "questions.json"
        baseline_path = compound_dir / "baseline" / seed / "questions.json"
        cond_path = compound_dir / "conditional" / seed / "conditional_questions.json"

        if not all(p.exists() for p in [n1_path, baseline_path, cond_path]):
            print(f"  {seed}: SKIP — missing files")
            continue

        with open(n1_path) as f:
            n1_data = json.load(f)
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        with open(cond_path) as f:
            cond_data = json.load(f)

        n1_h123 = {q["question_id"]: q for q in n1_data["questions"]
                    if q["horizon"] in ("H1", "H2", "H3")}

        # Check baseline question IDs match N=1
        baseline_ids = {q["question_id"].replace("_control", "") for q in baseline_data["questions"]}
        n1_ids = set(n1_h123.keys())

        missing = n1_ids - baseline_ids
        extra = baseline_ids - n1_ids

        if missing or extra:
            print(f"  {seed}: FAIL — baseline missing={len(missing)}, extra={len(extra)}")
            ok = False
        else:
            # Verify parameter match for each question
            baseline_by_id = {
                q["question_id"].replace("_control", ""): q
                for q in baseline_data["questions"]
            }
            param_mismatches = 0
            for qid, n1q in n1_h123.items():
                bq = baseline_by_id.get(qid)
                if bq:
                    n1_pk = param_key(n1q["template_id"], n1q["parameters"])
                    b_pk = param_key(bq["template_id"], bq["parameters"])
                    if n1_pk != b_pk:
                        param_mismatches += 1

            # Check counts
            n_baseline = len(baseline_data["questions"])
            n_cond = len(cond_data["questions"])

            if param_mismatches > 0:
                print(f"  {seed}: FAIL — {param_mismatches} parameter mismatches")
                ok = False
            elif n_baseline != len(n1_h123) or n_cond != len(n1_h123):
                print(f"  {seed}: FAIL — counts baseline={n_baseline}, cond={n_cond} (expected {len(n1_h123)})")
                ok = False
            else:
                # Spot-check: government questions have same gov_type for same ID
                gov_checks = 0
                for qid, n1q in n1_h123.items():
                    if n1q["template_id"] == "government_at":
                        bq = baseline_by_id.get(qid)
                        if bq:
                            n1_gov = n1q["parameters"].get("government_type")
                            b_gov = bq["parameters"].get("government_type")
                            if n1_gov == b_gov:
                                gov_checks += 1
                            else:
                                print(f"  {seed}: FAIL — {qid} gov mismatch: N=1={n1_gov}, compound={b_gov}")
                                ok = False

                print(f"  {seed}: OK — 126+126 questions, IDs match, {gov_checks} gov checks passed")

    return ok


def parse_seed_range(s: str) -> list[str]:
    """Parse seed range like '0-10' or '0,5,10' into seed names."""
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return [f"seed{i}" for i in range(int(start), int(end) + 1)]
    elif "," in s:
        return [f"seed{i}" for i in s.split(",")]
    else:
        return [f"seed{int(s)}"]


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate N=2/N=3 eval dirs with question IDs matching N=1"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output matches N=1 IDs instead of regenerating")
    parser.add_argument("--seeds", type=str, default="0-20",
                        help="Seed range (default: 0-20)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    n1_dir = base_dir / "data" / "conditional" / "republic"
    seeds = parse_seed_range(args.seeds)

    conditions = [
        ("republic_gold500", N2_CONDITIONAL_TEXT_TEMPLATES, N2_CONDITION_METADATA),
        ("republic_gold500_navigation", N3_CONDITIONAL_TEXT_TEMPLATES, N3_CONDITION_METADATA),
    ]

    if args.verify:
        print("Verifying regenerated eval data...\n")
        all_ok = True
        for cond_name, _, _ in conditions:
            compound_dir = base_dir / "data" / "conditional" / cond_name
            print(f"=== {cond_name} ===")
            ok = verify_condition(cond_name, n1_dir, compound_dir, seeds)
            if not ok:
                all_ok = False
            print()

        if all_ok:
            print("All verifications passed.")
        else:
            print("VERIFICATION FAILED — see above for details.")
            return 1
        return 0

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    for cond_name, cond_templates, cond_metadata in conditions:
        compound_dir = base_dir / "data" / "conditional" / cond_name
        print(f"{'=' * 60}")
        print(f"Regenerating: {cond_name}")
        print(f"{'=' * 60}")

        n_base, n_cond, n_rekey = regenerate_condition(
            condition_name=cond_name,
            conditional_text_templates=cond_templates,
            condition_metadata=cond_metadata,
            n1_dir=n1_dir,
            compound_dir=compound_dir,
            seeds=seeds,
            dry_run=args.dry_run,
        )

        print(f"\n  Total: {n_base} baseline + {n_cond} conditional ({n_rekey} re-keyed)")
        print()

    if not args.dry_run:
        print("Done. Run with --verify to check output.")


if __name__ == "__main__":
    exit(main() or 0)
