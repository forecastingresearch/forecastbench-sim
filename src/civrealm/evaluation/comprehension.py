"""Generate and score basic comprehension checks on a world report."""

from __future__ import annotations

import json
import asyncio
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from civrealm.evaluation.compress_world_report import compress_world_report
from civrealm.evaluation.rate_limiter import ProviderRateLimiter

_thread_pool = ThreadPoolExecutor(max_workers=50, thread_name_prefix="comp_query")


def _turn_metric(data: dict, metric: str, turn: int) -> dict[int, Any]:
    """Extract metric values at a given turn.

    Works with compressed format (history._turns + history.metric.civ_id = [values])
    or falls back to full format (time_series.metric.turn.civ_id = value).
    """
    # Try compressed format first (history with _turns array)
    history = data.get("history", {})
    if history and "_turns" in history and metric in history:
        turns_list = history["_turns"]
        metric_data = history[metric]
        # Find the index for this turn
        if turn in turns_list:
            idx = turns_list.index(turn)
            result = {}
            for civ_id_str, values in metric_data.items():
                if civ_id_str == "_turns":
                    continue
                if idx < len(values) and values[idx] is not None:
                    result[int(civ_id_str)] = values[idx]
            return result
        return {}

    # Fall back to full format
    metric_data = data.get("time_series", {}).get(metric, {})
    turn_data = metric_data.get(str(turn)) or metric_data.get(turn) or {}
    return {int(k): v for k, v in turn_data.items() if isinstance(turn_data, dict)} if isinstance(turn_data, dict) else {}


def _civ_names(data: dict) -> dict[int, str]:
    """Extract civilization names.

    Works with compressed format (metadata.civs = [names]) or full format (civilizations).
    """
    # Try compressed format first
    civ_names_list = data.get("metadata", {}).get("civs", [])
    if civ_names_list:
        # In compressed format, civ IDs are derived from history keys
        history = data.get("history", {})
        civ_ids = []
        for key, val in history.items():
            if key != "_turns" and isinstance(val, dict):
                civ_ids = sorted(int(k) for k in val.keys())
                break
        if civ_ids and len(civ_ids) == len(civ_names_list):
            return {cid: civ_names_list[i] for i, cid in enumerate(civ_ids)}

    # Fall back to full format
    civs = data.get("civilizations", {}) or {}
    return {int(pid): info.get("name", f"Civ {pid}") for pid, info in civs.items()}


def _top_civs_from_dict(values: dict[int, Any]) -> list[int]:
    """Return all civ ids tied for the maximum value."""
    if not values:
        return []
    max_val = max(values.values())
    return [cid for cid, val in values.items() if val == max_val]


def generate_comprehension_questions(
    report_data: dict,
    game_id: str,
) -> list[dict[str, Any]]:
    """Build a set of factual yes/no comprehension questions from a world report.

    Works with both compressed and full report formats.
    """
    questions: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    turn = int(report_data.get("metadata", {}).get("turn", 0))
    civ_names = _civ_names(report_data)
    civ_ids = sorted(civ_names.keys())

    # Determine available turns - try compressed format first (history._turns)
    history = report_data.get("history", {})
    if history and "_turns" in history:
        turns_available = list(history["_turns"])
    else:
        turns_available = report_data.get("metadata", {}).get("turns_analyzed") or []
        if not turns_available:
            ts = report_data.get("time_series", {})
            if ts:
                first_metric = next(iter(ts.values()), {})
                if isinstance(first_metric, dict):
                    turns_available = sorted(int(t) for t in first_metric.keys())
    if not turns_available:
        turns_available = [turn]
    turn_choices = turns_available or [turn]

    def add(question_text: str, truth: bool, template_id: str) -> bool:
        if question_text in seen_texts:
            return False
        seen_texts.add(question_text)
        questions.append(
            {
                "question_id": f"comp{len(questions)+1:03d}",
                "game_id": game_id,
                "question_text": question_text,
                "ground_truth": bool(truth),
                "difficulty": {"composite": 0, "info": "I0", "horizon": 0},
                "template_id": template_id,
            }
        )
        return True

    def add_unique(builder, template_id: str, target: int = 3, max_attempts: int = 30):
        added = 0
        attempts = 0
        while added < target and attempts < max_attempts:
            attempts += 1
            res = builder()
            if not res:
                continue
            question_text, truth = res
            if add(question_text, truth, template_id):
                added += 1
        return added

    def pick_turn() -> int:
        return int(random.choice(turn_choices))

    def pick_civ() -> int | None:
        return random.choice(civ_ids) if civ_ids else None

    # Tech leader
    def build_tech_question():
        tsel = pick_turn()
        techs = _turn_metric(report_data, "techs_known", tsel)
        tech_leaders = _top_civs_from_dict(techs)
        if not tech_leaders:
            return None
        tech_leader = random.choice(tech_leaders)
        return (
            f"Does {civ_names.get(tech_leader, 'this civilization')} have the highest technology count at turn {tsel}? (ties count as YES)",
            True,
        )

    add_unique(build_tech_question, "comp_tech_leader")

    # Population leader
    def build_pop_question():
        tsel = pick_turn()
        pop = _turn_metric(report_data, "population", tsel)
        pop_leaders = _top_civs_from_dict(pop)
        if not pop_leaders:
            return None
        pop_leader = random.choice(pop_leaders)
        return (
            f"Is {civ_names.get(pop_leader, 'this civilization')} the most populous at turn {tsel}? (ties count as YES)",
            True,
        )

    add_unique(build_pop_question, "comp_pop_leader")

    # Territory leader
    def build_terr_question():
        tsel = pick_turn()
        territory = _turn_metric(report_data, "territory_size", tsel)
        terr_leaders = _top_civs_from_dict(territory)
        if not terr_leaders:
            return None
        terr_leader = random.choice(terr_leaders)
        return (
            f"Does {civ_names.get(terr_leader, 'this civilization')} control the most territory at turn {tsel}? (ties count as YES)",
            True,
        )

    add_unique(build_terr_question, "comp_territory_leader")

    # Treasury leader
    def build_treas_question():
        tsel = pick_turn()
        treasury = _turn_metric(report_data, "treasury", tsel)
        treas_leaders = _top_civs_from_dict(treasury)
        if not treas_leaders:
            return None
        treas_leader = random.choice(treas_leaders)
        return (
            f"Does {civ_names.get(treas_leader, 'this civilization')} have the largest treasury at turn {tsel}? (ties count as YES)",
            True,
        )

    add_unique(build_treas_question, "comp_treasury_leader")

    # City presence (random civ)
    def build_city_presence():
        civ_id = pick_civ()
        if civ_id is None:
            return None
        tsel = pick_turn()
        city_counts = _turn_metric(report_data, "cities_count", tsel)
        city_has = city_counts.get(civ_id, 0) > 0
        return (
            f"Does {civ_names[civ_id]} have at least one city at turn {tsel}?",
            city_has,
        )

    add_unique(build_city_presence, "comp_city_presence")

    # Any city founded
    # Compressed format uses "t" for turn and "city" for type (abbreviated)
    def build_city_founded():
        tsel = pick_turn()
        founded = any(
            (e.get("type") == "city_founded" or e.get("type") == "city")
            and int(e.get("turn", e.get("t", 0))) <= tsel
            for e in report_data.get("events", [])
        )
        return (
            f"Has any civilization founded a city by turn {tsel}?",
            founded,
        )

    add_unique(build_city_founded, "comp_city_founded")

    # Any wonder completion
    # Compressed format uses "t" for turn, "wond" for type, and "civ" for player_id
    def build_wonder():
        tsel = pick_turn()
        wonder_events = [
            e for e in report_data.get("events", [])
            if (e.get("type") == "wonder_completed" or e.get("type") == "wond")
            and int(e.get("turn", e.get("t", 0))) <= tsel
        ]
        if wonder_events:
            first = random.choice(wonder_events)
            civ_id = int(first.get("player_id", first.get("civ", -1)))
            return (
                f"Did {civ_names.get(civ_id, 'a civilization')} complete a wonder by turn {tsel}?",
                True,
            )
        return (
            f"Has any civilization completed a wonder by turn {tsel}?",
            False,
        )

    add_unique(build_wonder, "comp_wonder")

    # Diplomacy questions - handle both compressed and full formats
    diplomacy = report_data.get("diplomacy", {})
    relations = diplomacy.get("relations", {}) or {}

    # Check if this is compressed format (has current_state/state_changes)
    is_compressed_diplomacy = "current_state" in diplomacy or "state_changes" in diplomacy

    def _has_met_by_turn(tsel: int) -> bool:
        if is_compressed_diplomacy:
            # For compressed format, check state_changes for any meeting before tsel
            # or current_state if tsel == current turn
            state_changes = diplomacy.get("state_changes", [])
            current_state = diplomacy.get("current_state", {})
            # If any pair has current state != "Never met" and tsel >= turn, they've met
            if tsel >= turn and any(s != "Never met" for s in current_state.values()):
                return True
            # Check state changes for earlier turns
            for change in state_changes:
                if int(change.get("t", 0)) <= tsel:
                    # Any state change means they've met (you can't have a state change without meeting)
                    return True
            return False
        else:
            # Full format
            return any(
                state.get("state") and state.get("state") != "Never met"
                for history in relations.values()
                for turn_key, state in history.items()
                if isinstance(history, dict) and isinstance(state, dict) and int(turn_key) <= tsel
            )

    def _at_war_at_turn(tsel: int) -> bool:
        if is_compressed_diplomacy:
            state_changes = diplomacy.get("state_changes", [])
            current_state = diplomacy.get("current_state", {})
            # For each pair, determine state at tsel by replaying state changes
            pair_states: dict[str, str] = {}
            for change in sorted(state_changes, key=lambda c: int(c.get("t", 0))):
                if int(change.get("t", 0)) <= tsel:
                    pair_states[change.get("pair", "")] = change.get("to", "")
            # If tsel >= turn, use current_state for any pairs not in state_changes
            if tsel >= turn:
                for pair, state in current_state.items():
                    if pair not in pair_states:
                        pair_states[pair] = state
            return "War" in pair_states.values()
        else:
            # Full format
            return any(
                state.get("state") == "War"
                for history in relations.values()
                for turn_key, state in history.items()
                if isinstance(history, dict) and isinstance(state, dict) and int(turn_key) <= tsel
            )

    # Any meeting
    def build_met():
        tsel = pick_turn()
        met = _has_met_by_turn(tsel)
        return (
            f"Have any civilizations met by turn {tsel}?",
            met,
        )

    add_unique(build_met, "comp_met")

    # Any war
    def build_war():
        tsel = pick_turn()
        at_war = _at_war_at_turn(tsel)
        return (
            f"Are any civilizations at war at turn {tsel}?",
            at_war,
        )

    add_unique(build_war, "comp_war")

    # Military units presence (random civ)
    def build_military():
        civ_id = pick_civ()
        if civ_id is None:
            return None
        tsel = pick_turn()
        mil_counts = _turn_metric(report_data, "military_units_count", tsel)
        has_military = mil_counts.get(civ_id, 0) > 0
        return (
            f"Does {civ_names[civ_id]} have any military units at turn {tsel}?",
            has_military,
        )

    add_unique(build_military, "comp_military_presence")

    return questions


def summarize_comprehension_accuracy(
    question_results: list[dict],
    model_ids: list[str],
) -> dict[str, dict[str, float | int]]:
    """Compute per-model accuracy counts from comprehension question results."""
    summary = {mid: {"correct": 0, "total": 0, "accuracy": float("nan")} for mid in model_ids}

    for result in question_results:
        truth = bool(result.get("ground_truth"))
        predictions = result.get("predictions", {})
        for mid in model_ids:
            prob = predictions.get(mid, {}).get("probability")
            if prob is None:
                continue
            summary[mid]["total"] += 1
            if (prob >= 0.5) == truth:
                summary[mid]["correct"] += 1

    for mid, stats in summary.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]

    return summary


def find_world_report(game_dir: Path) -> Path | None:
    """Locate a world report JSON file for a game.

    Prefers world_report/turn_050_data.json, falls back to latest turn_*_data.json in
    world_report/, and finally to turn_*_data.json directly under the game dir.
    """
    preferred = game_dir / "world_report" / "turn_050_data.json"
    if preferred.exists():
        return preferred
    candidates = sorted((game_dir / "world_report").glob("turn_*_data.json"))
    if candidates:
        return candidates[-1]
    direct = sorted(game_dir.glob("turn_*_data.json"))
    return direct[-1] if direct else None


def load_world_report_data(report_path: Path) -> dict:
    with report_path.open() as f:
        return json.load(f)


def build_comprehension_prompt(questions: list[dict], world_report: str) -> str:
    """Build prompt requesting YES/NO answers."""
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}"
        for i, q in enumerate(questions)
    )

    instruction = f"""You are being checked on your understanding of the World Report below.
Answer each question with YES or NO.
Format your response exactly like this:
<<<ANSWERS>>>
YES
NO
...
<<<END>>>

List one answer per line for all {num_questions} questions, in order."""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. You are being checked on your understanding of the World Report below.

## World Report Data (Turn 50)
{world_report}

## Questions
{questions_text}

{instruction}"""


def parse_batch_yesno(response: str, num_questions: int) -> list[bool | None]:
    """Parse YES/NO answers from model response."""
    block_match = re.search(r'<<<ANSWERS>>>(.*?)<<<END>>>', response, re.DOTALL | re.IGNORECASE)
    content = block_match.group(1) if block_match else response
    content = content.strip()

    # Prefer line-wise parsing; fall back to whitespace tokenization
    tokens = [
        re.sub(r'^\d+[\.\:\)]\s*', '', line).strip()
        for line in content.splitlines()
        if line.strip()
    ]
    if not tokens and content:
        tokens = [tok for tok in re.split(r'\s+', content) if tok]

    answers: list[bool | None] = []
    for tok in tokens:
        t = tok.upper()
        if t in {"YES", "Y", "TRUE", "T", "1"}:
            answers.append(True)
        elif t in {"NO", "N", "FALSE", "F", "0"}:
            answers.append(False)
        if len(answers) >= num_questions:
            break

    answers.extend([None] * (num_questions - len(answers)))
    return answers[:num_questions]


async def query_model_batch_yesno_async(
    model: Any,
    prompt: str,
    num_questions: int,
    semaphore: asyncio.Semaphore,
    timeout: float | None = None,
    verbose: bool = False,
) -> tuple[str, list[float | None], float, str | None]:
    """Query a model for YES/NO answers."""
    async with semaphore:
        if verbose:
            print(f"\n{'='*80}\n[PROMPT to {model.id}]\n{'='*80}\n{prompt}\n{'='*80}")

        start = time.monotonic()
        error: str | None = None
        probs: list[float | None]

        try:
            loop = asyncio.get_running_loop()
            max_tokens = max(500, num_questions * 50)

            def make_call(m=model, p=prompt, mt=max_tokens):
                return m.get_response(p, temperature=0.0, max_tokens=mt)

            api_call = loop.run_in_executor(_thread_pool, make_call)
            if timeout:
                response = await asyncio.wait_for(api_call, timeout=timeout)
            else:
                response = await api_call

            if verbose:
                print(f"\n{'-'*80}\n[RESPONSE from {model.id}]\n{'-'*80}\n{response}\n{'-'*80}")

            answers = parse_batch_yesno(response, num_questions)
            probs = [1.0 if a is True else 0.0 if a is False else None for a in answers]

        except Exception as e:
            error = str(e)
            probs = [None] * num_questions

        latency = (time.monotonic() - start) * 1000
        return (model.id, probs, latency, error)


async def run_comprehension_checks(
    game_ids: list[str],
    models: list,
    rate_limiter: ProviderRateLimiter,
    data_dir: Path,
    timeout: float | None = None,
    verbose: bool = False,
    logger=None,
    batch_size: int | None = None,
) -> tuple[list[dict], dict[str, dict[str, float | int]]]:
    """Run comprehension checks for listed games."""
    all_results: list[dict] = []
    log = logger.info if logger else print
    warn = logger.warning if logger else print

    log(f"\nRunning comprehension checks for {len(game_ids)} games...")

    for game_id in game_ids:
        game_dir = data_dir / game_id
        # If data_dir itself is the game directory (e.g., --data-dir pointed directly at it)
        if not game_dir.exists() and data_dir.name == game_id:
            game_dir = data_dir
        report_path = find_world_report(game_dir)
        if not report_path:
            warn(f"  [comp] No world report found for {game_id}, skipping")
            continue

        try:
            report_data = load_world_report_data(report_path)
        except Exception as e:
            warn(f"  [comp] Failed to load report for {game_id}: {e}")
            continue

        # Compress first, then generate questions from the compressed data
        # This ensures questions match what the model will see
        compressed = compress_world_report(report_data)
        questions = generate_comprehension_questions(compressed, game_id)
        if not questions:
            warn(f"  [comp] No comprehension questions generated for {game_id}")
            continue

        world_report = json.dumps(compressed, separators=(",", ":"))

        # Chunk questions if requested
        chunks = []
        if batch_size and batch_size > 0:
            for i in range(0, len(questions), batch_size):
                chunks.append(questions[i:i + batch_size])
        else:
            chunks = [questions]

        log(f"  [comp] {game_id}: {len(questions)} questions in {len(chunks)} batch(es)")

        for chunk in chunks:
            prompt = build_comprehension_prompt(chunk, world_report)

            tasks = []
            for model in models:
                sem = rate_limiter.get_semaphore(model.provider_cls)
                tasks.append(
                    query_model_batch_yesno_async(
                        model=model,
                        prompt=prompt,
                        num_questions=len(chunk),
                        semaphore=sem,
                        timeout=timeout,
                        verbose=verbose,
                    )
                )

            batch_results = await asyncio.gather(*tasks)

            for q_idx, question in enumerate(chunk):
                predictions = {}
                for mid, probs, latency_ms, error in batch_results:
                    prob = probs[q_idx] if q_idx < len(probs) else None
                    predictions[mid] = {
                        "probability": prob,
                        "latency_ms": latency_ms,
                        "error": error if prob is None else None,
                    }

                all_results.append(
                    {
                        "question_id": question["question_id"],
                        "game_id": question["game_id"],
                        "template_id": question.get("template_id"),
                        "question_text": question["question_text"],
                        "difficulty": question.get("difficulty", {}),
                        "ground_truth": question["ground_truth"],
                        "predictions": predictions,
                    }
                )

    summary = summarize_comprehension_accuracy(all_results, [m.id for m in models])
    return all_results, summary
