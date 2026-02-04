"""
Runner for executing conditional question forks.

OPTIMIZATION: This module now extracts baseline (control) values directly from
existing savegames instead of running a redundant control fork. This cuts fork
execution time in half.

The key insight: We already have a complete game recording. The "control fork"
would just replay the recorded game, which is already captured in savegames.
We only need to run the *intervention* fork to see what changes.

Flow:
1. Load baseline player states from end_turn savegame (parse_player_states_for_conditional)
2. Run only the intervention fork (with modification applied)
3. Compare intervention outcome to savegame baseline

Uses subprocess isolation to avoid civrealm's global state issues. Each fork
runs in its own Python process.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .conditional_schema import (
    Condition,
    ConditionalQuestion,
    ConditionalQuestionBank,
    ConditionalResult,
    ForkOutcome,
)
from ..utils.savegame_parser import (
    find_local_savegame_for_turn,
    load_local_savegame,
    decompress_savegame_content,
    parse_player_states_for_conditional,
)


class ConditionalQuestionRunner:
    """
    Runs conditional question forks and resolves target questions.

    OPTIMIZATION: Extracts baseline values from existing savegames instead of
    running redundant control forks. Only runs intervention forks, cutting
    execution time in half.

    Uses subprocess isolation pattern (from test_parallel_forks.py) to
    avoid civrealm global state issues.
    """

    def __init__(
        self,
        recording_dir: str,
        base_seed: int,
        parallel: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the runner.

        Args:
            recording_dir: Path to recording directory with savegames/
            base_seed: Base seed of the recording
            parallel: Whether to run control/intervention forks in parallel (deprecated, only intervention runs now)
            verbose: Whether to print progress
        """
        self.recording_dir = recording_dir
        self.base_seed = base_seed
        self.parallel = parallel  # Kept for API compatibility but now only intervention runs
        self.verbose = verbose
        self._fork_counter = 0
        self._baseline_cache: dict[int, dict[int, dict]] = {}  # Cache baseline by end_turn

    def _get_username(self) -> str:
        """Extract username from recording directory (e.g., 's100' from '.../s100/')."""
        return Path(self.recording_dir).name

    def _load_baseline_from_savegame(self, end_turn: int) -> Optional[dict[int, dict]]:
        """
        Load baseline player states from existing savegame instead of running control fork.

        This is the key optimization: the original game recording already contains
        what happened at every turn. We don't need to replay it - just parse the
        savegame for the end_turn to get baseline values.

        Args:
            end_turn: Turn to load baseline for

        Returns:
            Dict mapping player_id to player state, or None if savegame not found
        """
        # Check cache first
        if end_turn in self._baseline_cache:
            return self._baseline_cache[end_turn]

        username = self._get_username()

        # Find savegame for the end turn
        savegame_name = find_local_savegame_for_turn(username, end_turn, self.recording_dir)
        if not savegame_name:
            if self.verbose:
                print(f"  Warning: No savegame found for turn {end_turn}")
            return None

        # Load and decompress
        result = load_local_savegame(savegame_name, self.recording_dir)
        if not result:
            if self.verbose:
                print(f"  Warning: Could not load savegame {savegame_name}")
            return None

        savegame_bytes, actual_filename = result

        try:
            content = decompress_savegame_content(savegame_bytes, actual_filename)
            player_states = parse_player_states_for_conditional(content)

            # Cache for reuse
            self._baseline_cache[end_turn] = player_states

            if self.verbose:
                print(f"  Loaded baseline from savegame {actual_filename}")

            return player_states

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error parsing savegame: {e}")
            return None

    def run_conditional_question(
        self,
        question: ConditionalQuestion,
        end_turn: int,
    ) -> ConditionalResult:
        """
        Run a single conditional question.

        OPTIMIZATION: Instead of running paired forks (control + intervention),
        we now:
        1. Load baseline from existing savegame (no fork needed!)
        2. Run only the intervention fork
        3. Compare intervention outcome to savegame baseline

        This cuts execution time in half since we skip the redundant control fork.

        Args:
            question: ConditionalQuestion to run
            end_turn: Turn to run forks until

        Returns:
            ConditionalResult with baseline and intervention outcomes
        """
        cond = question.condition
        checkpoint = question.checkpoint_turn

        if self.verbose:
            print(f"Running conditional question {question.conditional_id}")
            print(f"  Condition: {cond.description}")
            print(f"  Checkpoint: {checkpoint}, End: {end_turn}")

        # OPTIMIZATION: Load baseline from savegame instead of running control fork
        # The original recording already has what happened - no need to replay it
        baseline_states = self._load_baseline_from_savegame(end_turn)

        if baseline_states:
            control_outcome = ForkOutcome(
                fork_name="savegame_baseline",
                success=True,
                final_turn=end_turn,
                player_states=baseline_states,
                error=None,
            )
        else:
            # Fallback: if savegame parsing fails, mark as failed
            control_outcome = ForkOutcome(
                fork_name="savegame_baseline",
                success=False,
                final_turn=0,
                player_states={},
                error="Could not load baseline from savegame",
            )

        # Run only the intervention fork
        # Generate descriptive name from condition (no underscores allowed by Freeciv)
        fork_name = f"{cond.condition_type.replace('_', '')}{cond.value}p{cond.player_id}"
        intervention_mods = [self._condition_to_modification(cond)]
        intervention_script = self._create_fork_script(
            checkpoint_turn=checkpoint,
            end_turn=end_turn,
            modifications=intervention_mods,
            fork_name=fork_name,
        )
        self._fork_counter += 1

        intervention_result = self._run_fork_sequential(intervention_script, "intervention")
        intervention_outcome = self._parse_fork_result(intervention_result, "intervention")

        # Resolve target question against outcomes
        answer_control = None
        answer_intervention = None

        if control_outcome.success:
            answer_control = self._resolve_target_question(
                question, control_outcome.player_states
            )

        if intervention_outcome.success:
            answer_intervention = self._resolve_target_question(
                question, intervention_outcome.player_states
            )

        # Compute conditional effect
        conditional_effect = None
        if answer_control is not None and answer_intervention is not None:
            conditional_effect = 1.0 if answer_control != answer_intervention else 0.0

        if self.verbose:
            print(f"  Baseline answer: {answer_control}")
            print(f"  Intervention answer: {answer_intervention}")
            print(f"  Conditional effect: {conditional_effect}")

        return ConditionalResult(
            conditional_id=question.conditional_id,
            control_outcome=control_outcome,
            intervention_outcome=intervention_outcome,
            answer_control=answer_control,
            answer_intervention=answer_intervention,
            conditional_effect=conditional_effect,
            computed_at=datetime.now().isoformat() + "Z",
        )

    def run_batch(
        self,
        bank: ConditionalQuestionBank,
    ) -> ConditionalQuestionBank:
        """
        Run all questions in a conditional question bank.

        OPTIMIZATION: Runs ONE fork per CONDITION, not per question.
        Multiple questions sharing the same condition reuse the fork result.

        Flow:
        1. Load baseline from savegame (once, cached)
        2. Group questions by condition_id
        3. For each unique condition:
           - Run ONE intervention fork
           - Cache the result
        4. For each question:
           - Look up cached fork result for its condition
           - Resolve question against cached states (instant)

        This makes execution time O(conditions) instead of O(questions).

        Args:
            bank: ConditionalQuestionBank with questions to run

        Returns:
            Updated bank with results populated
        """
        # Step 1: Load baseline from savegame (shared across all questions)
        baseline_states = self._load_baseline_from_savegame(bank.end_turn)
        if baseline_states:
            baseline_outcome = ForkOutcome(
                fork_name="savegame_baseline",
                success=True,
                final_turn=bank.end_turn,
                player_states=baseline_states,
                error=None,
            )
        else:
            baseline_outcome = ForkOutcome(
                fork_name="savegame_baseline",
                success=False,
                final_turn=0,
                player_states={},
                error="Could not load baseline from savegame",
            )

        # Step 2: Group questions by condition_id
        questions_by_condition: dict[str, list[ConditionalQuestion]] = {}
        for question in bank.questions:
            cond_id = question.condition.condition_id
            if cond_id not in questions_by_condition:
                questions_by_condition[cond_id] = []
            questions_by_condition[cond_id].append(question)

        if self.verbose:
            print(f"\nGrouped {len(bank.questions)} questions into {len(questions_by_condition)} conditions")

        # Step 3: Run ONE fork per condition, cache results
        intervention_cache: dict[str, ForkOutcome] = {}

        for i, (cond_id, questions) in enumerate(questions_by_condition.items()):
            cond = questions[0].condition  # All questions in group share same condition
            checkpoint = questions[0].checkpoint_turn

            if self.verbose:
                print(f"\n[Condition {i + 1}/{len(questions_by_condition)}] {cond.description}")
                print(f"  Running fork for {len(questions)} questions...")

            # Run the intervention fork ONCE for this condition
            # Generate descriptive name from condition (no underscores allowed by Freeciv)
            fork_name = f"{cond.condition_type.replace('_', '')}{cond.value}p{cond.player_id}"
            intervention_mods = [self._condition_to_modification(cond)]
            intervention_script = self._create_fork_script(
                checkpoint_turn=checkpoint,
                end_turn=bank.end_turn,
                modifications=intervention_mods,
                fork_name=fork_name,
            )
            self._fork_counter += 1

            intervention_result = self._run_fork_sequential(intervention_script, f"intervention_{cond_id}")
            intervention_outcome = self._parse_fork_result(intervention_result, f"intervention_{cond_id}")

            # Cache for all questions with this condition
            intervention_cache[cond_id] = intervention_outcome

            if self.verbose:
                if intervention_outcome.success:
                    print(f"  Fork completed successfully")
                else:
                    print(f"  Fork failed: {intervention_outcome.error}")

        # Step 4: Resolve all questions against cached fork results (instant)
        if self.verbose:
            print(f"\nResolving {len(bank.questions)} questions against cached fork results...")

        for question in bank.questions:
            cond_id = question.condition.condition_id
            intervention_outcome = intervention_cache[cond_id]

            # Resolve answers
            answer_control = None
            answer_intervention = None

            if baseline_outcome.success:
                answer_control = self._resolve_target_question(
                    question, baseline_outcome.player_states
                )

            if intervention_outcome.success:
                answer_intervention = self._resolve_target_question(
                    question, intervention_outcome.player_states
                )

            # Compute conditional effect
            conditional_effect = None
            if answer_control is not None and answer_intervention is not None:
                conditional_effect = 1.0 if answer_control != answer_intervention else 0.0

            # Store result
            bank.results[question.conditional_id] = ConditionalResult(
                conditional_id=question.conditional_id,
                control_outcome=baseline_outcome,
                intervention_outcome=intervention_outcome,
                answer_control=answer_control,
                answer_intervention=answer_intervention,
                conditional_effect=conditional_effect,
                computed_at=datetime.now().isoformat() + "Z",
            )

        return bank

    def _condition_to_modification(self, condition: Condition) -> dict:
        """Convert a Condition to a ForkManager modification dict."""
        return {
            "type": condition.condition_type,
            "player_id": condition.player_id,
            "value": condition.value,
        }

    def _create_fork_script(
        self,
        checkpoint_turn: int,
        end_turn: int,
        modifications: list[dict],
        fork_name: str,
    ) -> str:
        """Generate Python script to run a fork in subprocess."""
        mods_json = json.dumps(modifications)

        return f'''
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(".") / "src"))
from civrealm.forking import ForkManager

manager = ForkManager("{self.recording_dir}", {self.base_seed})
fork = manager.create_fork(
    checkpoint_turn={checkpoint_turn},
    modifications={mods_json},
    fork_name="{fork_name}"
)
result = manager.run_fork(fork, {end_turn})

# Output result as JSON on the last line
output = {{
    "success": result.success,
    "final_turn": result.final_turn,
    "player_states": result.player_states,
    "error": result.error
}}
print("RESULT_JSON:" + json.dumps(output))
'''

    def _run_forks_parallel(
        self,
        control_script: str,
        intervention_script: str,
    ) -> tuple[dict, dict]:
        """Run control and intervention forks in parallel."""
        cwd = self._get_cwd()

        # Launch both subprocesses simultaneously
        proc_control = subprocess.Popen(
            [sys.executable, "-c", control_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        proc_intervention = subprocess.Popen(
            [sys.executable, "-c", intervention_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        # Wait for both
        stdout_ctrl, stderr_ctrl = proc_control.communicate()
        stdout_intv, stderr_intv = proc_intervention.communicate()

        return (
            {"stdout": stdout_ctrl, "stderr": stderr_ctrl, "returncode": proc_control.returncode},
            {"stdout": stdout_intv, "stderr": stderr_intv, "returncode": proc_intervention.returncode},
        )

    def _run_fork_sequential(self, script: str, name: str) -> dict:
        """Run a single fork."""
        cwd = self._get_cwd()

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=cwd,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def _get_cwd(self) -> str:
        """Get working directory for subprocesses."""
        # Try to find civbench root (parent of src/)
        current = Path(self.recording_dir).resolve()
        while current.parent != current:
            if (current / "src" / "civrealm").exists():
                return str(current)
            current = current.parent

        # Fall back to current directory
        return str(Path.cwd())

    def _parse_fork_result(self, result: dict, fork_name: str) -> ForkOutcome:
        """Parse subprocess result into ForkOutcome."""
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        returncode = result.get("returncode", 1)

        # Find the JSON result line
        for line in stdout.split('\n'):
            if line.startswith("RESULT_JSON:"):
                try:
                    data = json.loads(line[len("RESULT_JSON:"):])
                    return ForkOutcome(
                        fork_name=fork_name,
                        success=data.get("success", False),
                        final_turn=data.get("final_turn", 0),
                        player_states={
                            int(k): v for k, v in data.get("player_states", {}).items()
                        },
                        error=data.get("error"),
                    )
                except json.JSONDecodeError as e:
                    return ForkOutcome(
                        fork_name=fork_name,
                        success=False,
                        final_turn=0,
                        player_states={},
                        error=f"JSON parse error: {e}",
                    )

        # No result found
        error_msg = f"No result from {fork_name}"
        if returncode != 0:
            error_msg += f" (exit {returncode})"
        if stderr:
            error_msg += f": {stderr[-500:]}"

        return ForkOutcome(
            fork_name=fork_name,
            success=False,
            final_turn=0,
            player_states={},
            error=error_msg,
        )

    def _resolve_target_question(
        self,
        question: ConditionalQuestion,
        player_states: dict[int, dict],
    ) -> bool | None:
        """
        Resolve target question using player states.

        Now supports both fork-derived states and savegame-parsed states.
        The savegame parser provides: gold, techs, cities, population, units, wonders, landarea.

        Args:
            question: ConditionalQuestion containing target template
            player_states: Player states from fork outcome or savegame

        Returns:
            Boolean answer or None if cannot resolve
        """
        template_id = question.target_template_id
        params = question.target_parameters

        if template_id in ["treasury_comparative"]:
            return self._resolve_comparative(params, player_states, "gold")

        elif template_id in ["score_comparative"]:
            # Score not directly in savegame, but we can compute a proxy from components
            # Try score first (from fork), fall back to cities+techs (from savegame)
            result = self._resolve_comparative(params, player_states, "score")
            if result is None:
                # Compute score proxy: techs * 10 + cities * 5 + population / 100
                result = self._resolve_comparative_with_score_proxy(params, player_states)
            return result

        elif template_id in ["tech_comparative"]:
            # Now available from savegame parser
            return self._resolve_comparative(params, player_states, "techs")

        elif template_id in ["population_comparative"]:
            # Now available from savegame parser
            return self._resolve_comparative(params, player_states, "population")

        elif template_id in ["cities_comparative"]:
            return self._resolve_comparative(params, player_states, "cities")

        elif template_id == "score_rank_1":
            return self._resolve_rank_1(params, player_states)

        elif template_id == "government_at":
            # Government state not in basic player_states
            # Would need to parse savegame for accurate answer
            return None

        elif template_id == "tech_discovered":
            # Tech discovery not in basic player_states
            return None

        return None

    def _resolve_comparative_with_score_proxy(
        self,
        params: dict[str, Any],
        player_states: dict[int, dict],
    ) -> bool | None:
        """Resolve comparative question using a score proxy computed from components."""
        player_a = params.get("player_id_a")
        player_b = params.get("player_id_b")

        if player_a is None or player_b is None:
            return None

        state_a = player_states.get(player_a, {})
        state_b = player_states.get(player_b, {})

        def compute_score_proxy(state: dict) -> int:
            """Compute a score proxy from available fields."""
            return (
                state.get("techs", 0) * 10 +
                state.get("cities", 0) * 5 +
                state.get("population", 0) // 100 +
                state.get("wonders", 0) * 20
            )

        score_a = compute_score_proxy(state_a)
        score_b = compute_score_proxy(state_b)

        return score_a > score_b

    def _resolve_comparative(
        self,
        params: dict[str, Any],
        player_states: dict[int, dict],
        field: str,
    ) -> bool | None:
        """Resolve a comparative question (A > B)."""
        player_a = params.get("player_id_a")
        player_b = params.get("player_id_b")

        if player_a is None or player_b is None:
            return None

        state_a = player_states.get(player_a, {})
        state_b = player_states.get(player_b, {})

        value_a = state_a.get(field)
        value_b = state_b.get(field)

        if value_a is None or value_b is None:
            return None

        return value_a > value_b

    def _resolve_rank_1(
        self,
        params: dict[str, Any],
        player_states: dict[int, dict],
    ) -> bool | None:
        """Resolve a rank #1 question."""
        player_id = params.get("player_id")
        if player_id is None:
            return None

        # Sort by score to find rank
        scores = [
            (pid, state.get("score", 0))
            for pid, state in player_states.items()
            if state.get("is_alive", False)
        ]

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        rank_1_player = scores[0][0]

        return rank_1_player == player_id
