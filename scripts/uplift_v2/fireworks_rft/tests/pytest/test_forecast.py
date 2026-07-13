"""Fireworks RFT evaluator for the FB-Sim dense-vs-hard A/B (free <16B arm).

Reward = 1 - (p - target)^2, where p is parsed from the model's last
<probability> tag (after </think> if present) and target is the per-row
ground_truth: the hard 0/1 label in the hard-arm dataset, the Monte-Carlo
frequency p_mc in the dense-arm dataset. Missing/invalid tag scores 0.0
(their required score range is [0,1]; this mirrors OpenForecaster's
format penalty being maximal).

Select the arm by pointing input_dataset at forecast_hard.jsonl or
forecast_dense.jsonl (env var ARM for local runs; the RFT job pins the
dataset at creation time).
"""
import os
import re

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

ARM = os.environ.get("ARM", "dense")
PROB_RE = re.compile(r"<probability>\s*([0-9.eE+-]+)\s*</probability>")


@evaluation_test(
    input_dataset=[f"development/forecast_{ARM}.jsonl"],
    completion_params=[{
        "temperature": 1.0,
        "max_tokens": 8192,
        "model": "fireworks_ai/accounts/fireworks/models/qwen3-4b",
    }],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
    max_dataset_rows=5,
)
def test_forecast(row: EvaluationRow, **kwargs) -> EvaluationRow:
    target = float(row.ground_truth)
    text = str(row.messages[-1].content or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = PROB_RE.findall(text)
    score, reason = 0.0, "missing probability tag"
    if matches:
        try:
            p = float(matches[-1])
            if -0.01 <= p <= 1.01:
                p = min(1.0, max(0.0, p))
                score = 1.0 - (p - target) ** 2
                reason = f"p={p:.3f} target={target:.3f}"
            else:
                reason = f"out-of-range p={p}"
        except ValueError:
            reason = "unparseable float"
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
