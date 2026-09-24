#!/usr/bin/env -S uv run python3
"""Score a gathered dataset without simulation, providers or paper normalization."""
import argparse
import json
from pathlib import Path
from fbsim_benchmark.adapters import CachedForecast
from fbsim_benchmark.scoring.micropolis import PERCENTILE_LEVELS
from fbsim_core.metrics import compute_brier_score


def score_dataset(data, kind):
    questions = {q["question_id"]: q for q in data["questions"]}
    rows = []
    for forecast in data["forecasts"]:
        question = questions[forecast["question_id"]]
        value = forecast["probability" if kind == "binary" else "percentiles"]
        row = {"model_id": forecast["model_id"], "question_id": forecast["question_id"]}
        if value is None:
            rows.append(dict(row, status="unparsed", score=None))
            continue
        if kind == "binary":
            truth = question["answer"]
            if not isinstance(truth, bool):
                raise ValueError("Binary answer must be a realized boolean outcome")
            CachedForecast.parse(dict(schema_version="1", world="micropolis", metric="excess_brier", forecast=value, truth=float(truth)))
            score = compute_brier_score([value], [truth])
            metric = "realized_brier"
        else:
            record = CachedForecast.parse(dict(schema_version="1", world="micropolis", metric="quantile_crps", forecast=[value[k] for k in PERCENTILE_LEVELS], truth=[question["value"]]))
            score = record.score()
            metric = "raw_quantile_crps"
        rows.append(dict(row, status="scored", metric=metric, score=score))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=["binary", "continuous"], required=True)
    args = parser.parse_args()
    try:
        rows = score_dataset(json.loads(args.input.read_text()), args.kind)
        output = json.dumps(rows, allow_nan=False)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
