"""Offline CLI: successful output is always finite, standard JSON."""
import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from importlib.resources import files
from pathlib import Path

from .adapters import CachedForecast, load_record
from .parsing.registry import parse_response


def main():
    parser = argparse.ArgumentParser(description="Offline cached-forecast scoring; never simulates or calls models.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    validate = commands.add_parser("validate")
    validate.add_argument("--input", required=True)
    parse = commands.add_parser("parse")
    parse.add_argument("--world", required=True)
    parse.add_argument("--format", required=True)
    parse.add_argument("--input", required=True)
    parse.add_argument("--options", default="{}", help="JSON object of parser arguments, e.g. keys or labels")
    docs = commands.add_parser("docs")
    docs.add_argument("--name", choices=["PROVENANCE.json", "SEMANTICS.md", "NATIVE_INTERFACES.md", "cached-forecast-v1.schema.json", "WORKFLOWS.md"], default="SEMANTICS.md")
    args = parser.parse_args()
    try:
        if args.command == "smoke":
            record = CachedForecast.parse(dict(schema_version="1", world="starsim", metric="excess_brier", forecast=.4, truth=.6))
            assert math.isclose(record.score(), .04)
            result = dict(status="PASS", scope="synthetic public fixture", schema_version="1", network=False, simulations=0)
        elif args.command == "parse":
            options = json.loads(args.options)
            if not isinstance(options, dict):
                raise ValueError("--options must be a JSON object")
            with redirect_stdout(sys.stderr):
                result = parse_response(args.world, args.format, Path(args.input).read_text(), **options)
        elif args.command == "docs":
            result = dict(name=args.name, text=files("fbsim_benchmark").joinpath("docs", args.name).read_text())
        else:
            record = load_record(args.input)
            result = dict(schema_version="1", world=record.world, metric=record.metric, score=record.score())
        # Serialize before writing anything: a nonfinite parser output or an
        # arithmetic overflow is a failure, never successful Infinity/NaN JSON.
        output = json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, OSError, OverflowError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
