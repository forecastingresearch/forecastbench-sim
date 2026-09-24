#!/usr/bin/env -S uv run python3
"""Print the model-facing world report for already-run Micropolis simulations.

Loads each sim's log/events files from disk (does not run the sim), for every
(city, disasters) combination in the config file.

Usage:
    scripts/gen_report.py
    scripts/gen_report.py my_config.json --seed 7
    scripts/gen_report.py --cities kyoto --disasters false
"""

import argparse
import sys

from micropolis_world.city_sim import CitySimulation
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
    scenarios_from,
)
from micropolis_world.report import gen_world_report


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    report_turn = cfg.get_int("report_turn")
    history_freq = cfg.get_int("history_freq")
    history_length = cfg.get_int_or("history_length", -1)
    snapshot_only = cfg.get_bool_or("snapshot_only_report", False)
    report_effectiveness = cfg.get_bool_or("report_effectiveness", False)
    censor_city_funds = cfg.get_bool_or("censorCityFunds", True)
    label = cfg.get_label(args.label)
    scenarios = scenarios_from(cfg, args.cities, args.disasters)

    for city, disasters in scenarios:
        sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
        try:
            sim.load_from_disk()
        except FileNotFoundError as e:
            print(
                f"[error] {e}\nDid you run the simulation first? "
                f"e.g. scripts/run_sim.py {args.config or ''}".rstrip(),
                file=sys.stderr,
            )
            sys.exit(1)

        assert sim.log_data is not None
        # A negative report_turn counts back from the last logged turn.
        turn = report_turn if report_turn >= 0 else len(sim.log_data) + report_turn

        print("=" * 70)
        print(f"{sim.get_id_str()} — turn {turn}")
        print("=" * 70)
        print(
            gen_world_report(
                sim,
                turn=turn,
                history_freq=history_freq,
                label=label,
                snapshot_only=snapshot_only,
                history_length=history_length,
                report_effectiveness=report_effectiveness,
                censor_city_funds=censor_city_funds,
            )
        )
        print()


if __name__ == "__main__":
    main()
