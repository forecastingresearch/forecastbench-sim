#!/usr/bin/env -S uv run python3
"""Run Micropolis city simulations via CitySimulation.run.

Runs every (city, disasters) combination in the config file, writing a plot of
each run to data/micropolis/runs/<city>/ alongside its log/events files.

Usage:
    scripts/run_sim.py
    scripts/run_sim.py my_config.json
    scripts/run_sim.py my_config.json --seed 7 --quiet --no-plot
    scripts/run_sim.py --cities kyoto bruce --disasters false
"""

import argparse

from micropolis_world.city_sim import CitySimulation
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
    scenarios_from,
)
from micropolis_world.plot_sim import save_run_plot


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip plotting the simulation results after running",
    )
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    turns = cfg.get_int("turns")
    scenarios = scenarios_from(cfg, args.cities, args.disasters)

    for i, (city, disasters) in enumerate(scenarios):
        sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
        print(
            f"[{i + 1}/{len(scenarios)}] running {sim.get_id_str()} for {turns} turns"
        )
        sim.run(nturns=turns, quiet=args.quiet)

        if args.plot:
            save_run_plot(sim)


if __name__ == "__main__":
    main()
