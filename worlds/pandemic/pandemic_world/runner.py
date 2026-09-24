"""Starsim SIR runner + serializer into the fbsim-core turn-major schema.

Each region is an independent SIR population (no inter-region transmission).
A region can be run with or without a vaccination campaign.
"""

from numbers import Integral

import starsim as ss
import numpy as np

N_AGENTS = 5_000
N_DAYS = 90
SIM_START = 2000
N_CONTACTS = 6
P_DEATH = 0.02
METRICS = ["cumulative_cases", "active_infections", "cumulative_deaths"]


def run_region(beta: float, vaccinate: bool, vax_efficacy: float,
               vax_day: int, vax_coverage: float, seed: int) -> dict:
    """Run one region's SIR sim; return per-day metric arrays.

    Days are integer timestep labels on the existing annual Starsim grid, not
    calendar days. A campaign after N_DAYS has no effect within this run.
    """
    if vaccinate and (isinstance(vax_day, bool) or
                      not isinstance(vax_day, Integral) or vax_day < 0):
        raise ValueError("vax_day must be a nonnegative integer timestep")
    sir = ss.SIR(init_prev=0.005, beta=beta, dur_inf=10, p_death=P_DEATH)
    pars = dict(
        n_agents=N_AGENTS,
        networks=dict(type="random", n_contacts=N_CONTACTS),
        diseases=sir,
        start=SIM_START,
        dur=N_DAYS,
        dt=1,
        verbose=0,
        rand_seed=seed,
    )
    if vaccinate and vax_day <= N_DAYS:
        vax = ss.simple_vx(efficacy=vax_efficacy, leaky=True)
        # CampaignDelivery takes calendar years and clamps out-of-range dates
        # to the nearest step. Couple the epoch to start and omit future events.
        pars["interventions"] = [ss.campaign_vx(
            product=vax, years=[SIM_START + vax_day], prob=vax_coverage,
        )]
    sim = ss.Sim(pars)
    sim.run()
    r = sim.results
    return {
        "cumulative_cases": np.array(r["sir"]["cum_infections"]).tolist(),
        "active_infections": np.array(r["sir"]["n_infected"]).tolist(),
        "cumulative_deaths": np.array(r["cum_deaths"]).tolist(),
    }


def to_world(region_series: dict[int, dict], names: dict[int, str]) -> dict:
    """Assemble game_data in the core TURN-MAJOR schema:
    time_series[metric][day][region_id] = value.
    """
    ts = {m: {} for m in METRICS}
    for rid, series in region_series.items():
        for m in METRICS:
            for day, val in enumerate(series[m]):
                ts[m].setdefault(str(day), {})[str(rid)] = val
    return {
        "time_series": ts,
        "civilizations": {str(rid): {"name": names[rid], "nation_id": str(rid)}
                          for rid in region_series},
    }
