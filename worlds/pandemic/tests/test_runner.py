"""Tiny offline regressions through run_region and real Starsim initialization."""
import numpy as np
import pytest
import starsim as ss

from pandemic_world import runner


@pytest.fixture
def tiny_runner(monkeypatch):
    monkeypatch.setattr(runner, "N_AGENTS", 100)


def run(day, *, vaccinate=True, coverage=0.8):
    return runner.run_region(0.05, vaccinate, 0.7, day, coverage, seed=1)


@pytest.mark.parametrize("day", [0, 5, 15, 25, 35, 45, 60, 90, np.int64(21)])
@pytest.mark.parametrize("epoch", [2000, 2040])
def test_real_initialized_schedule(monkeypatch, tiny_runner, day, epoch):
    monkeypatch.setattr(runner, "SIM_START", epoch)
    observed = []

    def initialize_only(sim):
        sim.init()
        iv = sim.interventions[0]
        observed.append(iv.timepoints.tolist())
        assert sim.t.yearvec[day] == epoch + day
        assert iv.prob.tolist() == [0.8]
        assert not np.asarray(iv.vaccinated).any()

    monkeypatch.setattr(ss.Sim, "run", initialize_only)
    run(day)
    assert observed == [[day]]


@pytest.mark.parametrize("day", [-1, 2.5, float("nan"), float("inf"), True, "25"])
def test_invalid_campaign_day(tiny_runner, day):
    with pytest.raises(ValueError, match="nonnegative integer timestep"):
        run(day)


def test_timing_and_noop_controls(monkeypatch, tiny_runner):
    monkeypatch.setattr(runner, "N_AGENTS", 500)
    monkeypatch.setattr(runner, "N_DAYS", 12)
    real_run = ss.Sim.run
    simulations = []

    def capture_run(sim):
        real_run(sim)
        simulations.append(sim)

    monkeypatch.setattr(ss.Sim, "run", capture_run)
    control = run(0, vaccinate=False)
    assert control["cumulative_cases"][-1] > control["cumulative_cases"][5] > 0
    # Test the actual public API and serialized outputs, not a reconstructed sim.
    assert run(5, coverage=0.0) == control
    assert run(999) == control
    assert len(simulations[-1].interventions) == 0
    for day in [0, 5, 10, 12]:
        vaccinated = run(day)
        sim = simulations[-1]
        iv = sim.interventions[0]
        recipients = np.asarray(iv.vaccinated)
        assert recipients.any()  # No vacuous timing/uptake success.
        assert np.all(np.asarray(iv.ti_vaccinated)[recipients] == day)
        assert np.all(np.asarray(iv.n_doses)[recipients] == 1)
        assert np.allclose(np.asarray(sim.diseases.sir.rel_sus)[recipients], 0.3)
        for metric in runner.METRICS:
            assert len(vaccinated[metric]) == 13
            assert vaccinated[metric][:day] == control[metric][:day]
        if day == 5:
            assert vaccinated["cumulative_cases"][day] != control["cumulative_cases"][day]
    world = runner.to_world({0: control}, {0: "Example"})
    assert world["time_series"]["cumulative_cases"]["12"]["0"] == control["cumulative_cases"][12]
