"""Suite-wide guard: nothing here may make a real network call.

Every test that exercises the prompting path stubs the client by replacing a
name on llm_backend. Patching the wrong target used to be silent — the real
client ran, the suite hit the live API and a "no such model" test asserted
against a genuine 400 — so the transport is blocked outright and a missed stub
now fails as a RuntimeError naming the URL it tried to reach.
"""

import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def blocked(*args, **kwargs):
        url = kwargs.get("url") or (args[1] if len(args) > 1 else "?")
        raise RuntimeError(
            f"test tried to make a real HTTP request to {url} — stub the LLM "
            f"client by monkeypatching acompletion/completion on "
            f"micropolis_world.llm_backend"
        )

    monkeypatch.setattr(httpx, "post", blocked)
    monkeypatch.setattr(httpx.Client, "request", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "request", blocked)


@pytest.fixture(autouse=True)
def _block_engine_and_sockets(monkeypatch):
    import socket
    from micropolis_world.city_sim import CitySimulation
    def blocked(*args, **kwargs):
        raise AssertionError("Synthetic tests must not launch engines or network calls")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(CitySimulation, "run", blocked)


@pytest.fixture
def synthetic_corpus(monkeypatch, tmp_path):
    """Invented cached trajectories; this is not live engine validation."""
    import micropolis_world.module_globals as g
    from micropolis_world.city_sim import CitySimulation
    from test_report import make_row
    def load(self, nturns, quiet=True):
        self.log_data = [make_row(t) for t in range(nturns)]
        self.events_data = []
        self.nturns = nturns
    monkeypatch.setattr(g, "DATA_DIR", tmp_path)
    monkeypatch.setattr(g, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(CitySimulation, "run_if_needed_and_load", load)
