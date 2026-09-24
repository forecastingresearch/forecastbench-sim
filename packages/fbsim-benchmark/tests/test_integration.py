"""Installed-package ownership, contracts and safe core interoperability."""
import importlib.metadata as metadata
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator
from fbsim_benchmark.adapters import CachedForecast
from fbsim_core.metrics import compute_brier_score

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

def test_single_core_owner():
    owners = metadata.packages_distributions()
    assert owners['fbsim_core'] == ['fbsim-core']
    assert owners['fbsim_benchmark'] == ['fbsim-benchmark']
    assert not any(str(p).startswith('fbsim_core/') for p in metadata.files('fbsim-benchmark'))
    assert any(r.startswith('fbsim-core==0.1.0') for r in metadata.requires('fbsim-benchmark'))

def test_distinct_score_meanings():
    rec = json.loads((ROOT/'fixtures/freeciv.json').read_text())
    assert CachedForecast.parse(rec).score() == pytest.approx(.04)
    assert compute_brier_score([.4], [True]) == pytest.approx(.36)

@pytest.mark.parametrize('world,expected', [('freeciv',.04),('micropolis',.09),('starsim',.01)])
def test_installed_fixtures(world, expected):
    path = ROOT/'fixtures'/f'{world}.json'
    result = subprocess.run([sys.executable, '-m', 'fbsim_benchmark', 'validate', '--input', str(path)], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['score'] == pytest.approx(expected)

def test_metric_shape_schema():
    schema = json.loads((ROOT/'docs/cached-forecast-v1.schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    base = json.loads((ROOT/'fixtures/freeciv.json').read_text())
    for metric in ('excess_brier', 'excess_bits', 'quantile_crps'):
        for forecast in (.4, [0,1,2,3,4]):
            for truth in (.6, [0,1]):
                record = dict(base, metric=metric, forecast=forecast, truth=truth)
                valid = isinstance(forecast,list) and isinstance(truth,list) if metric=='quantile_crps' else not isinstance(forecast,list) and not isinstance(truth,list)
                assert validator.is_valid(record) == valid
                if valid:
                    CachedForecast.parse(record)
                else:
                    with pytest.raises(ValueError): CachedForecast.parse(record)

def test_core_interfaces_remain_importable():
    from fbsim_core.questions import schema, resolver, io
    from fbsim_core.conditional import schema as conditional_schema
    assert all(module.__name__.startswith('fbsim_core.') for module in [schema, resolver, io, conditional_schema])
