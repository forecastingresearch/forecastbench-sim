"""Boundary regressions; historical parsing and scientific scores remain separate."""
from importlib import metadata, resources
import json
from pathlib import Path
import subprocess
import sys

import pytest
from fbsim_benchmark.parsing.registry import parse_response


@pytest.mark.parametrize('token', ['1.5', '10', '0.8e2', '-0.1', '2', '1e309', 'NaN', 'Infinity', 'true', 'false', '"0.4"', 'null', '1.0x', '0.8.2'])
def test_strict_binary_rejects_invalid_whole_tokens(token):
    with pytest.raises(ValueError):
        parse_response('starsim', 'binary', '{"a":' + token + '}', keys=['a'])


@pytest.mark.parametrize('token,expected', [('0', 0.), ('1', 1.), ('0.8', .8), ('8e-1', .8), ('0.8e-2', .008)])
def test_strict_binary_valid_numbers(token, expected):
    assert parse_response('starsim', 'binary', '{"a":' + token + '}', keys=['a']) == {'a': expected}


@pytest.mark.parametrize('text', ['{"a":0.4,"a":0.8}', '{"b":0.4}', '{"a":0.4} then {"a":1.5}', '"a":0.4'])
def test_strict_binary_rejects_ambiguous_or_incomplete_answers(text):
    with pytest.raises(ValueError):
        parse_response('starsim', 'binary', text, keys=['a'])


@pytest.mark.parametrize('token,expected', [('1.5', 1.), ('10', 1.), ('0.8e2', .8), ('true', 1.)])
def test_historical_binary_defects_remain_explicit(token, expected):
    assert parse_response('starsim', 'binary-historical', '{"a":' + token + '}', keys=['a']) == {'a': expected}


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'fbsim_benchmark', *args], capture_output=True, text=True)


def failure(result):
    assert result.returncode != 0
    assert not result.stdout
    assert 'error:' in result.stderr
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('world,fmt,text,options', [
    ('starsim', 'continuous', '{"a":1,"b":2,"c":3,"d":4,"e":1e309}', {'keys':list('abcde')}),
    ('micropolis', 'continuous-single', '{"p10":1,"p25":2,"p50":3,"p75":4,"p90":1e309}', {'quiet':True}),
    ('starsim', 'binary', '{"a":1.5}', {'keys':['a']}),
])
def test_cli_invalid_parser_output(tmp_path, world, fmt, text, options):
    path = tmp_path/'response.txt';path.write_text(text)
    failure(cli('parse', '--world', world, '--format', fmt, '--input', str(path), '--options', json.dumps(options)))


@pytest.mark.parametrize('world', ['freeciv', 'micropolis', 'starsim'])
def test_cli_overflow_score(tmp_path, world):
    path = tmp_path/'forecast.json'
    path.write_text(json.dumps(dict(schema_version='1', world=world, metric='quantile_crps', forecast=[-1e308]*5, truth=[1e308])))
    failure(cli('validate', '--input', str(path)))


def test_cli_options_and_input_errors(tmp_path):
    path=tmp_path/'response.txt';path.write_text('0.4')
    failure(cli('parse', '--world', 'micropolis', '--format', 'binary-single', '--input', str(path), '--options', '[]'))
    failure(cli('validate', '--input', str(tmp_path/'missing.json')))


def test_noneditable_installed_packages():
    for name in ['fbsim-core', 'fbsim-benchmark']:
        direct = json.loads(metadata.distribution(name).read_text('direct_url.json'))
        assert not direct.get('dir_info', {}).get('editable', False)
    import fbsim_core, fbsim_benchmark
    assert Path(fbsim_core.__file__).is_relative_to(sys.prefix)
    assert Path(fbsim_benchmark.__file__).is_relative_to(sys.prefix)


def test_installed_documentation_matches_checkout():
    docs = Path(__file__).resolve().parents[1]/'docs'
    for path in docs.iterdir():
        assert resources.files('fbsim_benchmark').joinpath('docs', path.name).read_bytes() == path.read_bytes()
    result = cli('docs', '--name', 'PROVENANCE.json')
    assert result.returncode == 0
    provenance=json.loads(json.loads(result.stdout)['text'])
    assert all(name in provenance['attribution'] for name in ['freeciv','micropolis','starsim'])


def test_cli_native_diagnostics_do_not_corrupt_json(tmp_path):
    path = tmp_path/'response.txt'; path.write_text('not an answer')
    result = cli('parse', '--world', 'micropolis', '--format', 'binary-single', '--input', str(path))
    assert result.returncode == 0
    assert json.loads(result.stdout) is None
    assert result.stderr
