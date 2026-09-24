"""Unit tests for the determinism check's row comparison.

Covers the difference-locating helpers, which are what turn "the files differ"
into a turn and field name. Nothing here runs the simulation or touches disk.
"""

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_module():
    """Import the check script, which is not on the package path."""
    spec = importlib.util.spec_from_file_location(
        "check_determinism", SCRIPTS_DIR / "check_determinism.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_rows_have_no_difference():
    m = load_module()
    rows = [{"tick": 15, "cityPop": 100}, {"tick": 31, "cityPop": 110}]
    assert m.first_row_difference(rows, [dict(r) for r in rows]) is None
    assert m.differing_fields(rows, [dict(r) for r in rows]) == {}


def test_first_difference_reports_row_field_and_values():
    m = load_module()
    a = [{"tick": 15, "cityPop": 100}, {"tick": 31, "cityPop": 110}]
    b = [{"tick": 15, "cityPop": 100}, {"tick": 31, "cityPop": 999}]
    assert m.first_row_difference(a, b) == (1, "cityPop", 110, 999)


def test_first_difference_scans_fields_in_row_order():
    # Two fields differ in the same row; the leftmost one is the one named,
    # since that is the closest thing to a cause in the engine's log order.
    m = load_module()
    a = [{"cityScore": 700, "cityPop": 100}]
    b = [{"cityScore": 711, "cityPop": 999}]
    assert m.first_row_difference(a, b)[1] == "cityScore"


def test_row_count_mismatch_is_a_difference():
    m = load_module()
    a = [{"tick": 15}, {"tick": 31}]
    assert m.first_row_difference(a, a[:1]) == (1, "<row count>", 2, 1)


def test_a_field_present_in_only_one_run_differs():
    m = load_module()
    a = [{"tick": 15}]
    b = [{"tick": 15, "extra": 1}]
    assert m.first_row_difference(a, b) == (0, "extra", None, 1)


def test_differing_fields_counts_rows_per_field():
    m = load_module()
    a = [{"cityPop": 1, "cityScore": 9}, {"cityPop": 2, "cityScore": 9}]
    b = [{"cityPop": 5, "cityScore": 9}, {"cityPop": 6, "cityScore": 8}]
    assert m.differing_fields(a, b) == {"cityPop": 2, "cityScore": 1}


def test_differing_fields_ignores_rows_only_one_run_has():
    # The count is over the shared prefix; the row-count gap is reported
    # separately by first_row_difference.
    m = load_module()
    a = [{"cityPop": 1}, {"cityPop": 2}]
    assert m.differing_fields(a, a[:1]) == {}
