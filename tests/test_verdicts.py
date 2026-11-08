"""The frozen verdict rules, exercised at their boundaries."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_results", Path(__file__).resolve().parent.parent / "scripts" / "make_results.py"
)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def test_h1_boundaries():
    assert mr.verdict_h1(0.60, 0.70) == "CONFIRMED"   # difference exactly 0.10
    assert mr.verdict_h1(0.60, 0.71) == "REFUTED"
    assert mr.verdict_h1(0.60, 0.50) == "CONFIRMED"
    assert mr.verdict_h1(None, 0.5) == "INCONCLUSIVE"
    assert mr.verdict_h1(0.5, None) == "INCONCLUSIVE"


def test_h2_boundaries():
    assert mr.verdict_h2(0.70, 0.75, 0.65) == "CONFIRMED"   # exactly plus/minus 5 pp
    assert mr.verdict_h2(0.70, 0.751, 0.70) == "REFUTED"    # B outside
    assert mr.verdict_h2(0.70, 0.70, 0.649) == "REFUTED"    # C outside
    assert mr.verdict_h2(0.70, None, 0.70) == "INCONCLUSIVE"


def test_h3_boundaries():
    assert mr.verdict_h3(1.25) == "CONFIRMED"
    assert mr.verdict_h3(1.30) == "CONFIRMED"
    assert mr.verdict_h3(1.00) == "REFUTED"
    assert mr.verdict_h3(0.90) == "REFUTED"
    assert mr.verdict_h3(1.10) == "INCONCLUSIVE"
    assert mr.verdict_h3(None) == "INCONCLUSIVE"
