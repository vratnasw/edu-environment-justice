"""Smoke test: every model stub must import + expose run().

Phase A only verifies the scaffold parses; Phase B will add real tests.
"""
from __future__ import annotations


def test_modules_import():
    from src.models import ej_funding_gap
    from src.models import pollution_iv
    from src.models import tri_proximity
    from src.models import ej_paper_assembly
    assert hasattr(ej_funding_gap, 'run'), 'ej_funding_gap.run missing'
    assert hasattr(pollution_iv, 'run'), 'pollution_iv.run missing'
    assert hasattr(tri_proximity, 'run'), 'tri_proximity.run missing'
    assert hasattr(ej_paper_assembly, 'run'), 'ej_paper_assembly.run missing'
