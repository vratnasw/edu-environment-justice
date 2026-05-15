"""Smoke tests.

Phase A: every model stub imports + exposes run().
Phase B: every module's run() returns a dict; when run on a synthetic panel
with enough rows, the funding-gap coefficient is finite and non-zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# Import smoke (kept from Phase A)
# --------------------------------------------------------------------------

def test_modules_import():
    from src.models import ej_funding_gap
    from src.models import pollution_iv
    from src.models import tri_proximity
    from src.models import ej_paper_assembly
    for mod, name in (
        (ej_funding_gap, "ej_funding_gap"),
        (pollution_iv, "pollution_iv"),
        (tri_proximity, "tri_proximity"),
        (ej_paper_assembly, "ej_paper_assembly"),
    ):
        assert hasattr(mod, "run"), f"{name}.run missing"
        assert callable(mod.run), f"{name}.run is not callable"


# --------------------------------------------------------------------------
# Synthetic-panel sanity checks (Phase B)
# --------------------------------------------------------------------------

def _synthetic_panel(n_districts: int = 60, years: tuple[int, ...] = (2018, 2019, 2020),
                     seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_county = 6
    cds_per_county = n_districts // n_county
    rows = []
    for c in range(n_county):
        county_code = f"06{c+1:03d}"
        county_mean_ces = rng.uniform(15, 40)
        pm_iv_county = rng.uniform(7, 14)
        for d in range(cds_per_county):
            cds = f"{c+1:02d}-{d:05d}-0000000"
            pov = rng.uniform(0.05, 0.35)
            ell = rng.uniform(0.05, 0.4)
            tri_w = rng.uniform(0.0, 8.0)
            injury = 50 + tri_w * 5 + rng.normal(0, 15)
            # Within-county CES variation so county FE doesn't absorb the signal
            ces_score = county_mean_ces + rng.normal(0, 4.0)
            ces_diesel = rng.uniform(0.05, 0.4)
            pm_iv = pm_iv_county + rng.normal(0, 1.5)  # within-county IV variation
            for y in years:
                pm25 = pm_iv + rng.normal(0, 1.0)
                # outcomes driven partly by CES + PM + TRI
                math = 70 - 0.4 * ces_score - 0.6 * pm25 - 0.2 * tri_w + rng.normal(0, 3)
                ela = 65 - 0.3 * ces_score - 0.4 * pm25 + rng.normal(0, 3)
                absent = 12 + 0.15 * ces_score + 0.5 * tri_w + 0.01 * injury + rng.normal(0, 2)
                susp = 3 + 0.04 * ces_score + 0.05 * tri_w + rng.normal(0, 1)
                # synthetic per-pupil funding NEGATIVELY related to CES
                # (the EJ hypothesis): cleaner districts get more funding
                funding = 12000 - 50 * ces_score + rng.normal(0, 300)
                rows.append({
                    "cds": cds, "county_code": county_code, "year_num": y,
                    "enrollment": rng.integers(200, 5000),
                    "pct_frl": rng.uniform(0.1, 0.9),
                    "pct_ell": ell,
                    "census_saipe_child_poverty_rate": pov,
                    "calenviroscreen_score": ces_score,
                    "calenviroscreen_pm25": pm_iv,
                    "calenviroscreen_diesel_pm": ces_diesel,
                    "epa_pm25_annual_mean": pm25,
                    "epa_ozone_annual_mean": rng.uniform(0.028, 0.05),
                    "caaspp_math_met_pct": math,
                    "caaspp_ela_met_pct": ela,
                    "chronic_absenteeism_rate": absent,
                    "suspension_rate_pct": susp,
                    "epa_tri_district_weighted_mean": tri_w,
                    "epa_tri_district_count_2km_mean": tri_w * 0.7,
                    "epa_tri_district_nearest_km": 5.0 / (tri_w + 0.5),
                    "kd_health_injury_injury_hospitalization_rate_mean": injury,
                    "per_pupil_funding": funding,
                })
    return pd.DataFrame(rows)


def test_funding_gap_synthetic_signed_nonzero():
    from src.models import ej_funding_gap
    panel = _synthetic_panel(n_districts=60)
    out = ej_funding_gap.run(panel=panel)
    assert isinstance(out, dict)
    assert out.get("blocker") is None, f"unexpected blocker: {out.get('blocker')}"
    coef = out.get("ces_coef")
    assert coef is not None and np.isfinite(coef), "CES coef should be finite"
    # We baked a negative CES->funding relationship into the synthetic DGP
    assert coef < 0, f"expected negative CES coef, got {coef}"
    assert abs(coef) > 1.0, f"coef magnitude implausibly small: {coef}"
    assert out["n_districts"] >= 50
    assert out["model_type"] in {"spatial_error", "ols_cluster_fallback"}


def test_pollution_iv_synthetic_returns_dict():
    from src.models import pollution_iv
    panel = _synthetic_panel(n_districts=60)
    out = pollution_iv.run(panel=panel)
    assert isinstance(out, dict)
    assert "caaspp_math_met_pct" in out
    math = out["caaspp_math_met_pct"]
    assert math.get("instrument_used") == "distance_weighted_fallback"
    assert math.get("n", 0) > 50
    iv_coef = math.get("iv_coef_pp_per_ugm3")
    assert iv_coef is None or np.isfinite(iv_coef)


def test_tri_proximity_synthetic_returns_dict():
    from src.models import tri_proximity
    panel = _synthetic_panel(n_districts=60)
    out = tri_proximity.run(panel=panel)
    assert isinstance(out, dict)
    assert "absent_coef" in out
    assert "suspend_coef" in out
    # synthetic positive TRI->absent relationship
    if out["absent_coef"] is not None:
        assert np.isfinite(out["absent_coef"])


def test_assembly_synthetic_returns_dict():
    from src.models import ej_paper_assembly
    panel = _synthetic_panel(n_districts=60)
    out = ej_paper_assembly.run(panel=panel)
    assert isinstance(out, dict)
    for k in ("funding_gap", "pollution_iv", "tri_proximity", "generated_at"):
        assert k in out, f"summary missing {k}"


# --------------------------------------------------------------------------
# Funding-gap blocker path (no DV column)
# --------------------------------------------------------------------------

def test_funding_gap_blocker_when_no_dv():
    from src.models import ej_funding_gap
    panel = _synthetic_panel(n_districts=30).drop(columns=["per_pupil_funding"])
    out = ej_funding_gap.run(panel=panel)
    assert out.get("blocker") is not None
    assert out["model_type"] == "blocked"
    assert out["ces_coef"] is None
