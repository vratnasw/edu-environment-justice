"""Headline assembler. Calls run() on the three EJ modules, downloads the
master panel ONCE, and writes ``results/ej_paper_summary.json``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)


def run(fast: bool = False, panel: pd.DataFrame | None = None) -> dict:
    """Run all three EJ modules and write the unified summary JSON."""
    from src.utils.io import load_master_panel, results_dir, save_json
    from src.models import ej_funding_gap, pollution_iv, tri_proximity

    if panel is None:
        log.info("ej_paper_assembly: downloading master panel from R2 ...")
        panel = load_master_panel()
    log.info("ej_paper_assembly: panel %s (fast=%s)", panel.shape, fast)

    log.info("ej_paper_assembly: running funding_gap ...")
    funding = ej_funding_gap.run(fast=fast, panel=panel)

    log.info("ej_paper_assembly: running pollution_iv ...")
    pollution = pollution_iv.run(fast=fast, panel=panel)

    log.info("ej_paper_assembly: running tri_proximity ...")
    tri = tri_proximity.run(fast=fast, panel=panel)

    def _g(d: dict | None, key: str):
        return None if d is None else d.get(key)

    math_p = pollution.get("caaspp_math_met_pct", {})
    ela_p = pollution.get("caaspp_ela_met_pct", {})

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fast": fast,
        "funding_gap": {
            "dollars_per_sd": _g(funding, "funding_gap_dollars_per_sd"),
            "ces_coef": _g(funding, "ces_coef"),
            "ces_se": _g(funding, "ces_se"),
            "model": _g(funding, "model_type"),
            "n_districts": _g(funding, "n_districts"),
            "moran_i_residuals": _g(funding, "moran_i_residuals"),
            "blocker": _g(funding, "blocker"),
        },
        "pollution_iv": {
            "math": {
                "iv_coef_pp_per_ugm3": _g(math_p, "iv_coef_pp_per_ugm3"),
                "iv_se": _g(math_p, "iv_se"),
                "first_stage_f": _g(math_p, "first_stage_f"),
                "ols_bias_pct": _g(math_p, "ols_iv_bias_pct"),
                "n": _g(math_p, "n"),
            },
            "ela": {
                "iv_coef_pp_per_ugm3": _g(ela_p, "iv_coef_pp_per_ugm3"),
                "iv_se": _g(ela_p, "iv_se"),
                "first_stage_f": _g(ela_p, "first_stage_f"),
                "ols_bias_pct": _g(ela_p, "ols_iv_bias_pct"),
                "n": _g(ela_p, "n"),
            },
            "instrument_used": "distance_weighted_fallback",
            "original_spec_blocker": "NOAA wind direction not on R2",
        },
        "tri_proximity": {
            "absent_coef": _g(tri, "absent_coef"),
            "absent_se": _g(tri, "absent_se"),
            "absent_days_per_year_per_sd": _g(tri, "absent_days_per_year_per_sd"),
            "suspend_coef": _g(tri, "suspend_coef"),
            "suspend_se": _g(tri, "suspend_se"),
            "suspension_pp_per_sd": _g(tri, "suspend_pp_per_sd"),
            "prop_mediated_injury": _g(tri, "prop_mediated_injury"),
            "mediator_col": _g(tri, "mediator_col"),
            "absent_n": _g(tri, "absent_n"),
            "suspend_n": _g(tri, "suspend_n"),
        },
    }
    out_path = results_dir() / "ej_paper_summary.json"
    save_json(out_path, summary)
    log.info("ej_paper_assembly: wrote %s", out_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    print(run())
