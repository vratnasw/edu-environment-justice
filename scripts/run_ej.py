"""Orchestrator for the Environmental Justice paper modules.

Runs ej_funding_gap, pollution_iv, and tri_proximity in sequence, writes
results/ej_paper_summary.json, and prints headline numbers.

Usage:
    python scripts/run_ej.py            # full sample
    python scripts/run_ej.py --fast     # 30% sample for quick iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the EJ paper pipeline.")
    ap.add_argument("--fast", action="store_true",
                    help="Subset to 30%% of district-years for quick iteration.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("run_ej")

    from src.models.ej_paper_assembly import run as run_summary
    summary = run_summary(fast=args.fast)

    def _fmt(v, prec: int = 4) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):.{prec}f}"
        except (TypeError, ValueError):
            return str(v)

    fg = summary["funding_gap"]
    pi = summary["pollution_iv"]
    tp = summary["tri_proximity"]

    log.info("=" * 72)
    log.info("EJ PAPER HEADLINES (fast=%s)", args.fast)
    log.info("=" * 72)
    if fg.get("blocker"):
        log.info("  FUNDING GAP : BLOCKED -- %s", fg["blocker"])
    else:
        log.info("  FUNDING GAP : CES coef = %s, gap $/SD = %s, model = %s, n = %s",
                 _fmt(fg["ces_coef"]),
                 _fmt(fg["dollars_per_sd"], prec=2),
                 fg.get("model"), fg.get("n_districts"))
    log.info("  POLLUTION IV:")
    log.info("    math : IV=%s/ug-m3  bias=%s%%  F1=%s  n=%s",
             _fmt(pi["math"]["iv_coef_pp_per_ugm3"]),
             _fmt(pi["math"]["ols_bias_pct"], prec=1),
             _fmt(pi["math"]["first_stage_f"], prec=2),
             pi["math"]["n"])
    log.info("    ela  : IV=%s/ug-m3  bias=%s%%  F1=%s  n=%s",
             _fmt(pi["ela"]["iv_coef_pp_per_ugm3"]),
             _fmt(pi["ela"]["ols_bias_pct"], prec=1),
             _fmt(pi["ela"]["first_stage_f"], prec=2),
             pi["ela"]["n"])
    log.info("    instrument = %s", pi["instrument_used"])
    log.info("  TRI PROXIMITY:")
    log.info("    chronic absent days/yr per SD = %s  (n=%s)",
             _fmt(tp["absent_days_per_year_per_sd"], prec=3),
             tp.get("absent_n"))
    log.info("    suspension pp per SD          = %s  (n=%s)",
             _fmt(tp["suspension_pp_per_sd"], prec=4),
             tp.get("suspend_n"))
    log.info("    prop mediated by injury       = %s  (mediator=%s)",
             _fmt(tp["prop_mediated_injury"], prec=3),
             tp.get("mediator_col"))
    log.info("=" * 72)
    log.info("summary written to results/ej_paper_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
