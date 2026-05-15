"""Wind-direction IV for PM2.5 -> test scores, with distance-weighted instrument fallback when wind data is sparse.

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - processed/joined/master_panel.parquet
  - processed/canonical/epa_aqs.parquet
Outputs: results/pollution_iv/iv_estimates.json
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
