"""EPA TRI proximity on absenteeism + suspension, mediation through injury hospitalization.

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - processed/joined/master_panel.parquet
  - processed/canonical/epa_tri.parquet
Outputs: results/tri_proximity/proximity_effects.json
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
