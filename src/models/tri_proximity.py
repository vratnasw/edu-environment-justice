"""EPA TRI proximity on chronic absenteeism + suspension, mediated through
injury hospitalization rates.

Spec (Phase B):
- Treatment: ``epa_tri_district_weighted_mean`` (district-level inverse-distance
  weighted TRI facility count from master panel).
- Outcomes: ``chronic_absenteeism_rate``, ``suspension_rate_pct`` (two
  separate OLS regressions, cluster-robust SEs at county level).
- Controls: ``census_saipe_child_poverty_rate``, ``pct_ell``, ``log(enrollment)``,
  county FE.
- Mediator: ``kd_health_injury_injury_hospitalization_rate_mean`` (5-17 yr
  injury-hospitalization rate per 100k from KidsData). Bootstrap mediation
  with 2000 reps via ``pingouin.mediation_analysis``.

Outputs: ``results/tri_proximity/results.json``
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CONTROLS: tuple[str, ...] = (
    "census_saipe_child_poverty_rate",
    "pct_ell",
)

# Injury mediator candidates (first available wins)
MEDIATOR_CANDIDATES: tuple[str, ...] = (
    "kd_health_injury_injury_hospitalization_rate_mean",
    "kd_health_injury_self_harm_hospitalizations_mean",
    "kd_health_healthcare_injury_hospitalization_rate_mean",
    "kd_health_hospital_injury_hospitalization_rate_mean",
)


def _build_design(panel: pd.DataFrame, outcome: str, mediator: str | None) -> pd.DataFrame:
    cols = ["cds", "county_code", "year_num", outcome,
            "epa_tri_district_weighted_mean", "enrollment"] + list(CONTROLS)
    if mediator and mediator in panel.columns:
        cols.append(mediator)
    cols = [c for c in cols if c in panel.columns]
    df = panel[cols].dropna().copy()
    df["log_enrollment"] = np.log(df["enrollment"].clip(lower=1.0))
    return df


def _ols_with_county_fe(df: pd.DataFrame, outcome: str) -> dict:
    """OLS with county FE + county-clustered SEs."""
    import statsmodels.formula.api as smf
    controls = [c for c in CONTROLS if c in df.columns]
    rhs = ["epa_tri_district_weighted_mean"] + controls + ["log_enrollment"]
    rhs_str = " + ".join(rhs) + " + C(county_code)"
    formula = f"{outcome} ~ {rhs_str}"
    res = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["county_code"]}
    )
    return {
        "coef": float(res.params["epa_tri_district_weighted_mean"]),
        "se": float(res.bse["epa_tri_district_weighted_mean"]),
        "p": float(res.pvalues["epa_tri_district_weighted_mean"]),
        "n": int(res.nobs),
        "r2": float(res.rsquared),
    }


def _pick_mediator(df: pd.DataFrame) -> str | None:
    for c in MEDIATOR_CANDIDATES:
        if c in df.columns and df[c].notna().sum() > 100 and df[c].std() > 0:
            return c
    return None


def _mediation(df: pd.DataFrame, outcome: str, mediator: str,
               n_boot: int = 2000) -> dict:
    """pingouin.mediation_analysis with bootstrap."""
    import pingouin as pg
    # pingouin doesn't natively cluster; we feed the cross-section + controls
    sub = df[[outcome, "epa_tri_district_weighted_mean", mediator,
              "log_enrollment"] + [c for c in CONTROLS if c in df.columns]].dropna()
    if len(sub) < 100:
        return {"prop_mediated": None,
                "indirect_effect": None,
                "direct_effect": None,
                "n_mediation": int(len(sub)),
                "mediator": mediator,
                "note": "insufficient observations for mediation"}
    covar = ["log_enrollment"] + [c for c in CONTROLS if c in sub.columns]
    try:
        med = pg.mediation_analysis(
            data=sub,
            x="epa_tri_district_weighted_mean",
            m=mediator,
            y=outcome,
            covar=covar,
            n_boot=n_boot,
            seed=42,
        )
        # Extract rows
        idx_indirect = med["path"].str.contains("Indirect")
        idx_direct = med["path"].str.contains("Direct")
        idx_total = med["path"].str.contains("Total")
        indirect = float(med.loc[idx_indirect, "coef"].iloc[0]) if idx_indirect.any() else float("nan")
        direct = float(med.loc[idx_direct, "coef"].iloc[0]) if idx_direct.any() else float("nan")
        total = float(med.loc[idx_total, "coef"].iloc[0]) if idx_total.any() else (indirect + direct)
        prop = float(indirect / total) if abs(total) > 1e-12 else float("nan")
        return {
            "prop_mediated": prop,
            "indirect_effect": indirect,
            "direct_effect": direct,
            "total_effect": total,
            "n_mediation": int(len(sub)),
            "mediator": mediator,
            "n_boot": n_boot,
        }
    except Exception as e:  # noqa: BLE001
        log.warning("pingouin mediation failed: %s", e)
        return {
            "prop_mediated": None, "indirect_effect": None, "direct_effect": None,
            "n_mediation": int(len(sub)), "mediator": mediator,
            "error": f"{type(e).__name__}: {e}",
        }


def run(fast: bool = False, panel: pd.DataFrame | None = None) -> dict:
    """Run TRI proximity regressions + mediation, write results JSON."""
    from src.utils.io import load_master_panel, results_dir, save_json

    if panel is None:
        log.info("tri_proximity: downloading master panel from R2 ...")
        panel = load_master_panel()
    log.info("tri_proximity: panel %s", panel.shape)
    if fast:
        panel = panel.sample(frac=0.3, random_state=42)

    mediator = _pick_mediator(panel)
    log.info("tri_proximity: mediator=%s", mediator)

    # Absenteeism
    df_abs = _build_design(panel, "chronic_absenteeism_rate", mediator)
    if len(df_abs) > 50:
        abs_fit = _ols_with_county_fe(df_abs, "chronic_absenteeism_rate")
        tri_sd_abs = float(df_abs["epa_tri_district_weighted_mean"].std())
        # `chronic_absenteeism_rate` is a percentage of days absent. We convert
        # to expected absent days/year using a 180-day school year and the
        # standard deviation in TRI weighted proximity.
        abs_days_per_year_per_sd = float(abs_fit["coef"]) * tri_sd_abs * (180.0 / 100.0)
    else:
        abs_fit = {"coef": None, "se": None, "p": None, "n": int(len(df_abs))}
        abs_days_per_year_per_sd = None

    # Suspensions
    df_susp = _build_design(panel, "suspension_rate_pct", mediator)
    if len(df_susp) > 50:
        susp_fit = _ols_with_county_fe(df_susp, "suspension_rate_pct")
        tri_sd_susp = float(df_susp["epa_tri_district_weighted_mean"].std())
        susp_pp_per_sd = float(susp_fit["coef"]) * tri_sd_susp
    else:
        susp_fit = {"coef": None, "se": None, "p": None, "n": int(len(df_susp))}
        susp_pp_per_sd = None

    # Mediation (run only on the absenteeism design; same TRI effect)
    if mediator and len(df_abs) > 100:
        n_boot = 500 if fast else 2000
        med = _mediation(df_abs, "chronic_absenteeism_rate", mediator, n_boot=n_boot)
    else:
        med = {"prop_mediated": None, "mediator": mediator,
               "note": "no suitable mediator or insufficient rows"}

    payload = {
        "absent_coef": abs_fit["coef"],
        "absent_se": abs_fit["se"],
        "absent_p": abs_fit.get("p"),
        "absent_n": abs_fit.get("n"),
        "absent_days_per_year_per_sd": abs_days_per_year_per_sd,
        "suspend_coef": susp_fit["coef"],
        "suspend_se": susp_fit["se"],
        "suspend_p": susp_fit.get("p"),
        "suspend_n": susp_fit.get("n"),
        "suspend_pp_per_sd": susp_pp_per_sd,
        "prop_mediated_injury": med.get("prop_mediated"),
        "mediation_indirect": med.get("indirect_effect"),
        "mediation_direct": med.get("direct_effect"),
        "mediation_total": med.get("total_effect"),
        "mediation_n": med.get("n_mediation"),
        "mediator_col": med.get("mediator"),
        "mediation_n_boot": med.get("n_boot"),
        "fast": fast,
        "tri_treatment_col": "epa_tri_district_weighted_mean",
    }
    from src.utils.io import save_json as _save  # noqa
    _save(results_dir("tri_proximity") / "results.json", payload)
    log.info(
        "tri_proximity: absent_coef=%s suspend_coef=%s prop_mediated=%s",
        abs_fit["coef"], susp_fit["coef"], med.get("prop_mediated"),
    )
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    print(run())
