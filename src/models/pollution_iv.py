"""Wind-direction IV for PM2.5 -> CAASPP outcomes (with distance-weighted
exposure fallback).

ORIGINAL SPEC: instrument PM2.5 with wind-direction-mediated exposure from
upwind monitors. NOAA hourly-wind data is not on R2, so we use the documented
fallback: a distance-weighted exposure instrument.

FALLBACK CONSTRUCTION:
The canonical EPA AQS parquet on R2 is already aggregated to (county, year)
mean PM2.5/ozone — individual monitor coordinates are not present. The
closest analog to the spec's "monitor-distance-weighted exposure" is the
**CalEnviroScreen tract-area-weighted PM2.5** indicator (``calenviroscreen_pm25``):
it pools PM2.5 readings from CARB's monitor network using tract-level
inverse-distance weights and aggregates them to a per-county exposure score.

We use ``calenviroscreen_pm25`` as the IV for the AQS county-year
``epa_pm25_annual_mean``. Identification is cross-sectional (CES is fixed
vintage 2021). The exclusion restriction we lean on is the same as in the
original spec: tract-area-weighted exposure intensity affects test scores
only through actual ambient PM2.5 levels, not through any other channel.

Outcomes: ``caaspp_math_met_pct``, ``caaspp_ela_met_pct`` (one regression each).

Outputs: ``results/pollution_iv/<outcome>.json``
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CONTROLS: tuple[str, ...] = (
    "census_saipe_child_poverty_rate",
    "pct_ell",
    "pct_frl",
)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def _ols_fit(y: np.ndarray, X: np.ndarray, clusters: np.ndarray,
             names: list[str]) -> dict:
    """OLS with cluster-robust SEs."""
    import statsmodels.api as sm
    if not np.allclose(X[:, 0], 1.0):
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + names
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    return {
        "betas": {names[i]: float(res.params[i]) for i in range(len(names))},
        "se": {names[i]: float(res.bse[i]) for i in range(len(names))},
        "n": int(res.nobs),
    }


def _iv_fit(y: np.ndarray, endog: np.ndarray, exog_extra: np.ndarray,
            instrument: np.ndarray, names_exog: list[str],
            clusters: np.ndarray) -> dict:
    """linearmodels IV2SLS with cluster-robust SEs and weak-instrument tests."""
    from linearmodels.iv import IV2SLS
    n = len(y)
    # exog (controls) include a constant; endog has its own name
    exog = np.column_stack([np.ones(n), exog_extra])
    exog_names = ["const"] + names_exog
    exog_df = pd.DataFrame(exog, columns=exog_names)
    endog_df = pd.DataFrame(endog.reshape(-1, 1), columns=["pm25"])
    instr_df = pd.DataFrame(instrument.reshape(-1, 1), columns=["pm25_iv"])
    y_s = pd.Series(y, name="outcome")
    model = IV2SLS(dependent=y_s, exog=exog_df,
                   endog=endog_df, instruments=instr_df)
    res = model.fit(cov_type="clustered",
                    clusters=pd.Series(clusters, name="cluster"))

    iv_coef = float(res.params["pm25"])
    iv_se = float(res.std_errors["pm25"])

    # First-stage F (manual: regress pm25 on instrument+exog, then F-test
    # on the instrument).
    import statsmodels.api as sm
    Xfs = np.column_stack([np.ones(n), instrument, exog_extra])
    fs_res = sm.OLS(endog, Xfs).fit(cov_type="cluster",
                                     cov_kwds={"groups": clusters})
    # F-stat = (t-stat on instrument) ** 2 since there's one instrument
    fs_t = float(fs_res.tvalues[1])
    first_stage_f = fs_t ** 2

    # Cragg-Donald Wald F via linearmodels diagnostic if available
    cragg_donald = None
    try:
        diag = res.first_stage
        # diagnostic gives an F per endogenous; use the one for pm25
        diag_df = diag.diagnostics if hasattr(diag, "diagnostics") else None
        if diag_df is not None and "f.stat" in diag_df.columns:
            cragg_donald = float(diag_df.loc["pm25", "f.stat"])
        elif diag_df is not None and "f.pval" in diag_df.columns:
            # fall back to any F-stat column we can find
            cragg_donald = float(diag_df.iloc[0, 0])
    except Exception as e:  # noqa: BLE001
        log.warning("Cragg-Donald extraction failed: %s", e)

    return {
        "iv_coef": iv_coef,
        "iv_se": iv_se,
        "iv_p": float(res.pvalues["pm25"]),
        "first_stage_f": first_stage_f,
        "first_stage_coef_iv_on_pm25": float(fs_res.params[1]),
        "first_stage_se_iv_on_pm25": float(fs_res.bse[1]),
        "cragg_donald_wald_f": cragg_donald,
        "n": int(n),
    }


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------

def _build_design(panel: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, list[str]]:
    keep = ["cds", "county_code", "year_num", outcome,
            "epa_pm25_annual_mean", "calenviroscreen_pm25", "enrollment"] + \
           list(CONTROLS)
    keep = [c for c in keep if c in panel.columns]
    df = panel[keep].dropna().copy()
    df["log_enrollment"] = np.log(df["enrollment"].clip(lower=1.0))
    # Year FE (drop first as reference)
    year_dums = pd.get_dummies(df["year_num"], prefix="yr", drop_first=True)
    df = pd.concat([df.reset_index(drop=True),
                    year_dums.reset_index(drop=True)], axis=1).astype(
        {c: float for c in year_dums.columns})
    exog_names = ["log_enrollment"] + \
                 [c for c in CONTROLS if c in df.columns] + \
                 list(year_dums.columns)
    return df, exog_names


# --------------------------------------------------------------------------
# Public entry
# --------------------------------------------------------------------------

def run(fast: bool = False, panel: pd.DataFrame | None = None,
        outcomes: tuple[str, ...] = ("caaspp_math_met_pct", "caaspp_ela_met_pct")) -> dict:
    """Run IV regression for each outcome, write per-outcome JSONs."""
    from src.utils.io import load_master_panel, results_dir, save_json

    if panel is None:
        log.info("pollution_iv: downloading master panel from R2 ...")
        panel = load_master_panel()
    log.info("pollution_iv: panel %s", panel.shape)
    if fast:
        panel = panel.sample(frac=0.3, random_state=42)

    out: dict[str, dict] = {}
    for outcome in outcomes:
        if outcome not in panel.columns:
            log.warning("pollution_iv: outcome %s missing — skipping", outcome)
            continue
        df, exog_names = _build_design(panel, outcome)
        if len(df) < 50:
            payload = {
                "outcome": outcome,
                "blocker": f"only {len(df)} usable rows",
                "instrument_used": "distance_weighted_fallback",
                "original_spec_blocker": "NOAA wind direction not on R2",
            }
            save_json(results_dir("pollution_iv") / f"{outcome}.json", payload)
            out[outcome] = payload
            continue

        y = df[outcome].to_numpy(dtype=float)
        pm25 = df["epa_pm25_annual_mean"].to_numpy(dtype=float)
        iv_var = df["calenviroscreen_pm25"].to_numpy(dtype=float)
        exog_extra = df[exog_names].to_numpy(dtype=float)
        clusters = df["county_code"].astype("category").cat.codes.to_numpy()

        # OLS spec: regress outcome on PM2.5 + controls
        X_ols = np.column_stack([pm25, exog_extra])
        names_ols = ["pm25"] + exog_names
        ols = _ols_fit(y, X_ols, clusters, names_ols)
        ols_coef = ols["betas"]["pm25"]
        ols_se = ols["se"]["pm25"]

        # IV spec
        iv = _iv_fit(y, pm25, exog_extra, iv_var, exog_names, clusters)

        denom = ols_coef if abs(ols_coef) > 1e-12 else float("nan")
        ols_bias_pct = ((ols_coef - iv["iv_coef"]) / denom) * 100 if denom == denom else float("nan")

        payload = {
            "outcome": outcome,
            "instrument_used": "distance_weighted_fallback",
            "original_spec_blocker": "NOAA wind direction not on R2",
            "instrument_construction": (
                "calenviroscreen_pm25 (CARB tract-area-weighted PM2.5, vintage 2021) "
                "used as instrument for AQS county-year epa_pm25_annual_mean. "
                "Cross-sectional identification; year FE absorb panel-mean shifts."
            ),
            "n": int(len(df)),
            "ols_coef_pp_per_ugm3": float(ols_coef),
            "ols_se": float(ols_se),
            "iv_coef_pp_per_ugm3": float(iv["iv_coef"]),
            "iv_se": float(iv["iv_se"]),
            "iv_p_value": float(iv["iv_p"]),
            "first_stage_f": float(iv["first_stage_f"]),
            "first_stage_coef_iv_on_pm25": float(iv["first_stage_coef_iv_on_pm25"]),
            "first_stage_se_iv_on_pm25": float(iv["first_stage_se_iv_on_pm25"]),
            "cragg_donald_wald_f": iv.get("cragg_donald_wald_f"),
            "ols_iv_bias_pct": float(ols_bias_pct) if ols_bias_pct == ols_bias_pct else None,
            "controls": exog_names,
            "fast": fast,
        }
        save_json(results_dir("pollution_iv") / f"{outcome}.json", payload)
        out[outcome] = payload
        log.info(
            "pollution_iv[%s]: OLS=%.4f IV=%.4f bias=%.1f%% F1=%.2f n=%d",
            outcome, ols_coef, iv["iv_coef"],
            payload["ols_iv_bias_pct"] if payload["ols_iv_bias_pct"] is not None else float("nan"),
            iv["first_stage_f"], len(df),
        )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    print(run())
