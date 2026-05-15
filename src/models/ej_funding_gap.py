"""PySAL spreg spatial-error regression of per-pupil funding on CalEnviroScreen
pollution burden.

Spec (Phase B):
- DV: per-pupil funding (cross-section, latest year per district).
- IV: ``calenviroscreen_score``.
- Controls: ``census_saipe_child_poverty_rate``, ``pct_ell``, ``log(enrollment)``,
  ``county_code`` dummies.
- Spatial structure: queen contiguity from district polygons. If polygons are
  unavailable, k=5 nearest neighbors on district lat/lon from a `districts_meta`
  source. If no spatial structure is available, fall back to OLS with
  county-clustered SEs and flag in output.

DATA BLOCKER:
The master panel currently has no per-pupil-funding column (no `*_per_pupil*`,
`*_revenue*`, `*_expenditure*`, `*_funding*`). When `run()` is called without
data, it writes ``results/ej_funding_gap_blocker.json`` and returns a stub
with ``blocker`` set. All estimator code below is implemented and will work
as soon as a DV column is added to the master panel and named in the
``DV_CANDIDATES`` tuple (or passed via ``dv_col``).
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Candidate column names; the first one that's present + non-empty wins.
DV_CANDIDATES: tuple[str, ...] = (
    "per_pupil_funding",
    "per_pupil_expenditure",
    "per_pupil_revenue",
    "ppu_total",
    "adjusted_per_pupil_revenue",
    "total_expenditure_per_pupil",
)

CONTROL_COLS: tuple[str, ...] = (
    "census_saipe_child_poverty_rate",
    "pct_ell",
)


# --------------------------------------------------------------------------
# Spatial weights
# --------------------------------------------------------------------------

def _build_spatial_weights(df: pd.DataFrame,
                           lat_col: Optional[str] = None,
                           lon_col: Optional[str] = None,
                           k: int = 5):
    """Build a libpysal weights matrix.

    Prefer queen contiguity if `polygon` is in `df`; else k-NN on (lat, lon).
    Returns (W, model_type_str). On failure returns (None, "ols_cluster_fallback").
    """
    try:
        import libpysal
    except Exception as e:  # noqa: BLE001
        log.warning("libpysal unavailable: %s — falling back to OLS", e)
        return None, "ols_cluster_fallback"

    # Queen contiguity from polygons
    if "geometry" in df.columns:
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(df.copy(), geometry="geometry")
            w = libpysal.weights.Queen.from_dataframe(gdf)
            w.transform = "r"
            return w, "spatial_error"
        except Exception as e:  # noqa: BLE001
            log.warning("Queen contiguity failed: %s — trying k-NN", e)

    # k-NN on lat/lon
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        coords = df[[lon_col, lat_col]].to_numpy()
        try:
            w = libpysal.weights.KNN.from_array(coords, k=k)
            w.transform = "r"
            return w, "spatial_error"
        except Exception as e:  # noqa: BLE001
            log.warning("k-NN weights failed: %s", e)
    return None, "ols_cluster_fallback"


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def _fit_spatial_error(y: np.ndarray, X: np.ndarray, W,
                       x_names: list[str]) -> dict:
    """Maximum-likelihood spatial-error model via spreg.ML_Error."""
    from spreg import ML_Error
    res = ML_Error(y.reshape(-1, 1), X, W, name_y="dv", name_x=x_names)
    # res.betas shape (k+1, 1) with last row = lambda
    betas = np.asarray(res.betas).flatten()
    std_err = np.asarray(res.std_err).flatten()
    # First row of betas is the intercept; covariates start at index 1
    coef_map = {n: float(betas[i + 1]) for i, n in enumerate(x_names)}
    se_map = {n: float(std_err[i + 1]) for i, n in enumerate(x_names)}
    return {
        "betas": coef_map,
        "se": se_map,
        "lambda": float(betas[-1]),
        "n": int(res.n),
        "k": int(res.k),
        "log_likelihood": float(res.logll),
    }


def _fit_ols_cluster(y: np.ndarray, X: np.ndarray, x_names: list[str],
                     clusters: np.ndarray) -> dict:
    """OLS with cluster-robust standard errors (one-way clustering)."""
    import statsmodels.api as sm
    # X already has a leading constant column? Add one if not.
    if not np.allclose(X[:, 0], 1.0):
        X_full = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(x_names)
    else:
        X_full = X
        names = ["const"] + list(x_names) if X.shape[1] == len(x_names) else list(x_names)
    model = sm.OLS(y, X_full)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    coef_map = {names[i]: float(res.params[i]) for i in range(len(names))}
    se_map = {names[i]: float(res.bse[i]) for i in range(len(names))}
    return {
        "betas": coef_map,
        "se": se_map,
        "lambda": None,
        "n": int(res.nobs),
        "k": int(X_full.shape[1]),
        "log_likelihood": float(res.llf),
    }


def _morans_i(resid: np.ndarray, W) -> Optional[float]:
    """Moran's I on regression residuals."""
    try:
        from esda.moran import Moran
        m = Moran(resid, W)
        return float(m.I)
    except Exception as e:  # noqa: BLE001
        log.warning("Moran's I failed: %s", e)
        return None


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------

def _pick_dv(panel: pd.DataFrame,
             dv_col: Optional[str],
             candidates: Iterable[str] = DV_CANDIDATES) -> Optional[str]:
    if dv_col and dv_col in panel.columns and panel[dv_col].notna().sum() > 30:
        return dv_col
    for c in candidates:
        if c in panel.columns and panel[c].notna().sum() > 30:
            return c
    return None


def _build_design(panel: pd.DataFrame, dv_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Cross-section: take latest year per district, drop NaN on DV/controls."""
    df = panel.sort_values("year_num").groupby("cds", as_index=False).tail(1).copy()
    df["log_enrollment"] = np.log(df["enrollment"].clip(lower=1.0))
    keep = ["cds", "county_code", "year_num", dv_col,
            "calenviroscreen_score", "log_enrollment"] + list(CONTROL_COLS)
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna()
    # County FE via dummies (drop one as reference)
    if "county_code" in df.columns:
        dums = pd.get_dummies(df["county_code"], prefix="county", drop_first=True)
        df = pd.concat([df.reset_index(drop=True), dums.reset_index(drop=True)], axis=1)
        county_cols = list(dums.columns)
    else:
        county_cols = []
    rhs = ["calenviroscreen_score", "log_enrollment"] + \
          [c for c in CONTROL_COLS if c in df.columns] + county_cols
    return df, rhs


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def _funding_scatter(df: pd.DataFrame, resid: np.ndarray, out_path) -> None:
    """CES score (x) vs residual (y), colored by county."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    counties = df["county_code"].astype(str).to_numpy() if "county_code" in df.columns else None
    if counties is not None:
        uniq = sorted(set(counties))
        cmap = plt.get_cmap("tab20")
        for i, cc in enumerate(uniq):
            m = counties == cc
            ax.scatter(df["calenviroscreen_score"].to_numpy()[m], resid[m],
                       s=22, alpha=0.65, color=cmap(i % 20),
                       label=cc if i < 20 else None,
                       edgecolor="#1e293b", linewidth=0.3)
    else:
        ax.scatter(df["calenviroscreen_score"].to_numpy(), resid,
                   s=22, alpha=0.65, color="#475569", edgecolor="#1e293b",
                   linewidth=0.3)
    ax.axhline(0.0, color="#0f172a", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("CalEnviroScreen score (pollution burden)")
    ax.set_ylabel("Per-pupil funding residual (predicted − actual)")
    ax.set_title("Funding-burden residual scatter, by county", fontsize=12)
    ax.grid(alpha=0.25, linestyle=":")
    if counties is not None and len(set(counties)) <= 20:
        ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def run(fast: bool = False, panel: pd.DataFrame | None = None,
        dv_col: Optional[str] = None) -> dict:
    """Run spatial-error regression of per-pupil funding on CES burden.

    Returns a dict; writes ``results/ej_funding_gap/results.json``. If no
    DV column is found, writes ``results/ej_funding_gap_blocker.json`` and
    returns a stub dict with ``blocker`` set.
    """
    from src.utils.io import load_master_panel, results_dir, figures_dir, save_json

    if panel is None:
        log.info("ej_funding_gap: downloading master panel from R2 ...")
        panel = load_master_panel()
    log.info("ej_funding_gap: panel %s", panel.shape)

    if fast:
        # Sample 30% of district-years deterministically
        panel = panel.sample(frac=0.3, random_state=42)
        log.info("ej_funding_gap: fast-mode sample %s", panel.shape)

    picked = _pick_dv(panel, dv_col)
    if picked is None:
        blocker_msg = (
            "no per-pupil-funding column found in master_panel.parquet. "
            "Searched: " + ", ".join(DV_CANDIDATES) + ". "
            "Add a per-pupil-funding column (e.g. CDE J-200 expenditure data) "
            "to the canonical layer and re-run."
        )
        log.warning("ej_funding_gap: BLOCKER — %s", blocker_msg)
        stub = {
            "blocker": blocker_msg,
            "ces_coef": None,
            "ces_se": None,
            "funding_gap_dollars_per_sd": None,
            "n_districts": 0,
            "model_type": "blocked",
            "moran_i_residuals": None,
        }
        save_json(results_dir() / "ej_funding_gap_blocker.json", stub)
        save_json(results_dir("ej_funding_gap") / "results.json", stub)
        return stub

    log.info("ej_funding_gap: DV = %s", picked)
    df, rhs = _build_design(panel, picked)
    if len(df) < 30:
        stub = {
            "blocker": f"only {len(df)} usable rows after dropna on DV+controls",
            "ces_coef": None, "ces_se": None,
            "funding_gap_dollars_per_sd": None,
            "n_districts": len(df), "model_type": "blocked",
            "moran_i_residuals": None,
        }
        save_json(results_dir("ej_funding_gap") / "results.json", stub)
        return stub

    y = df[picked].to_numpy(dtype=float)
    X = df[rhs].to_numpy(dtype=float)

    # Spatial weights — no geometry/lat-lon in panel, will fall back to OLS
    W, model_type = _build_spatial_weights(df)

    if model_type == "spatial_error":
        try:
            fit = _fit_spatial_error(y, X, W, rhs)
        except Exception as e:  # noqa: BLE001
            log.warning("spatial-error fit failed (%s) — falling back to OLS", e)
            model_type = "ols_cluster_fallback"
            clusters = df["county_code"].astype("category").cat.codes.to_numpy()
            fit = _fit_ols_cluster(y, X, rhs, clusters)
    else:
        clusters = df["county_code"].astype("category").cat.codes.to_numpy()
        fit = _fit_ols_cluster(y, X, rhs, clusters)

    ces_coef = fit["betas"].get("calenviroscreen_score", float("nan"))
    ces_se = fit["se"].get("calenviroscreen_score", float("nan"))
    ces_sd = float(df["calenviroscreen_score"].std())
    funding_gap = ces_coef * ces_sd

    # Residuals + Moran's I
    if "const" in fit["betas"]:
        # OLS fallback already added a const column; recompute residuals
        X_full = np.column_stack([np.ones(len(y)), X])
        beta_vec = np.array([fit["betas"]["const"]] +
                            [fit["betas"][n] for n in rhs])
        pred = X_full @ beta_vec
    else:
        beta_vec = np.array([fit["betas"][n] for n in rhs])
        pred = X @ beta_vec + (fit.get("intercept", 0.0) or 0.0)
    resid = y - pred
    moran = _morans_i(resid, W) if W is not None else None

    # Figure
    try:
        _funding_scatter(df, resid, figures_dir() / "ej_funding_scatter.pdf")
    except Exception as e:  # noqa: BLE001
        log.warning("scatter plot failed: %s", e)

    result = {
        "dv_col": picked,
        "ces_coef": float(ces_coef),
        "ces_se": float(ces_se),
        "ces_sd": ces_sd,
        "funding_gap_dollars_per_sd": float(funding_gap),
        "n_districts": int(len(df)),
        "model_type": model_type,
        "moran_i_residuals": moran,
        "controls": [c for c in CONTROL_COLS if c in df.columns] + ["log_enrollment"],
        "fast": fast,
        "blocker": None,
    }
    save_json(results_dir("ej_funding_gap") / "results.json", result)
    log.info("ej_funding_gap: ces_coef=%.4f se=%.4f gap=%.2f n=%d model=%s",
             ces_coef, ces_se, funding_gap, len(df), model_type)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    print(run())
