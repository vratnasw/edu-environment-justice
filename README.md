# edu-environment-justice

**Status: Phase A scaffold.** Module stubs in place; implementation (Phase B) pending.

Environmental justice models — funding gap, pollution IV, TRI proximity. Models 18/24.

## Research questions

- Does pollution burden correlate with reduced per-pupil funding after spatial autocorrelation is accounted for?
- What is the causal effect of PM2.5 on test scores using wind direction as an instrument?
- How does proximity to TRI sites affect absenteeism, suspensions, and injury hospitalizations?

## Models / modules

- `src/models/ej_funding_gap.py` -- PySAL spreg spatial-error regression of per-pupil funding on CalEnviroScreen pollution burden. (-> `results/ej_funding_gap/spatial_error.json`)
- `src/models/pollution_iv.py` -- Wind-direction IV for PM2.5 -> test scores, with distance-weighted instrument fallback when wind data is sparse. (-> `results/pollution_iv/iv_estimates.json`)
- `src/models/tri_proximity.py` -- EPA TRI proximity on absenteeism + suspension, mediation through injury hospitalization. (-> `results/tri_proximity/proximity_effects.json`)
- `src/models/ej_paper_assembly.py` -- Headline assembler combining the three modules above. (-> `results/ej_paper_summary.json`)

## Data sources

Master panel (canonical join, 10,100 rows x 537 cols) at R2 key `processed/joined/master_panel.parquet`. Repo-specific parquets are listed in `config/config.yaml` under `data:`.

## Target journal

Environmental Health Perspectives or AERA Open

## Quick start

```
pip install -r requirements.txt
cp .env.example .env   # fill in 5 R2 vars
python scripts/preflight.py        # GO/NO-GO on R2 + parquets
python scripts/run_ej.py --fast  # scaffold stub
```

## Layout

```
config/        # config.yaml + r2_client.py (canonical copy)
src/models/    # one stub .py per model in this paper
src/causal/    # (Phase B)
src/utils/     # config_loader.py
scripts/       # preflight.py + run_ej.py orchestrator
tests/         # smoke tests for stub parsing
notebooks/     # exploration.ipynb (empty)
figures/       # paper figures (Phase B)
results/       # JSON outputs (Phase B)
```

