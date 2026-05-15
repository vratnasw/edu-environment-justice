"""Shared IO helpers: .env loader, R2 master-panel loader, JSON writer."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv() -> None:
    """Look for .env in repo, parent, then sibling edu-data-pipeline/."""
    candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / ".env",
        REPO_ROOT.parent / "edu-data-pipeline" / ".env",
    ]
    for c in candidates:
        if c.exists():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            log.info("loaded .env from %s", c)
            return
    log.info("no .env found; relying on already-exported env vars")


def _r2_client():
    """Lazy import of the r2_client module so callers don't need to manage sys.path."""
    sys.path.insert(0, str(REPO_ROOT / "config"))
    import r2_client  # noqa: WPS433
    return r2_client


def load_master_panel():
    """Download the master panel parquet from R2 and return as DataFrame."""
    load_dotenv()
    return _r2_client().download("processed/joined/master_panel.parquet")


def load_canonical(key_suffix: str):
    """Load any processed/canonical/<key>.parquet from R2."""
    load_dotenv()
    return _r2_client().download(f"processed/canonical/{key_suffix}")


def save_json(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write `payload` to `path` (creating parents) using a tolerant encoder."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        try:
            import numpy as np
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:  # noqa: BLE001
            pass
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    p.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return p


def results_dir(subdir: str = "") -> Path:
    p = REPO_ROOT / "results" / subdir if subdir else REPO_ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(subdir: str = "") -> Path:
    p = REPO_ROOT / "figures" / subdir if subdir else REPO_ROOT / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
