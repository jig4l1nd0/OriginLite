"""Project save/load (.olite) format.

Uses a zip container (.olite) with manifest + data + configs.

Manifest structure (version 2):
{
    "version": 2,
    "meta": {"created_at": ISO8601, "app_version": str, "name": str},
    "chart_config": {...},
    "data_model": {...},
    "extra": {... optional ...}
}
Version 1 manifests lacked the meta block and used version=1.
"""
from __future__ import annotations
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from ..core.data_model import DataModel

APP_VERSION = "0.1.0"


def save_project(
    path: str | Path,
    dm: DataModel,
    chart_config,
    extra: Dict[str, Any] | None = None,
    name: str | None = None,
):
    path = Path(path)
    if path.suffix != ".olite":
        path = path.with_suffix(".olite")
    # Embed operations redundantly into chart_config for quick access (compat)
    if "data_model_operations" not in chart_config:
        chart_config["data_model_operations"] = dm.operations
    manifest = {
        "version": 2,
        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "app_version": APP_VERSION,
            "name": name or path.stem,
        },
        "chart_config": chart_config,
        "data_model": dm.to_dict(),
    }
    if extra:
        manifest["extra"] = extra
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        csv_bytes = dm.df.to_csv(index=False).encode()
        z.writestr("data.csv", csv_bytes)
    return path


def load_project(path: str | Path):
    """Load a project and return (DataModel, chart_config, extra).

    For version >=2 manifests, meta is injected into extra['meta'] for
    backward compatibility (tests expecting 3-tuple).
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as z:
        manifest = json.loads(z.read("manifest.json").decode())
        from ..core.data_model import DataModel as _DM

        dm = _DM.from_dict(manifest["data_model"])
        chart_config = manifest.get("chart_config", {})
        extra = manifest.get("extra", {})
        meta = manifest.get("meta")
        if meta is not None:
            extra = dict(extra)  # shallow copy
            extra.setdefault('meta', meta)
    # Mirror operations into chart_config if absent
    if "data_model_operations" not in chart_config and dm.operations:
        chart_config["data_model_operations"] = dm.operations
    return dm, chart_config, extra


def list_projects(directory: str | Path) -> List[Path]:
    """Return list of .olite project paths in directory (non-recursive)."""
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(p.glob("*.olite"))

 
__all__ = [
    "save_project",
    "load_project",
    "list_projects",
    "APP_VERSION",
]
