"""Environment and dependency checks (run before any experiment)."""
from __future__ import annotations

import importlib
import os
import platform
import shutil

REQUIRED = {  # import name -> pip name
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
    "matplotlib": "matplotlib", "torch": "torch", "lightgbm": "lightgbm", "xgboost": "xgboost",
    "transformers": "transformers", "sentence_transformers": "sentence-transformers",
}


def _ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def check_environment(required=REQUIRED, strict: bool = True, path: str = ".") -> dict:
    """Return versions and hardware facts; raise with an install command if packages are missing."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(),
            "disk_free_gb": round(shutil.disk_usage(path).free / 1e9, 1), "packages": {}, "missing": []}
    for mod, pip_name in required.items():
        try:
            info["packages"][mod] = getattr(importlib.import_module(mod), "__version__", "unknown")
        except Exception:
            info["packages"][mod] = None
            info["missing"].append(pip_name)
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        info["gpu"] = None
    if strict and info["missing"]:
        raise RuntimeError("Missing packages: " + ", ".join(info["missing"]) +
                           "\nInstall with:  pip install " + " ".join(info["missing"]))
    return info
