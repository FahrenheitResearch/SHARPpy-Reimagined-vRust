"""Safe subprocess bridge to the bundled standalone ``ecape-rs`` solver."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
import os
import subprocess

import numpy as np


@dataclass(frozen=True)
class NativeEcapeResult:
    ecape: float
    ncape: float
    cape: float


def _binary_path() -> str:
    override = os.environ.get("SHARPMOD_ECAPE_BIN", "").strip()
    if override:
        return override
    name = "rw_ecape_analytic.exe" if os.name == "nt" else "rw_ecape_analytic"
    return str(files("sharpmod.resources").joinpath("bin", name))


def available() -> bool:
    return os.path.isfile(_binary_path())


def analytic_ecape(pres, hght, tmpc, dwpc, u_knots, v_knots,
                   *, timeout: float = 5.0) -> NativeEcapeResult | None:
    """Return verified analytic MU ECAPE, or ``None`` for safe fallback."""
    path = _binary_path()
    if not os.path.isfile(path):
        return None
    arrays = [np.asarray(value, dtype=float) for value in (
        pres, hght, tmpc, dwpc, u_knots, v_knots)]
    if not arrays or arrays[0].size < 3:
        return None
    if any(value.shape != arrays[0].shape for value in arrays):
        return None
    if not all(np.all(np.isfinite(value)) for value in arrays):
        return None
    payload = {
        "pressure_hpa": arrays[0].tolist(),
        "height_m_msl": arrays[1].tolist(),
        "temperature_c": arrays[2].tolist(),
        "dewpoint_c": arrays[3].tolist(),
        "u_knots": arrays[4].tolist(),
        "v_knots": arrays[5].tolist(),
    }
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            [path], input=json.dumps(payload, allow_nan=False),
            capture_output=True, text=True, timeout=timeout, check=True,
            **kwargs,
        )
        result = json.loads(completed.stdout)
        if result.get("schema") != "sharpmod.ecape.v1":
            return None
        ecape = float(result["ecape_jkg"])
        ncape = float(result["ncape_jkg"])
        cape = float(result["cape_jkg"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError,
            TypeError, json.JSONDecodeError):
        return None
    if not all(np.isfinite(value) for value in (ecape, ncape, cape)):
        return None
    if ecape < 0.0 or cape < 0.0 or ecape > cape + 1.0e-6:
        return None
    return NativeEcapeResult(ecape=ecape, ncape=ncape, cape=cape)
