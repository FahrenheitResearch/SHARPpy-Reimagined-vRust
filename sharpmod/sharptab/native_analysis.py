"""Stable, optional loader for the vRust bulk sounding-analysis extension."""

from __future__ import annotations

import logging
import os

import numpy as np
import numpy.ma as ma


_LOGGER = logging.getLogger(__name__)
_SCHEMA = "sharpmod.native-analysis.v1"


class NativeAnalysisUnavailable(RuntimeError):
    """Raised when the packaged native backend cannot be loaded or used."""


def _extension():
    if os.environ.get("SHARPMOD_DISABLE_NATIVE_ANALYSIS", "").strip() in {
            "1", "true", "yes", "on"}:
        raise NativeAnalysisUnavailable("native analysis disabled by environment")
    try:
        from sharpmod import sharpmod_native
    except (ImportError, OSError) as exc:
        raise NativeAnalysisUnavailable(str(exc)) from exc
    if not sharpmod_native.runtime_check():
        raise NativeAnalysisUnavailable("native extension runtime check failed")
    return sharpmod_native


def available() -> bool:
    try:
        _extension()
    except NativeAnalysisUnavailable:
        return False
    return True


def backend_info() -> dict:
    return dict(_extension().backend_info())


def best_guess_precip(phase, init_temp_c, init_level_agl_m,
                      positive_area, negative_area, surface_temp_c) -> str:
    return str(_extension().best_guess_precip(
        int(phase), float(init_temp_c), float(init_level_agl_m),
        float(positive_area), float(negative_area), float(surface_temp_c)))


def classify_watch(values: dict) -> str:
    return str(_extension().classify_watch(dict(values)))


def _values(value, *, optional=False):
    if value is None:
        return None if optional else []
    array = ma.asarray(value, dtype=float)
    data = ma.filled(array, np.nan).astype(float, copy=False)
    return data.tolist()


def _finite_optional(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def analyze_profile(prof, *, storm_motion=None) -> dict:
    """Analyze one SHARPpy-compatible profile in a single GIL-free Rust call."""
    extension = _extension()
    latitude = _finite_optional(getattr(prof, "latitude", None))
    longitude = _finite_optional(getattr(prof, "longitude", None))
    if longitude is None:
        meta = getattr(prof, "meta", None)
        if isinstance(meta, dict):
            longitude = _finite_optional(meta.get("lon", meta.get("longitude")))
    missing = _finite_optional(getattr(prof, "missing", -9999.0))
    if missing is None:
        missing = -9999.0
    if storm_motion is not None:
        storm_motion = tuple(float(value) for value in storm_motion)
        if len(storm_motion) != 4:
            raise ValueError("storm_motion must contain right/left u/v components")

    result = extension.analyze(
        _values(prof.pres),
        _values(prof.hght),
        _values(prof.tmpc),
        _values(prof.dwpc),
        _values(prof.wdir),
        _values(prof.wspd),
        omeg=_values(getattr(prof, "omeg", None), optional=True),
        latitude=latitude,
        longitude=longitude,
        missing=missing,
        storm_motion=storm_motion,
    )
    result = dict(result)
    if result.get("schema") != _SCHEMA:
        raise NativeAnalysisUnavailable(
            f"unsupported native analysis schema: {result.get('schema')!r}")
    return result


def try_analyze_profile(prof, *, storm_motion=None):
    """Return native results or ``None`` so callers can explicitly fallback."""
    try:
        return analyze_profile(prof, storm_motion=storm_motion)
    except (NativeAnalysisUnavailable, RuntimeError, ValueError) as exc:
        _LOGGER.warning("Native sounding analysis unavailable: %s", exc)
        return None


def lift_user_parcel(prof, pres, tmpc, dwpc, *, pbot=None, ptop=None):
    """Lift one interactively selected parcel through the sharprs core."""
    missing = _finite_optional(getattr(prof, "missing", -9999.0))
    if missing is None:
        missing = -9999.0
    return dict(_extension().lift_parcel(
        _values(prof.pres),
        _values(prof.hght),
        _values(prof.tmpc),
        _values(prof.dwpc),
        _values(prof.wdir),
        _values(prof.wspd),
        float(pres),
        float(tmpc),
        float(dwpc),
        pbot=_finite_optional(pbot),
        ptop=_finite_optional(ptop),
        missing=missing,
    ))
