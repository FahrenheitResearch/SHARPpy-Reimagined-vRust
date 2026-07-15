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


def streamwiseness(height_msl_m, u_kts, v_kts, *, sfc, storm_motion,
                   max_height_m=6000.0, step_m=100.0):
    """Calculate streamwiseness in Rust and return three NumPy arrays.

    The returned tuple is ``(height_m, percent, signed_percent)``. ``percent``
    is ``(omega_streamwise / omega_horizontal_mag) ** 2 * 100``;
    ``signed_percent`` carries the projection sign only for directional
    shading. ``None`` means the supplied profile has no usable streamwiseness
    samples; absence of the native feature raises
    :class:`NativeAnalysisUnavailable` so the visualization layer can select
    its explicit Python fallback.
    """
    extension = _extension()
    function = getattr(extension, "streamwiseness", None)
    if function is None:
        raise NativeAnalysisUnavailable(
            "native extension does not expose streamwiseness")

    arrays = []
    for name, values in (
            ("height_msl_m", height_msl_m),
            ("u_kts", u_kts),
            ("v_kts", v_kts)):
        array = np.asarray(
            ma.filled(ma.asarray(values, dtype=float), np.nan), dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        arrays.append(array)
    if not (len(arrays[0]) == len(arrays[1]) == len(arrays[2])):
        raise ValueError("height_msl_m, u_kts, and v_kts must have equal lengths")

    try:
        motion = tuple(float(value) for value in storm_motion)
    except (TypeError, ValueError) as exc:
        raise ValueError("storm_motion must contain two finite components") from exc
    if len(motion) != 2 or not np.all(np.isfinite(motion)):
        raise ValueError("storm_motion must contain two finite components")

    result = function(
        *[array.tolist() for array in arrays],
        int(sfc),
        motion[0],
        motion[1],
        max_height_m=float(max_height_m),
        step_m=float(step_m),
    )
    if result is None:
        return None
    if len(result) != 3:
        raise NativeAnalysisUnavailable("malformed native streamwiseness result")
    converted = tuple(np.asarray(values, dtype=float) for values in result)
    if any(values.ndim != 1 for values in converted) or not (
            len(converted[0]) == len(converted[1]) == len(converted[2])):
        raise NativeAnalysisUnavailable("malformed native streamwiseness arrays")
    return converted


def _values(value, *, optional=False):
    if value is None:
        return None if optional else []
    array = ma.asarray(value, dtype=float)
    data = ma.filled(array, np.nan).astype(float, copy=False)
    return data.tolist()


def _interpolate_internal_winds(pres, wdir, wspd):
    """Fill only bracketed missing winds using SHARPpy's log-p convention.

    Observed soundings commonly report thermodynamic levels between mandatory
    wind levels.  Legacy SHARPpy interpolates vector components across those
    internal gaps; passing the gaps to the stricter Rust profile unchanged can
    make an otherwise valid hodograph and every storm-relative diagnostic
    undefined.  Work in u/v space so directions crossing north interpolate
    correctly, and deliberately leave leading/trailing gaps untouched.
    """
    pres = np.asarray(pres, dtype=float)
    wdir = np.asarray(wdir, dtype=float).copy()
    wspd = np.asarray(wspd, dtype=float).copy()
    valid = (
        np.isfinite(pres) & (pres > 0.0)
        & np.isfinite(wdir) & np.isfinite(wspd) & (wspd >= 0.0)
    )
    missing = (
        np.isfinite(pres) & (pres > 0.0)
        & ~(np.isfinite(wdir) & np.isfinite(wspd))
    )
    if np.count_nonzero(valid) < 2 or not np.any(missing):
        return wdir, wspd

    radians = np.deg2rad(wdir[valid])
    u_valid = -wspd[valid] * np.sin(radians)
    v_valid = -wspd[valid] * np.cos(radians)
    logp_valid = np.log(pres[valid])
    order = np.argsort(logp_valid)
    logp_valid = logp_valid[order]
    u_valid = u_valid[order]
    v_valid = v_valid[order]

    logp = np.log(pres[missing])
    bracketed = (
        (logp >= logp_valid[0]) & (logp <= logp_valid[-1])
    )
    if not np.any(bracketed):
        return wdir, wspd
    missing_indices = np.flatnonzero(missing)[bracketed]
    targets = logp[bracketed]
    u = np.interp(targets, logp_valid, u_valid)
    v = np.interp(targets, logp_valid, v_valid)
    wspd[missing_indices] = np.hypot(u, v)
    wdir[missing_indices] = (
        np.rad2deg(np.arctan2(-u, -v)) + 360.0
    ) % 360.0
    return wdir, wspd


def _normalized_columns(prof):
    """Return the physical SHARPpy profile seen by the native analyzer.

    ``BasicProfile.sfc`` can follow one or more placeholder pressure levels.
    Slice those non-physical leading rows before crossing the bridge, then
    interpolate only internal (never exterior) missing wind observations.
    """
    names = ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd")
    columns = {
        name: np.asarray(
            ma.filled(ma.asarray(getattr(prof, name), dtype=float), np.nan),
            dtype=float,
        )
        for name in names
    }
    size = len(columns["pres"])
    if any(len(values) != size for values in columns.values()):
        raise ValueError("sounding columns must have identical lengths")
    start = int(getattr(prof, "sfc", 0) or 0)
    if start < 0 or start >= size:
        raise ValueError("profile surface index is outside the sounding")
    columns = {name: values[start:] for name, values in columns.items()}
    # Preserve which vectors were actually reported before filling bracketed
    # gaps for Rust's interpolation-heavy kinematic calculations. Diagnostics
    # such as PBL maximum wind must ignore those synthetic interpolation
    # levels just as SHARPpy's masked observation array does.
    columns["observed_wind_valid"] = (
        np.isfinite(columns["pres"])
        & (columns["pres"] > 0.0)
        & np.isfinite(columns["wdir"])
        & np.isfinite(columns["wspd"])
        & (columns["wspd"] >= 0.0)
    )
    columns["wdir"], columns["wspd"] = _interpolate_internal_winds(
        columns["pres"], columns["wdir"], columns["wspd"])

    omeg = getattr(prof, "omeg", None)
    if omeg is None:
        columns["omeg"] = None
    else:
        omeg = np.asarray(
            ma.filled(ma.asarray(omeg, dtype=float), np.nan), dtype=float)
        if len(omeg) != size:
            raise ValueError("omeg must have the same length as pres")
        columns["omeg"] = omeg[start:]
    return columns


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

    columns = _normalized_columns(prof)
    result = extension.analyze(
        *[columns[name].tolist() for name in (
            "pres", "hght", "tmpc", "dwpc", "wdir", "wspd")],
        omeg=(None if columns["omeg"] is None
              else columns["omeg"].tolist()),
        observed_wind_valid=columns["observed_wind_valid"].tolist(),
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
    columns = _normalized_columns(prof)
    return dict(_extension().lift_parcel(
        *[columns[name].tolist() for name in (
            "pres", "hght", "tmpc", "dwpc", "wdir", "wspd")],
        float(pres),
        float(tmpc),
        float(dwpc),
        pbot=_finite_optional(pbot),
        ptop=_finite_optional(ptop),
        missing=missing,
    ))
