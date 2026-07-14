"""Optional Rusty Weather model-acquisition backend.

The Qt GUI remains entirely Python.  This module invokes the two bundled Rust
command-line tools, then translates their versioned point-sounding JSON into
the existing SHARPpy Reimagined ``.npz`` contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sharpmod.tools.era5_extract import (
    RetrievalError,
    _atomic_write_json,
    _atomic_write_npz,
    _quiet_remove,
)

SCHEMA = "sharpmod.point-sounding.v1"
SUPPORTED_MODELS = frozenset(("hrrr", "gfs", "rrfs-a"))


def backend_mode():
    """Return ``auto``, ``rust``, or ``python`` from the environment."""
    mode = os.environ.get("SHARPMOD_MODEL_BACKEND", "auto").strip().lower()
    return mode if mode in ("auto", "rust", "python") else "auto"


def _exe_name(stem):
    return stem + (".exe" if os.name == "nt" else "")


def _binary_dir_candidates():
    configured = os.environ.get("SHARPMOD_RUSTY_WEATHER_BIN_DIR")
    if configured:
        yield Path(configured).expanduser()
    yield Path(__file__).resolve().parents[1] / "resources" / "bin"


def find_binaries():
    """Return the ingest/export executables, or ``None`` when unavailable."""
    ingest_name = _exe_name("rw_ingest")
    export_name = _exe_name("rw_sharpmod")
    for directory in _binary_dir_candidates():
        ingest = directory / ingest_name
        export = directory / export_name
        if ingest.is_file() and export.is_file():
            return ingest, export
    ingest = shutil.which(ingest_name)
    export = shutil.which(export_name)
    if ingest and export:
        return Path(ingest), Path(export)
    return None


def is_available(model=None):
    """Whether the Rust backend is enabled, supported, and installed."""
    if backend_mode() == "python":
        return False
    if model is not None and str(model).lower() not in SUPPORTED_MODELS:
        return False
    return find_binaries() is not None


def default_cache_root():
    """Persistent cache root; repeated points reuse the same model hour."""
    configured = os.environ.get("SHARPMOD_RUSTY_WEATHER_CACHE")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "SHARPpy-Reimagined" / "rusty-weather"
    return Path.home() / ".cache" / "sharpmod" / "rusty-weather"


def _run_command(command, description):
    try:
        result = subprocess.run(
            [os.fspath(part) for part in command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise RetrievalError(f"could not start {description}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        detail = detail[-2000:]
        raise RetrievalError(f"{description} failed: {detail}")
    return result


def _run_id(run_time):
    return run_time.astimezone(timezone.utc).strftime("%Y%m%d_%Hz")


def _store_model(model):
    return str(model).lower().replace("-", "_")


def _hour_path(cache_root, model, run_time, fxx):
    return (Path(cache_root) / "store" / _store_model(model) /
            _run_id(run_time) / f"f{int(fxx):03d}.rws")


def _report(progress, message, percent):
    if progress is not None:
        progress(str(message), int(percent))


def _ensure_hour(ingest, cache_root, model, run_time, fxx, progress=None):
    hour_path = _hour_path(cache_root, model, run_time, fxx)
    if hour_path.is_file():
        _report(progress, "Using cached model hour", 60)
        return hour_path
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    _report(
        progress,
        f"Trying {str(model).upper()} {run_time:%Y-%m-%d %H}Z "
        f"F{int(fxx):03d} via native provider",
        15,
    )
    command = [
        ingest,
        "--model", model,
        "--date", run_time.strftime("%Y%m%d"),
        "--cycle", str(run_time.hour),
        "--hours", str(int(fxx)),
        "--store-root", Path(cache_root) / "store",
        "--cache-dir", Path(cache_root) / "downloads",
        "--profile", "sounding",
        "--no-heavy",
    ]
    _run_command(command, f"Rusty Weather {model.upper()} ingest")
    if not hour_path.is_file():
        raise RetrievalError(
            f"Rusty Weather completed without creating {hour_path}")
    _report(progress, "Model hour cached; extracting point", 70)
    return hour_path


def _discard_raw_downloads(cache_root):
    """Drop decoded source GRIBs after the durable `.rws` hour is written."""
    keep = os.environ.get("SHARPMOD_KEEP_RAW_GRIB", "").strip().lower()
    if keep in ("1", "true", "yes", "on"):
        return
    shutil.rmtree(Path(cache_root) / "downloads", ignore_errors=True)


def _store_limit_bytes():
    try:
        gib = float(os.environ.get("SHARPMOD_RUST_CACHE_GB", "4"))
    except ValueError:
        gib = 4.0
    return int(gib * 1024 ** 3) if gib > 0 else 0


def prune_store(cache_root, preserve=None):
    """Bound persistent `.rws` storage by removing oldest complete runs."""
    limit = _store_limit_bytes()
    store = Path(cache_root) / "store"
    if not limit or not store.is_dir():
        return
    preserve = Path(preserve).resolve() if preserve else None
    runs = []
    total = 0
    for model_dir in store.iterdir():
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            files = [path for path in run_dir.rglob("*") if path.is_file()]
            size = sum(path.stat().st_size for path in files)
            modified = max((path.stat().st_mtime for path in files), default=0.0)
            total += size
            runs.append((modified, size, run_dir))
    for _modified, size, run_dir in sorted(runs):
        if total <= limit:
            break
        if preserve is not None and run_dir.resolve() == preserve:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        total -= size


def _parse_run(value):
    try:
        return datetime.strptime(value, "%Y%m%d_%Hz").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise RetrievalError(f"Rusty Weather returned an invalid run time {value!r}") from exc


def json_to_npz(json_path, out_path, label=None, loc=None):
    """Validate one bridge payload and atomically write the standard NPZ."""
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RetrievalError(f"could not read Rusty Weather sounding: {exc}") from exc
    if payload.get("schema") != SCHEMA:
        raise RetrievalError(
            f"unsupported Rusty Weather sounding schema {payload.get('schema')!r}")

    keys = (
        "pressure_hpa", "height_m_msl", "temperature_c", "dewpoint_c",
        "u_ms", "v_ms",
    )
    columns = {key: np.asarray(payload.get(key, ()), dtype=float) for key in keys}
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 8:
        raise RetrievalError("Rusty Weather returned incomplete sounding columns")
    if not all(np.all(np.isfinite(value)) for value in columns.values()):
        raise RetrievalError("Rusty Weather returned non-finite sounding values")
    if not np.all(np.diff(columns["pressure_hpa"]) < 0):
        raise RetrievalError("Rusty Weather pressure levels are not strictly descending")

    u_ms = columns["u_ms"]
    v_ms = columns["v_ms"]
    wspd = np.hypot(u_ms, v_ms) * 1.943_844_492_440_6
    wdir = (270.0 - np.degrees(np.arctan2(v_ms, u_ms))) % 360.0
    # When the optional sharprs extension is present, use its strict native
    # Profile constructor as a second validation boundary. We do not replace
    # displayed composite parameters until their parcel definitions have
    # reference parity with the legacy GUI.
    from sharpmod.sharptab import native
    try:
        native.validate_profile(
            columns["pressure_hpa"], columns["height_m_msl"],
            columns["temperature_c"], columns["dewpoint_c"], wdir, wspd)
    except (TypeError, ValueError) as exc:
        raise RetrievalError(f"sharprs rejected the model sounding: {exc}") from exc
    run_dt = _parse_run(str(payload["run"]))
    fxx = int(payload["forecast_hour"])
    valid_unix = payload.get("valid_unix")
    valid_dt = datetime.fromtimestamp(valid_unix, timezone.utc) \
        if valid_unix is not None else run_dt + timedelta(hours=fxx)
    selected = payload.get("selected") or {}
    requested = payload.get("requested") or {}
    selected_lat = float(selected["lat"])
    selected_lon = ((float(selected["lon"]) + 180.0) % 360.0) - 180.0
    model_label = label or str(payload.get("model", "Model")).upper()
    location = loc or f"{model_label} {selected_lat:.2f}, {selected_lon:.2f}"
    run_str = run_dt.strftime("%Y-%m-%d %H:%M")
    valid_str = valid_dt.strftime("%Y-%m-%d %H:%M")

    arrays = {
        "pres": columns["pressure_hpa"],
        "hght": columns["height_m_msl"],
        "tmpc": columns["temperature_c"],
        "dwpc": np.minimum(columns["dewpoint_c"], columns["temperature_c"]),
        "wdir": wdir,
        "wspd": wspd,
        "omeg": np.full(len(wspd), -9999.0),
        "uwnd": u_ms,
        "vwnd": v_ms,
        "lat": selected_lat,
        "lon": selected_lon,
        "loc": location,
        "model": model_label,
        "run": run_str,
        "valid": valid_str,
        "fxx": fxx,
    }
    meta = {
        "backend": "rusty-weather",
        "schema": SCHEMA,
        "model": model_label,
        "model_key": str(payload.get("model", "")),
        "loc": location,
        "requested_lat": float(requested.get("lat", selected_lat)),
        "requested_lon": float(requested.get("lon", selected_lon)),
        "selected_lat": selected_lat,
        "selected_lon": selected_lon,
        "run": run_str,
        "valid": valid_str,
        "fxx": fxx,
        "levels": len(wspd),
        "npz": os.path.abspath(out_path),
    }
    _atomic_write_npz(out_path, arrays)
    sidecar = os.path.splitext(os.fspath(out_path))[0] + ".json"
    try:
        _atomic_write_json(sidecar, meta)
    except BaseException:
        _quiet_remove(out_path)
        raise
    return os.fspath(out_path), run_dt


def extract(model, lat, lon, run_time, fxx, out_path, label=None, loc=None,
            cache_root=None, progress=None):
    """Acquire/cache one model hour in Rust and export its point sounding."""
    model = str(model).lower()
    if model not in SUPPORTED_MODELS:
        raise RetrievalError(f"Rusty Weather does not support model {model!r}")
    binaries = find_binaries()
    if binaries is None:
        raise RetrievalError("Rusty Weather executables are not installed")
    ingest, exporter = binaries
    cache_root = Path(cache_root) if cache_root else default_cache_root()
    if out_path is None:
        out_path = "%s_point_%.2fN_%.2fE_%s_f%03d.npz" % (
            model.replace("-", "_"), float(lat), float(lon),
            run_time.strftime("%Y%m%d%H"), int(fxx))
    _report(progress, "Checking native model cache", 5)
    try:
        hour_path = _ensure_hour(
            ingest, cache_root, model, run_time, fxx, progress=progress)
    finally:
        # `.rws` is the durable model cache.  Provider GRIB is only an ingest
        # artifact and must not accumulate alongside it, including when a
        # just-published/missing cycle makes ingestion fail partway through.
        _discard_raw_downloads(cache_root)
    prune_store(cache_root, preserve=hour_path.parent.resolve())

    fd, bridge_path = tempfile.mkstemp(suffix=".json", prefix="sharpmod-rust-")
    os.close(fd)
    try:
        _report(progress, "Extracting nearest-grid-point sounding", 78)
        command = [
            exporter,
            "--store-root", cache_root / "store",
            "--model", model,
            "--run", _run_id(run_time),
            "--forecast-hour", str(int(fxx)),
            "--lat", repr(float(lat)),
            "--lon", repr(float(lon)),
            "--output", bridge_path,
        ]
        _run_command(command, "Rusty Weather point-sounding export")
        _report(progress, "Converting sounding for SHARPpy", 84)
        return json_to_npz(bridge_path, out_path, label=label, loc=loc)
    finally:
        _quiet_remove(bridge_path)
