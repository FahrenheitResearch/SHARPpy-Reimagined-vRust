"""Optional Rusty Weather model-acquisition backend.

The Qt GUI remains entirely Python.  This module invokes the two bundled Rust
command-line tools, then translates their versioned point-sounding JSON into
the existing SHARPpy Reimagined ``.npz`` contract.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
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
_POINT_SERVER_REQUEST_TIMEOUT = 30.0
_LOGGER = logging.getLogger(__name__)


class RustCacheMiss(RetrievalError):
    """The requested model hour is not present in the durable Rust store."""


class _PointServerTransportError(RuntimeError):
    """The optional persistent exporter protocol could not be used."""


class _RustPointServer:
    """Serialize requests through one long-lived ``rw_sharpmod`` process.

    The helper retains its mmap-backed hour reader and decompressed grid, so a
    follow-up map click performs only the sub-millisecond column read.  A lock
    keeps the protocol safe if independent GUI workers briefly overlap.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._exporter = None
        self._responses = None
        self._reader = None
        self._errors = None
        self._stderr_reader = None
        self._request_id = 0
        self._disabled = set()

    @staticmethod
    def _key(exporter):
        return os.path.normcase(os.path.abspath(os.fspath(exporter)))

    @staticmethod
    def _pump_stdout(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        except (OSError, ValueError):
            # The owner may close the pipe while stopping a timed-out helper.
            pass
        finally:
            responses.put(None)

    @staticmethod
    def _pump_stderr(stream, errors):
        """Drain stderr continuously while retaining only bounded context."""
        try:
            for line in stream:
                errors.append(line.rstrip())
        except (OSError, ValueError):
            # Normal during concurrent helper shutdown.
            pass

    def _stop_locked(self):
        process = self._process
        self._process = None
        self._exporter = None
        self._responses = None
        self._reader = None
        self._errors = None
        self._stderr_reader = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass

    def stop(self):
        with self._lock:
            self._stop_locked()

    def disable(self, exporter):
        """Use the compatible one-shot CLI for this executable this session."""
        with self._lock:
            key = self._key(exporter)
            self._disabled.add(key)
            if self._exporter == key:
                self._stop_locked()

    def _start_locked(self, exporter):
        key = self._key(exporter)
        if key in self._disabled:
            raise _PointServerTransportError(
                "persistent export is disabled for this helper")
        if self._process is not None and self._exporter == key \
                and self._process.poll() is None:
            return
        self._stop_locked()
        responses = queue.Queue()
        errors = deque(maxlen=40)
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [os.fspath(exporter), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **kwargs,
            )
        except OSError as exc:
            raise _PointServerTransportError(
                f"could not start persistent point exporter: {exc}") from exc
        reader = threading.Thread(
            target=self._pump_stdout,
            args=(process.stdout, responses),
            name="rw-sharpmod-stdout",
            daemon=True,
        )
        reader.start()
        stderr_reader = threading.Thread(
            target=self._pump_stderr,
            args=(process.stderr, errors),
            name="rw-sharpmod-stderr",
            daemon=True,
        )
        stderr_reader.start()
        self._process = process
        self._exporter = key
        self._responses = responses
        self._reader = reader
        self._errors = errors
        self._stderr_reader = stderr_reader

    def _request(self, exporter, action, cache_root, model, run_time, fxx,
                 *, lat=None, lon=None, output=None, cancelled=None):
        with self._lock:
            self._start_locked(exporter)
            if cancelled is not None and cancelled():
                self._stop_locked()
                from sharpmod.model_transport import DownloadCancelled
                raise DownloadCancelled(
                    "Rusty Weather point-sounding export cancelled")
            self._request_id += 1
            request_id = self._request_id
            request = {
                "request_id": request_id,
                "action": str(action),
                "store_root": os.fspath(Path(cache_root) / "store"),
                "model": str(model),
                "run": _run_id(run_time),
                "forecast_hour": int(fxx),
            }
            if lat is not None:
                request["lat"] = float(lat)
            if lon is not None:
                request["lon"] = float(lon)
            if output is not None:
                request["output"] = os.fspath(output)
            process = self._process
            responses = self._responses
            deadline = time.monotonic() + _POINT_SERVER_REQUEST_TIMEOUT
            try:
                process.stdin.write(json.dumps(request, separators=(",", ":")))
                process.stdin.write("\n")
                process.stdin.flush()
            except (AttributeError, OSError, BrokenPipeError) as exc:
                self._stop_locked()
                raise _PointServerTransportError(
                    f"persistent point exporter pipe failed: {exc}") from exc

            while True:
                if cancelled is not None and cancelled():
                    self._stop_locked()
                    from sharpmod.model_transport import DownloadCancelled
                    raise DownloadCancelled(
                        "Rusty Weather point-sounding export cancelled")
                try:
                    line = responses.get(timeout=0.05)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        detail = "; ".join(self._errors or ())
                        self._stop_locked()
                        message = (
                            "persistent point exporter timed out after "
                            f"{_POINT_SERVER_REQUEST_TIMEOUT:.1f}s")
                        if detail:
                            message += f": {detail[-2000:]}"
                        raise _PointServerTransportError(message)
                    if process.poll() is not None:
                        detail = "; ".join(self._errors or ())
                        self._stop_locked()
                        message = (
                            "persistent point exporter exited without a response")
                        if detail:
                            message += f": {detail[-2000:]}"
                        raise _PointServerTransportError(message)
                    continue
                if line is None:
                    detail = "; ".join(self._errors or ())
                    self._stop_locked()
                    message = "persistent point exporter closed its output"
                    if detail:
                        message += f": {detail[-2000:]}"
                    raise _PointServerTransportError(message)
                try:
                    response = json.loads(line)
                except ValueError as exc:
                    self._stop_locked()
                    raise _PointServerTransportError(
                        "persistent point exporter returned invalid JSON") from exc
                if response.get("request_id") != request_id:
                    self._stop_locked()
                    raise _PointServerTransportError(
                        "persistent point exporter response was out of sequence")
                if not response.get("ok"):
                    raise RetrievalError(
                        "Rusty Weather point-sounding export failed: "
                        + str(response.get("error") or "unknown error"))
                return response

    def export(self, exporter, cache_root, model, run_time, fxx, lat, lon,
               output, cancelled=None):
        return self._request(
            exporter, "export", cache_root, model, run_time, fxx,
            lat=lat, lon=lon, output=output, cancelled=cancelled)

    def warm(self, exporter, cache_root, model, run_time, fxx,
             cancelled=None):
        return self._request(
            exporter, "warm", cache_root, model, run_time, fxx,
            cancelled=cancelled)


_POINT_SERVER = _RustPointServer()
atexit.register(_POINT_SERVER.stop)


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


def _run_command(command, description, cancelled=None):
    """Run one helper while allowing the Qt worker to stop the subprocess."""
    try:
        process = subprocess.Popen(
            [os.fspath(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise RetrievalError(f"could not start {description}: {exc}") from exc

    while True:
        if cancelled is not None and cancelled():
            process.terminate()
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            from sharpmod.model_transport import DownloadCancelled
            raise DownloadCancelled(f"{description} cancelled")
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            continue

    result = subprocess.CompletedProcess(
        process.args, process.returncode, stdout, stderr)
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


def hour_is_cached(model, run_time, fxx, cache_root=None):
    """Whether an exact model/run/hour can be exported without downloading."""
    if str(model).lower() not in SUPPORTED_MODELS:
        return False
    root = Path(cache_root) if cache_root else default_cache_root()
    return _hour_path(root, model, run_time, fxx).is_file()


def _report(progress, message, percent):
    if progress is not None:
        progress(str(message), int(percent))


def _cold_source_candidates(model):
    """Return efficient indexed providers for an explicitly cold Rust ingest.

    Rusty Weather's NOMADS production path downloads whole GRIB files.  AWS
    and Google use indexed byte ranges, so they are preferred when a caller
    explicitly requests a full-hour native cache build.  Normal interactive
    fetching uses the smaller upstream point/subregion transports before this
    cold path.
    """
    configured = os.environ.get("SHARPMOD_RUSTY_WEATHER_SOURCE", "").strip()
    if configured:
        return (configured,)
    if str(model).lower() in {"hrrr", "gfs"}:
        return ("aws", "google")
    if str(model).lower() == "rrfs-a":
        return ("aws",)
    return ()


def _ensure_hour(ingest, cache_root, model, run_time, fxx, progress=None,
                 cancelled=None):
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
    failures = []
    for source in _cold_source_candidates(model):
        command = [
            ingest,
            "--model", model,
            "--date", run_time.strftime("%Y%m%d"),
            "--cycle", str(run_time.hour),
            "--hours", str(int(fxx)),
            "--source", source,
            "--store-root", Path(cache_root) / "store",
            "--cache-dir", Path(cache_root) / "downloads",
            "--profile", "sounding",
            "--no-heavy",
        ]
        try:
            _run_command(
                command, f"Rusty Weather {model.upper()} {source} ingest",
                cancelled=cancelled)
            break
        except RetrievalError as exc:
            failures.append(f"{source}: {exc}")
    else:
        detail = "; ".join(failures) or "no indexed native provider"
        raise RetrievalError(
            f"Rusty Weather could not cache {model.upper()} from an indexed "
            f"provider: {detail}")
    if not hour_path.is_file():
        raise RetrievalError(
            f"Rusty Weather completed without creating {hour_path}")
    _report(progress, "Model hour cached", 70)
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


def _export_cached_point(exporter, cache_root, model, run_time, fxx, lat, lon,
                         out_path, label=None, loc=None, progress=None,
                         cancelled=None):
    """Export one already-cached ``.rws`` hour through the JSON bridge."""
    fd, bridge_path = tempfile.mkstemp(suffix=".json", prefix="sharpmod-rust-")
    os.close(fd)
    try:
        _report(progress, "Extracting nearest-grid-point sounding", 78)
        started = time.perf_counter()
        try:
            _POINT_SERVER.export(
                exporter, cache_root, model, run_time, fxx, lat, lon,
                bridge_path, cancelled=cancelled)
            mode = "persistent"
        except _PointServerTransportError as exc:
            # External/older rw_sharpmod binaries retain their documented
            # one-shot CLI. Remember the incompatibility for this process so
            # later clicks do not repeatedly attempt an unsupported protocol.
            _POINT_SERVER.disable(exporter)
            _LOGGER.warning(
                "persistent Rust point exporter unavailable; using one-shot "
                "fallback: %s", exc)
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
            _run_command(
                command, "Rusty Weather point-sounding export",
                cancelled=cancelled)
            mode = "one-shot"
        _LOGGER.info(
            "rusty_weather.cached_export mode=%s elapsed_ms=%.1f model=%s "
            "run=%s fxx=%03d",
            mode, (time.perf_counter() - started) * 1000.0, model,
            _run_id(run_time), int(fxx))
        _report(progress, "Converting sounding for SHARPpy", 84)
        return json_to_npz(bridge_path, out_path, label=label, loc=loc)
    finally:
        _quiet_remove(bridge_path)


def extract_cached(model, lat, lon, run_time, fxx, out_path, label=None,
                   loc=None, cache_root=None, progress=None, cancelled=None):
    """Export an exact native cache hit without initiating network I/O."""
    model = str(model).lower()
    if model not in SUPPORTED_MODELS:
        raise RetrievalError(f"Rusty Weather does not support model {model!r}")
    binaries = find_binaries()
    if binaries is None:
        raise RetrievalError("Rusty Weather executables are not installed")
    _ingest, exporter = binaries
    cache_root = Path(cache_root) if cache_root else default_cache_root()
    if out_path is None:
        out_path = "%s_point_%.2fN_%.2fE_%s_f%03d.npz" % (
            model.replace("-", "_"), float(lat), float(lon),
            run_time.strftime("%Y%m%d%H"), int(fxx))
    _report(progress, "Checking native model cache", 5)
    hour_path = _hour_path(cache_root, model, run_time, fxx)
    if not hour_path.is_file():
        raise RustCacheMiss(
            f"native model hour is not cached: {model} {_run_id(run_time)} "
            f"F{int(fxx):03d}")
    _report(progress, "Using cached model hour", 60)
    result = _export_cached_point(
        exporter, cache_root, model, run_time, fxx, lat, lon, out_path,
        label=label, loc=loc, progress=progress, cancelled=cancelled)
    # The server has switched away from any prior hour before pruning. This
    # matters on Windows, where an mmap-backed old `.rws` cannot be removed
    # while the helper still has it open.
    prune_store(cache_root, preserve=hour_path.parent.resolve())
    return result


def cache_hour(model, run_time, fxx, cache_root=None, progress=None,
               cancelled=None):
    """Build and retain one exact native model-hour cache without exporting.

    This is the explicit map-browsing path used by the GUI's **Cache This
    Hour** action.  It deliberately performs the larger full-hour ingest, but
    does not create a throwaway point sounding.  Once complete,
    :func:`extract_cached` can serve any map point from the resulting ``.rws``
    file without another provider request.
    """
    model = str(model).lower()
    if model not in SUPPORTED_MODELS:
        raise RetrievalError(f"Rusty Weather does not support model {model!r}")
    binaries = find_binaries()
    if binaries is None:
        raise RetrievalError("Rusty Weather executables are not installed")
    ingest, exporter = binaries
    cache_root = Path(cache_root) if cache_root else default_cache_root()
    # Release any mmap before an ingest may replace or prune its store file.
    _POINT_SERVER.stop()
    try:
        hour_path = _ensure_hour(
            ingest, cache_root, model, run_time, fxx, progress=progress,
            cancelled=cancelled)
    finally:
        # The durable cache is the decoded .rws file, not the source GRIB.
        # This also clears partial provider artifacts after cancellation or a
        # not-yet-published run fails.
        _discard_raw_downloads(cache_root)
    prune_store(cache_root, preserve=hour_path.parent.resolve())
    try:
        warm_started = time.perf_counter()
        _POINT_SERVER.warm(
            exporter, cache_root, model, run_time, fxx,
            cancelled=cancelled)
        _LOGGER.info(
            "rusty_weather.cache_warm elapsed_ms=%.1f model=%s run=%s "
            "fxx=%03d",
            (time.perf_counter() - warm_started) * 1000.0, model,
            _run_id(run_time), int(fxx))
    except _PointServerTransportError as exc:
        _POINT_SERVER.disable(exporter)
        _LOGGER.warning(
            "persistent Rust point exporter could not be warmed; cached "
            "one-shot export remains available: %s", exc)
    except RetrievalError as exc:
        # Caching succeeded; failure of the optional latency optimization
        # must never turn a valid durable hour into a failed cache operation.
        _LOGGER.warning("cached hour could not be pre-opened: %s", exc)
    _report(progress, "Model hour ready for fast map browsing", 100)
    return hour_path


def extract(model, lat, lon, run_time, fxx, out_path, label=None, loc=None,
            cache_root=None, progress=None, cancelled=None):
    """Build a full-hour native cache when needed, then export one point.

    This deliberately expensive path is used only when Rust is explicitly
    forced or after the optimized point/subregion providers fail.  Interactive
    ``auto`` mode calls :func:`extract_cached` first instead.
    """
    model = str(model).lower()
    if model not in SUPPORTED_MODELS:
        raise RetrievalError(f"Rusty Weather does not support model {model!r}")
    binaries = find_binaries()
    if binaries is None:
        raise RetrievalError("Rusty Weather executables are not installed")
    _ingest, exporter = binaries
    cache_root = Path(cache_root) if cache_root else default_cache_root()
    if out_path is None:
        out_path = "%s_point_%.2fN_%.2fE_%s_f%03d.npz" % (
            model.replace("-", "_"), float(lat), float(lon),
            run_time.strftime("%Y%m%d%H"), int(fxx))
    _report(progress, "Checking native model cache", 5)
    cache_hour(
        model, run_time, fxx, cache_root=cache_root, progress=progress,
        cancelled=cancelled)
    return _export_cached_point(
        exporter, cache_root, model, run_time, fxx, lat, lon, out_path,
        label=label, loc=loc, progress=progress, cancelled=cancelled)
