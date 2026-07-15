import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sharpmod.tools import model_extract, rusty_weather


def _payload():
    levels = list(range(1000, 799, -25))
    count = len(levels)
    return {
        "schema": rusty_weather.SCHEMA,
        "model": "hrrr",
        "run": "20260713_22z",
        "forecast_hour": 1,
        "valid_unix": None,
        "requested": {"lat": 35.2, "lon": -97.4},
        "selected": {"lat": 35.19, "lon": 262.61},
        "pressure_hpa": levels,
        "height_m_msl": list(np.linspace(300, 2100, count)),
        "temperature_c": list(np.linspace(25, 10, count)),
        "dewpoint_c": list(np.linspace(20, 5, count)),
        "u_ms": [5.0] * count,
        "v_ms": [0.0] * count,
    }


def test_json_bridge_writes_existing_npz_contract(tmp_path):
    bridge = tmp_path / "bridge.json"
    bridge.write_text(json.dumps(_payload()), encoding="utf-8")
    output = tmp_path / "sounding.npz"

    path, resolved = rusty_weather.json_to_npz(
        bridge, output, label="HRRR", loc="Norman")

    assert path == str(output)
    assert resolved == datetime(2026, 7, 13, 22, tzinfo=timezone.utc)
    with np.load(output) as data:
        assert str(data["model"]) == "HRRR"
        assert str(data["loc"]) == "Norman"
        assert str(data["valid"]) == "2026-07-13 23:00"
        assert np.allclose(data["wspd"], 9.719222462203)
        assert np.allclose(data["wdir"], 270.0)
        assert np.all(data["omeg"] == -9999.0)
        assert float(data["lon"]) == pytest.approx(-97.39)
    sidecar = json.loads(output.with_suffix(".json").read_text("utf-8"))
    assert sidecar["backend"] == "rusty-weather"
    assert sidecar["levels"] == 9


def test_json_bridge_rejects_non_descending_pressure(tmp_path):
    payload = _payload()
    payload["pressure_hpa"] = sorted(payload["pressure_hpa"])
    bridge = tmp_path / "bridge.json"
    bridge.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(model_extract.RetrievalError, match="strictly descending"):
        rusty_weather.json_to_npz(bridge, tmp_path / "bad.npz")


def test_cached_point_prefers_persistent_exporter(monkeypatch, tmp_path):
    run = datetime(2026, 7, 13, 22, tzinfo=timezone.utc)
    output = tmp_path / "point.npz"
    calls = []

    def persistent(
            exporter, cache_root, model, run_time, fxx, lat, lon,
            bridge_path, cancelled=None):
        calls.append((
            exporter, cache_root, model, run_time, fxx, lat, lon,
            cancelled))
        Path(bridge_path).write_text(json.dumps(_payload()), encoding="utf-8")

    monkeypatch.setattr(rusty_weather._POINT_SERVER, "export", persistent)
    monkeypatch.setattr(
        rusty_weather, "_run_command",
        lambda *_args, **_kwargs: pytest.fail("one-shot fallback was used"))

    path, resolved = rusty_weather._export_cached_point(
        "rw_sharpmod", tmp_path, "hrrr", run, 1, 35.2, -97.4,
        output, label="HRRR")

    assert path == str(output)
    assert resolved == run
    assert calls == [(
        "rw_sharpmod", tmp_path, "hrrr", run, 1, 35.2, -97.4,
        None)]


def test_cached_point_falls_back_for_legacy_exporter(monkeypatch, tmp_path):
    run = datetime(2026, 7, 13, 22, tzinfo=timezone.utc)
    output = tmp_path / "fallback.npz"
    disabled = []
    commands = []

    monkeypatch.setattr(
        rusty_weather._POINT_SERVER, "export",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            rusty_weather._PointServerTransportError("no server mode")))
    monkeypatch.setattr(
        rusty_weather._POINT_SERVER, "disable",
        lambda exporter: disabled.append(exporter))

    def one_shot(command, _description, cancelled=None):
        commands.append((command, cancelled))
        bridge_path = Path(command[command.index("--output") + 1])
        bridge_path.write_text(json.dumps(_payload()), encoding="utf-8")

    monkeypatch.setattr(rusty_weather, "_run_command", one_shot)

    path, resolved = rusty_weather._export_cached_point(
        "legacy_rw_sharpmod", tmp_path, "hrrr", run, 1, 35.2, -97.4,
        output, label="HRRR")

    assert path == str(output)
    assert resolved == run
    assert disabled == ["legacy_rw_sharpmod"]
    assert commands[0][0][0] == "legacy_rw_sharpmod"


class _FakePipe:
    def __init__(self):
        self.text = ""
        self.closed = False

    def write(self, value):
        self.text += value

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _HungProcess:
    def __init__(self):
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None

    def wait(self, timeout=None):
        self.stopped = True
        return 0

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.stopped = True


def _install_hung_server(monkeypatch, server, process, tmp_path):
    def start(exporter):
        server._process = process
        server._exporter = server._key(exporter)
        server._responses = rusty_weather.queue.Queue()
        server._errors = rusty_weather.deque(maxlen=40)

    monkeypatch.setattr(server, "_start_locked", start)
    return dict(
        exporter=tmp_path / "rw_sharpmod",
        cache_root=tmp_path,
        model="hrrr",
        run_time=datetime(2026, 7, 13, 22, tzinfo=timezone.utc),
        fxx=1,
        lat=35.2,
        lon=-97.4,
        output=tmp_path / "bridge.json",
    )


def test_persistent_exporter_has_bounded_request_deadline(
        monkeypatch, tmp_path):
    server = rusty_weather._RustPointServer()
    process = _HungProcess()
    request = _install_hung_server(
        monkeypatch, server, process, tmp_path)
    monkeypatch.setattr(
        rusty_weather, "_POINT_SERVER_REQUEST_TIMEOUT", 0.01)

    with pytest.raises(
            rusty_weather._PointServerTransportError, match="timed out"):
        server.export(**request)

    assert process.stopped


def test_persistent_exporter_cancellation_stops_hung_helper(
        monkeypatch, tmp_path):
    from sharpmod.model_transport import DownloadCancelled

    server = rusty_weather._RustPointServer()
    process = _HungProcess()
    request = _install_hung_server(
        monkeypatch, server, process, tmp_path)
    checks = iter((False, True))
    request["cancelled"] = lambda: next(checks, True)

    with pytest.raises(DownloadCancelled, match="cancelled"):
        server.export(**request)

    assert process.stopped


def test_explicit_rust_backend_reports_missing_binaries(monkeypatch):
    monkeypatch.setenv("SHARPMOD_MODEL_BACKEND", "rust")
    monkeypatch.setattr(rusty_weather, "find_binaries", lambda: None)

    with pytest.raises(model_extract.RetrievalError, match="executables were not found"):
        model_extract.extract_with_cycle_fallback(
            "hrrr", 35.2, -97.4,
            run_time=datetime(2026, 7, 13, 22, tzinfo=timezone.utc),
            fxx=0,
        )


def test_auto_backend_falls_back_to_python(monkeypatch, tmp_path):
    run = datetime(2026, 7, 13, 22, tzinfo=timezone.utc)
    output = tmp_path / "fallback.npz"
    monkeypatch.setenv("SHARPMOD_MODEL_BACKEND", "auto")
    monkeypatch.setattr(rusty_weather, "find_binaries", lambda: ("ingest", "export"))
    monkeypatch.setattr(
        rusty_weather, "extract",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_extract.RetrievalError("Rust fetch unavailable")))
    monkeypatch.setattr(
        model_extract, "probe",
        lambda *args, **kwargs: {"available": True})
    monkeypatch.setattr(model_extract, "extract", lambda *args, **kwargs: str(output))

    path, resolved = model_extract.extract_with_cycle_fallback(
        "hrrr", 35.2, -97.4, run_time=run, fxx=0, out_path=output,
        max_cycles=1)

    assert path == str(output)
    assert resolved == run


def test_raw_downloads_are_removed_unless_preserved(monkeypatch, tmp_path):
    raw = tmp_path / "downloads" / "provider" / "fetch.grib2"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"grib")
    rusty_weather._discard_raw_downloads(tmp_path)
    assert not raw.exists()

    monkeypatch.setenv("SHARPMOD_KEEP_RAW_GRIB", "1")
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"grib")
    rusty_weather._discard_raw_downloads(tmp_path)
    assert raw.exists()


def test_failed_ingest_does_not_leave_raw_grib(monkeypatch, tmp_path):
    raw = tmp_path / "downloads" / "provider" / "fetch.grib2"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"partial grib")
    monkeypatch.setattr(rusty_weather, "find_binaries", lambda: ("ingest", "export"))
    monkeypatch.setattr(
        rusty_weather,
        "_ensure_hour",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            model_extract.RetrievalError("not published")),
    )

    with pytest.raises(model_extract.RetrievalError, match="not published"):
        rusty_weather.extract(
            "hrrr", 35.2, -97.4,
            datetime(2026, 7, 14, 0, tzinfo=timezone.utc), 4,
            tmp_path / "sounding.npz", cache_root=tmp_path,
        )

    assert not raw.exists()


def test_cache_hour_builds_durable_rws_without_exporting_point(
        monkeypatch, tmp_path):
    run = datetime(2026, 7, 14, 0, tzinfo=timezone.utc)
    hour = tmp_path / "store" / "hrrr" / "20260714_00z" / "f004.rws"
    raw = tmp_path / "downloads" / "provider" / "fetch.grib2"
    events = []

    def ensure_hour(
            ingest, cache_root, model, run_time, fxx, progress=None,
            cancelled=None):
        assert ingest == "rw_ingest"
        assert cache_root == tmp_path
        assert (model, run_time, fxx) == ("hrrr", run, 4)
        assert cancelled is None
        hour.parent.mkdir(parents=True)
        hour.write_bytes(b"native hour")
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"temporary grib")
        return hour

    monkeypatch.setattr(
        rusty_weather, "find_binaries",
        lambda: ("rw_ingest", "rw_sharpmod"))
    monkeypatch.setattr(rusty_weather, "_ensure_hour", ensure_hour)
    stopped = []
    warmed = []
    monkeypatch.setattr(
        rusty_weather._POINT_SERVER, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(
        rusty_weather._POINT_SERVER, "warm",
        lambda *args, **kwargs: warmed.append((args, kwargs)))

    result = rusty_weather.cache_hour(
        "hrrr", run, 4, cache_root=tmp_path,
        progress=lambda message, percent: events.append((message, percent)),
    )

    assert result == hour
    assert hour.read_bytes() == b"native hour"
    assert not raw.exists()
    assert events[-1] == (
        "Model hour ready for fast map browsing", 100)
    assert stopped == [True]
    assert warmed == [((
        "rw_sharpmod", tmp_path, "hrrr", run, 4), {
            "cancelled": None,
        })]


def test_store_pruning_preserves_active_run(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARPMOD_RUST_CACHE_GB", "0.00000001")
    old_run = tmp_path / "store" / "hrrr" / "old"
    active_run = tmp_path / "store" / "hrrr" / "active"
    old_run.mkdir(parents=True)
    active_run.mkdir(parents=True)
    (old_run / "f000.rws").write_bytes(b"x" * 100)
    (active_run / "f000.rws").write_bytes(b"y" * 100)

    rusty_weather.prune_store(tmp_path, preserve=active_run)

    assert not old_run.exists()
    assert active_run.exists()
