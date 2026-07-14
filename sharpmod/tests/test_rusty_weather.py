import json
from datetime import datetime, timezone

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
