"""Unit tests for forecast-model backend selection order.

These tests deliberately replace every retrieval boundary so they exercise the
ordering policy without touching a provider, local native binary, or cache.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sharpmod.tools import model_extract, rusty_weather


RUN = datetime(2026, 7, 15, 7, tzinfo=timezone.utc)


def _unexpected(name):
    def fail(*_args, **_kwargs):
        raise AssertionError(f"{name} must not be called")

    return fail


def _set_native_available(monkeypatch, mode):
    monkeypatch.setattr(rusty_weather, "backend_mode", lambda: mode)
    monkeypatch.setattr(rusty_weather, "is_available", lambda _model: True)


def test_auto_cached_rust_wins_without_python_or_cold_rust(
        monkeypatch, tmp_path):
    output = tmp_path / "cached-rust.npz"
    calls = []
    _set_native_available(monkeypatch, "auto")

    def cached(_model, _lat, _lon, *, run_time, fxx, out_path, **_kwargs):
        calls.append("cached-rust")
        assert run_time == RUN
        assert fxx == 0
        assert out_path == output
        return str(output), run_time

    monkeypatch.setattr(rusty_weather, "hour_is_cached", lambda *_args: True)
    monkeypatch.setattr(rusty_weather, "extract_cached", cached)
    monkeypatch.setattr(rusty_weather, "extract", _unexpected("cold Rust"))
    monkeypatch.setattr(model_extract, "probe", _unexpected("Python probe"))
    monkeypatch.setattr(model_extract, "extract", _unexpected("Python extract"))

    result = model_extract.extract_with_cycle_fallback(
        "hrrr", 35.2, -97.4, run_time=RUN, fxx=0,
        out_path=output, max_cycles=1)

    assert result == (str(output), RUN)
    assert calls == ["cached-rust"]


def test_uncached_auto_uses_python_without_cold_rust_when_python_succeeds(
        monkeypatch, tmp_path):
    output = tmp_path / "python.npz"
    calls = []
    _set_native_available(monkeypatch, "auto")

    def probe(*_args, **_kwargs):
        calls.append("python-probe")
        return {"available": True}

    def python_extract(*_args, **kwargs):
        calls.append("python-extract")
        assert kwargs["run_time"] == RUN
        return str(output)

    monkeypatch.setattr(rusty_weather, "hour_is_cached", lambda *_args: False)
    monkeypatch.setattr(
        rusty_weather, "extract_cached", _unexpected("cached Rust"))
    monkeypatch.setattr(rusty_weather, "extract", _unexpected("cold Rust"))
    monkeypatch.setattr(model_extract, "probe", probe)
    monkeypatch.setattr(model_extract, "extract", python_extract)

    result = model_extract.extract_with_cycle_fallback(
        "hrrr", 35.2, -97.4, run_time=RUN, fxx=0,
        out_path=output, max_cycles=1)

    assert result == (str(output), RUN)
    assert calls == ["python-probe", "python-extract"]


def test_auto_python_failure_invokes_cold_rust_fallback(monkeypatch, tmp_path):
    output = tmp_path / "cold-rust-fallback.npz"
    calls = []
    _set_native_available(monkeypatch, "auto")

    def python_extract(*_args, **_kwargs):
        calls.append("python-extract")
        raise model_extract.RetrievalError("point providers unavailable")

    def cold_rust(
            _model, _lat, _lon, *, run_time, fxx, out_path, **_kwargs):
        calls.append("cold-rust")
        assert run_time == RUN
        assert fxx == 0
        assert out_path == output
        return str(output), run_time

    monkeypatch.setattr(rusty_weather, "hour_is_cached", lambda *_args: False)
    monkeypatch.setattr(
        rusty_weather, "extract_cached", _unexpected("cached Rust"))
    monkeypatch.setattr(rusty_weather, "extract", cold_rust)
    monkeypatch.setattr(
        model_extract, "probe", lambda *_args, **_kwargs: {"available": True})
    monkeypatch.setattr(model_extract, "extract", python_extract)

    result = model_extract.extract_with_cycle_fallback(
        "hrrr", 35.2, -97.4, run_time=RUN, fxx=0,
        out_path=output, max_cycles=1)

    assert result == (str(output), RUN)
    assert calls == ["python-extract", "cold-rust"]


def test_forced_rust_skips_python_and_invokes_cold_rust(monkeypatch, tmp_path):
    output = tmp_path / "forced-rust.npz"
    calls = []
    _set_native_available(monkeypatch, "rust")

    def cold_rust(
            _model, _lat, _lon, *, run_time, fxx, out_path, **_kwargs):
        calls.append("cold-rust")
        assert run_time == RUN
        assert fxx == 0
        assert out_path == output
        return str(output), run_time

    monkeypatch.setattr(
        rusty_weather, "hour_is_cached", _unexpected("cache check"))
    monkeypatch.setattr(
        rusty_weather, "extract_cached", _unexpected("cached Rust"))
    monkeypatch.setattr(rusty_weather, "extract", cold_rust)
    monkeypatch.setattr(model_extract, "probe", _unexpected("Python probe"))
    monkeypatch.setattr(model_extract, "extract", _unexpected("Python extract"))

    result = model_extract.extract_with_cycle_fallback(
        "hrrr", 35.2, -97.4, run_time=RUN, fxx=0,
        out_path=output, max_cycles=1)

    assert result == (str(output), RUN)
    assert calls == ["cold-rust"]
