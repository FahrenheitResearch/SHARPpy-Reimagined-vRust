"""Tests for the standalone ecape-rs analytic bridge and fallback order."""

from pathlib import Path

import numpy as np
import pytest

from sharpmod.io.decoder import load_npz
from sharpmod.sharptab import ecape as ecape_mod
from sharpmod.sharptab import native_ecape


SAMPLE = (Path(__file__).resolve().parents[2] / "examples" / "soundings" /
          "hrrr_point_36.68N_95.66W_f018.npz")


def _sample_profile():
    collection, _ = load_npz(str(SAMPLE))
    return next(iter(collection._profs.values()))[0]


def _components(prof):
    direction = np.deg2rad(np.asarray(prof.wdir, dtype=float))
    speed = np.asarray(prof.wspd, dtype=float)
    return -speed * np.sin(direction), -speed * np.cos(direction)


def test_native_bridge_returns_none_for_missing_binary(monkeypatch):
    monkeypatch.setenv("SHARPMOD_ECAPE_BIN", str(SAMPLE.parent / "missing-ecape"))
    prof = _sample_profile()
    u, v = _components(prof)
    assert native_ecape.analytic_ecape(
        prof.pres, prof.hght, prof.tmpc, prof.dwpc, u, v) is None


@pytest.mark.skipif(not SAMPLE.exists(), reason="HRRR sample is unavailable")
def test_bundled_native_matches_ecape_parcel_reference():
    pytest.importorskip("ecape_parcel")
    if not native_ecape.available():
        pytest.skip("native ECAPE executable is unavailable on this platform")

    prof = _sample_profile()
    u, v = _components(prof)
    result = native_ecape.analytic_ecape(
        prof.pres, prof.hght, prof.tmpc, prof.dwpc, u, v)
    reference = ecape_mod._ecape_parcel_reference(
        np.asarray(prof.pres, dtype=float),
        np.asarray(prof.hght, dtype=float),
        np.asarray(prof.tmpc, dtype=float),
        np.asarray(prof.dwpc, dtype=float), u, v)

    assert result is not None
    assert reference is not None
    assert 0.0 <= result.ecape <= result.cape
    assert abs(result.ecape - reference) <= max(10.0, 0.05 * reference)


@pytest.mark.skipif(not SAMPLE.exists(), reason="HRRR sample is unavailable")
def test_public_ecape_prefers_native_and_clamps_to_mucape(monkeypatch):
    prof = _sample_profile()
    monkeypatch.setattr(ecape_mod, "_sharppy_mucape", lambda *args: 2000.0)
    monkeypatch.setattr(
        native_ecape, "analytic_ecape",
        lambda *args, **kwargs: native_ecape.NativeEcapeResult(
            ecape=2200.0, ncape=100.0, cape=3000.0),
    )
    monkeypatch.setattr(
        ecape_mod, "_ecape_parcel_reference",
        lambda *args: pytest.fail("Python fallback called despite native result"),
    )
    assert ecape_mod.ecape(prof) == 2000.0


@pytest.mark.skipif(not SAMPLE.exists(), reason="HRRR sample is unavailable")
def test_public_ecape_uses_python_reference_when_native_fails(monkeypatch):
    prof = _sample_profile()
    monkeypatch.setattr(ecape_mod, "_sharppy_mucape", lambda *args: 2000.0)
    monkeypatch.setattr(native_ecape, "analytic_ecape", lambda *args, **kwargs: None)
    monkeypatch.setattr(ecape_mod, "_ecape_parcel_reference", lambda *args: 1234.5)
    monkeypatch.setattr(
        ecape_mod, "_building_blocks",
        lambda *args: pytest.fail("local fallback called despite Python result"),
    )
    assert ecape_mod.ecape(prof) == 1234.5
