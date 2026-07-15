"""Contracts for the bulk sharppyrs/sharprs analysis adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pytest

from sharpmod.sharptab import native_analysis


pytestmark = pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "examples" / "soundings" / "hrrr_point_36.68N_95.66W_f018.npz"


def _kwargs():
    with np.load(SAMPLE, allow_pickle=True) as data:
        return {
            name: np.asarray(data[name], dtype=float)
            for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg")
        } | {
            "latitude": float(data["lat"]),
            "location": str(data["loc"]),
            "date": datetime.strptime(str(data["valid"]), "%Y-%m-%d %H:%M"),
            "missing": -9999.0,
            "strictQC": False,
        }


def test_bulk_native_analysis_has_complete_reimagined_contract():
    from sharppy.sharptab import profile as sp_profile

    raw = sp_profile.create_profile(profile="raw", **_kwargs())
    result = native_analysis.analyze_profile(raw)

    assert result["schema"] == "sharpmod.native-analysis.v1"
    assert result["provenance"] == {
        "profile": "sharprs-core",
        "parcels": "sharprs-core",
        "derived": "sharppyrs-rust",
        "ecape": "ecape-rs",
    }
    assert len(result["derived"]) == 84
    assert set(result["parcels"]) == {
        "surface", "forecast", "most_unstable", "mixed_layer", "effective"}
    assert result["derived"]["ecape"] == pytest.approx(
        result["ecape"]["ecape"])
    assert result["derived"]["ecape"] <= result["ecape"]["cape"]


def test_native_adapter_does_not_call_covered_python_analysis_methods(monkeypatch):
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    def forbidden(*_args, **_kwargs):
        raise AssertionError("covered Python calculation was called")

    for name in (
            "get_parcels", "get_thermo", "get_kinematics", "get_severe",
            "get_traj", "get_indices", "get_watch", "get_fire",
            "get_precip", "get_sars"):
        monkeypatch.setattr(sp_profile.ConvectiveProfile, name, forbidden)

    prof = NativeConvectiveProfile(**_kwargs())
    assert prof._sharpmod_calculation_backend == "sharppyrs/sharprs"
    assert prof.mupcl.bplus > 0
    assert prof.ecape > 0
    assert prof.watch_type != ""
    assert prof._sharpmod_python_fallbacks == (
        "fire-pbl-details",
        "precipitation-source/layer-energies",
        "SARS-analog-databases",
        "PWV-station-climatology",
    )


def test_python_only_feature_subset_matches_full_vendored_methods():
    """Removing overwritten work must not change retained Python outputs."""
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    prof = NativeConvectiveProfile(**_kwargs())

    fire_names = (
        "ppbl_top", "sfc_rh", "pbl_h", "rh01km", "pblrh",
        "meanwind01km", "meanwindpbl", "pblmaxwind",
    )
    optimized_fire = {name: getattr(prof, name) for name in fire_names}
    sp_profile.ConvectiveProfile.get_fire(prof)
    prof._apply_native_fire(prof._sharpmod_native_analysis)
    for name, expected in optimized_fire.items():
        np.testing.assert_equal(getattr(prof, name), expected)

    precip_names = (
        "dgz_meanomeg", "oprh", "plevel", "phase", "tmp", "st",
        "tpos", "tneg", "ttop", "tbot", "wpos", "wneg", "wtop",
        "wbot", "precip_type",
    )
    # Restore the optimized path before taking its precip snapshot; get_fire
    # does not alter these values, but this keeps the comparison explicit.
    prof._run_python_precip_details()
    prof._apply_native_precip_type()
    optimized_precip = {
        name: getattr(prof, name) for name in precip_names}
    sp_profile.ConvectiveProfile.get_precip(prof)
    prof._apply_native_winter(prof._sharpmod_native_analysis)
    prof._apply_native_precip_type()
    for name, expected in optimized_precip.items():
        np.testing.assert_equal(getattr(prof, name), expected)

    match_names = (
        "right_matches", "left_matches", "right_supercell_matches",
        "left_supercell_matches", "matches", "supercell_matches",
    )
    prof._run_python_sars_matches()
    optimized_matches = {name: getattr(prof, name) for name in match_names}
    sp_profile.ConvectiveProfile.get_sars(prof)
    for name, expected in optimized_matches.items():
        np.testing.assert_equal(getattr(prof, name), expected)


def test_sars_tables_are_parsed_once_across_profiles(monkeypatch):
    from sharpmod.sharptab import sars_cache
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    calls = []
    original = sars_cache._ORIGINAL_LOADTXT

    def counted(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)

    sars_cache.clear_cache()
    monkeypatch.setattr(sars_cache, "_ORIGINAL_LOADTXT", counted)
    try:
        NativeConvectiveProfile(**_kwargs())
        NativeConvectiveProfile(**_kwargs())
        assert len(calls) == 2
        assert sum(path.endswith("sars_hail.txt") for path in calls) == 1
        assert sum(path.endswith("sars_supercell.txt") for path in calls) == 1
    finally:
        sars_cache.clear_cache()


def test_native_adapter_matches_python_oracle_on_supported_fields():
    from sharpmod.viz import _qt6_compat
    _qt6_compat.apply()
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _kwargs()
    legacy = sp_profile.ConvectiveProfile(**kwargs)
    native = NativeConvectiveProfile(**kwargs)

    for parcel_name in ("sfcpcl", "mupcl", "mlpcl", "fcstpcl"):
        legacy_parcel = getattr(legacy, parcel_name)
        native_parcel = getattr(native, parcel_name)
        assert native_parcel.bplus == pytest.approx(
            legacy_parcel.bplus, rel=0.05, abs=1.0)
        assert native_parcel.bminus == pytest.approx(
            legacy_parcel.bminus, rel=0.05, abs=1.0)
        assert native_parcel.lclhght == pytest.approx(
            legacy_parcel.lclhght, rel=0.05, abs=5.0)

    for name in (
            "pwat", "k_idx", "lapserate_3km", "lapserate_3_6km",
            "right_scp", "left_scp", "right_stp_cin", "right_stp_fixed",
            "ship", "dcape", "tei", "sig_severe"):
        assert float(getattr(native, name)) == pytest.approx(
            float(getattr(legacy, name)), rel=0.05, abs=0.05), name


def test_profile_collection_and_companion_use_native_results(monkeypatch):
    from sharpmod.io.decoder import load_npz
    from sharpmod.sharptab import profile as derived_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    collection, _ = load_npz(str(SAMPLE))
    assert collection._target_type is NativeConvectiveProfile
    prof = collection.getHighlightedProf()

    monkeypatch.setitem(
        derived_profile._SINGLE_COMPUTE,
        "ecape",
        lambda _prof: (_ for _ in ()).throw(
            AssertionError("Python ECAPE fallback should not run")),
    )
    companion = derived_profile.derived_profile_from(
        prof, warm=derived_profile.DISPLAY_DERIVED_ATTRS)
    assert companion.ecape == pytest.approx(prof.ecape)
    assert companion._sharpmod_calculation_backend == "sharppyrs/sharprs"
    assert "ecape" in companion._sharpmod_native_fields


def test_missing_omega_moshe_does_not_build_legacy_profile(monkeypatch):
    """Cached .rws points omit OMEGA, so MOSHE is undefined, not a fallback."""
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab import profile as derived_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _kwargs()
    kwargs["omeg"] = np.full_like(kwargs["pres"], -9999.0)
    prof = NativeConvectiveProfile(**kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing OMEGA triggered a legacy full-profile build")

    monkeypatch.setattr(sp_profile.ConvectiveProfile, "__init__", forbidden)
    companion = derived_profile.derived_profile_from(
        prof, warm=derived_profile.DISPLAY_DERIVED_ATTRS)

    assert np.ma.is_masked(companion.modified_sherbe)


def test_native_bulk_analysis_latency_contract():
    from sharppy.sharptab import profile as sp_profile

    raw = sp_profile.create_profile(profile="raw", **_kwargs())
    native_analysis.analyze_profile(raw)  # cold import/allocation warm-up
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        native_analysis.analyze_profile(raw)
        samples.append((time.perf_counter() - started) * 1000.0)
    assert min(samples) < 100.0, samples


def test_interactive_user_parcel_uses_sharprs(monkeypatch):
    from sharppy.sharptab import params as sp_params
    from sharpmod.sharptab import native_profile

    prof = native_profile.NativeConvectiveProfile(**_kwargs())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("interactive parcel fell back to Python")

    monkeypatch.setattr(native_profile, "_ORIGINAL_PARCELX", forbidden)
    parcel = sp_params.parcelx(
        prof, flag=5, pres=900.0, tmpc=24.0, dwpc=18.0)

    assert parcel._sharpmod_calculation_backend == "sharprs"
    assert parcel.bplus > 0.0
    assert len(parcel.ptrace) == len(parcel.ttrace)


def test_legacy_profile_parcel_still_delegates_to_python(monkeypatch):
    from sharppy.sharptab import params as sp_params
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab import native_profile

    raw = sp_profile.create_profile(profile="raw", **_kwargs())
    sentinel = object()

    def delegated(prof, **kwargs):
        assert prof is raw
        assert kwargs["flag"] == 1
        return sentinel

    monkeypatch.setattr(native_profile, "_ORIGINAL_PARCELX", delegated)
    assert sp_params.parcelx(raw, flag=1) is sentinel


def test_native_disable_switch_selects_complete_python_fallback(monkeypatch):
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab import native_profile

    monkeypatch.setenv("SHARPMOD_DISABLE_NATIVE_ANALYSIS", "1")
    assert native_profile.target_profile_type() is sp_profile.ConvectiveProfile

    prof = native_profile.NativeConvectiveProfile(**_kwargs())
    assert prof._sharpmod_calculation_backend == "python-fallback"
    assert prof._sharpmod_native_analysis is None
    assert prof._sharpmod_python_fallbacks == ("all",)
    assert prof.mupcl.bplus > 0.0


def test_interactive_native_failure_delegates_to_python(monkeypatch):
    from sharppy.sharptab import params as sp_params
    from sharpmod.sharptab import native_analysis, native_profile

    prof = native_profile.NativeConvectiveProfile(**_kwargs())
    sentinel = object()

    def unavailable(*_args, **_kwargs):
        raise native_analysis.NativeAnalysisUnavailable("test fallback")

    def delegated(candidate, **kwargs):
        assert candidate is prof
        assert kwargs["flag"] == 5
        return sentinel

    monkeypatch.setattr(native_analysis, "lift_user_parcel", unavailable)
    monkeypatch.setattr(native_profile, "_ORIGINAL_PARCELX", delegated)
    assert sp_params.parcelx(
        prof, flag=5, pres=900.0, tmpc=24.0, dwpc=18.0
    ) is sentinel
