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
OAX = ROOT / "examples" / "soundings" / "14061619.OAX"


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


def _copied(kwargs):
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in kwargs.items()
    }


def _oax_kwargs():
    from sharpmod.io import decoder

    collection = decoder.getDecoder("spc")(str(OAX)).getProfiles()
    raw = next(iter(collection._profs.values()))[0]
    result = {
        name: np.asarray(
            np.ma.asarray(getattr(raw, name)).filled(-9999.0), dtype=float)
        for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd")
    }
    omeg = getattr(raw, "omeg", None)
    result["omeg"] = (
        np.full_like(result["pres"], -9999.0)
        if omeg is None
        else np.asarray(np.ma.asarray(omeg).filled(-9999.0), dtype=float)
    )
    result.update(
        latitude=float(raw.latitude),
        location=str(raw.location),
        date=raw.date,
        missing=-9999.0,
        strictQC=False,
    )
    return result


def _assert_masked_array_equal(actual, expected):
    actual = np.ma.asarray(actual)
    expected = np.ma.asarray(expected)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_allclose(
        actual.filled(np.nan), expected.filled(np.nan), equal_nan=True)


@pytest.mark.parametrize("missing_omega", [False, True])
def test_fire_pbl_details_are_native_and_match_sharppy(
        monkeypatch, missing_omega):
    from sharppy.sharptab import interp as sp_interp
    from sharppy.sharptab import params as sp_params
    from sharppy.sharptab import profile as sp_profile
    from sharppy.sharptab import thermo as sp_thermo
    from sharppy.sharptab import winds as sp_winds
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _kwargs()
    if missing_omega:
        kwargs["omeg"][:] = kwargs["missing"]
    legacy = sp_profile.BasicProfile(**_copied(kwargs))
    ppbl_top = sp_params.pbl_top(legacy)
    pbl_h = sp_interp.to_agl(
        legacy, sp_interp.hght(legacy, ppbl_top))
    surface = legacy.pres[legacy.sfc]
    p1km = sp_interp.pres(legacy, sp_interp.to_msl(legacy, 1000.0))
    expected = {
        "ppbl_top": ppbl_top,
        "pbl_h": pbl_h,
        "sfc_rh": sp_thermo.relh(
            surface, legacy.tmpc[legacy.sfc], legacy.dwpc[legacy.sfc]),
        "rh01km": sp_params.mean_relh(legacy, pbot=surface, ptop=p1km),
        "pblrh": sp_params.mean_relh(
            legacy, pbot=surface, ptop=ppbl_top),
        "meanwind01km": sp_winds.mean_wind(
            legacy, pbot=surface, ptop=p1km),
        "meanwindpbl": sp_winds.mean_wind(
            legacy, pbot=surface, ptop=ppbl_top),
        "pblmaxwind": sp_winds.max_wind(legacy, lower=0, upper=pbl_h),
    }

    def forbidden_python_fire(_self):
        raise AssertionError("successful native path called Python fire/PBL")

    monkeypatch.setattr(
        NativeConvectiveProfile, "_run_python_fire_details",
        forbidden_python_fire)
    native = NativeConvectiveProfile(**_copied(kwargs))

    assert "fire-pbl-details" not in native._sharpmod_python_fallbacks
    for name, value in expected.items():
        np.testing.assert_allclose(
            np.ma.asarray(getattr(native, name)).filled(np.nan),
            np.ma.asarray(value).filled(np.nan),
            rtol=1e-10, atol=1e-8, equal_nan=True,
        )


@pytest.mark.parametrize("missing_omega", [False, True])
def test_precip_details_are_native_and_match_sharppy(
        monkeypatch, missing_omega):
    from sharppy.sharptab import params as sp_params
    from sharppy.sharptab import profile as sp_profile
    from sharppy.sharptab import watch_type as sp_watch_type
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _kwargs()
    if missing_omega:
        kwargs["omeg"][:] = kwargs["missing"]

    # Build the focused Python oracle before disabling every precipitation
    # calculation that the successful native path must now bypass.
    legacy = sp_profile.BasicProfile(**_copied(kwargs))
    sp_profile.ConvectiveProfile.get_precip(legacy)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("successful native path called Python precip")

    monkeypatch.setattr(
        NativeConvectiveProfile, "_run_python_precip_details", forbidden)
    monkeypatch.setattr(sp_params, "mean_omega", forbidden)
    for name in (
            "init_phase", "posneg_temperature", "posneg_wetbulb",
            "best_guess_precip"):
        monkeypatch.setattr(sp_watch_type, name, forbidden)

    native = NativeConvectiveProfile(**_copied(kwargs))

    assert native._sharpmod_native_precip is True
    assert "precipitation-source/layer-energies" not in \
        native._sharpmod_python_fallbacks
    numeric_names = (
        "dgz_meanomeg", "plevel", "tmp", "tpos", "tneg",
        "ttop", "tbot", "wpos", "wneg", "wtop", "wbot",
    )
    for name in numeric_names:
        np.testing.assert_allclose(
            np.ma.asarray(getattr(native, name)).filled(np.nan),
            np.ma.asarray(getattr(legacy, name)).filled(np.nan),
            rtol=1e-10, atol=1e-8, equal_nan=True,
            err_msg=name,
        )
    # Before this port, the optimized path's Python precip subset combined
    # Python mean omega with the already-native DGZ PW/RH fields.  Preserve
    # that established public OPRH value rather than silently switching it to
    # the slightly different all-Python DGZ integrations.
    expected_oprh = legacy.dgz_meanomeg * native.dgz_pw * (
        native.dgz_meanrh / 100.0)
    assert native.oprh == pytest.approx(expected_oprh, abs=1e-12)
    # The pre-existing native PW/RH integration seam is 0.51% on this case;
    # keep it tightly bounded against a wholly vendored calculation as well.
    assert native.oprh == pytest.approx(legacy.oprh, rel=0.006, abs=1e-12)
    assert native.phase == legacy.phase
    assert native.st == legacy.st
    assert native.precip_type == legacy.precip_type


@pytest.mark.skipif(not OAX.exists(), reason="bundled OAX profile is unavailable")
def test_oax_pbl_max_wind_uses_only_reported_wind_levels(monkeypatch):
    from sharppy.sharptab import interp as sp_interp
    from sharppy.sharptab import params as sp_params
    from sharppy.sharptab import profile as sp_profile
    from sharppy.sharptab import winds as sp_winds
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _oax_kwargs()
    raw = sp_profile.BasicProfile(**_copied(kwargs))
    start = raw.sfc
    normalized = _copied(kwargs)
    for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg"):
        normalized[name] = normalized[name][start:]
    legacy = sp_profile.BasicProfile(**normalized)
    ppbl_top = sp_params.pbl_top(legacy)
    pbl_h = sp_interp.to_agl(
        legacy, sp_interp.hght(legacy, ppbl_top))
    expected = sp_winds.max_wind(legacy, lower=0, upper=pbl_h)

    def forbidden_python_fire(_self):
        raise AssertionError("successful native path called Python fire/PBL")

    monkeypatch.setattr(
        NativeConvectiveProfile, "_run_python_fire_details",
        forbidden_python_fire)
    native = NativeConvectiveProfile(**_copied(kwargs))

    # The 877-hPa vector is filled only for interpolation-heavy native
    # kinematics. It must not displace the reported 904.95-hPa maximum.
    assert np.ma.is_masked(legacy.wspd[np.where(legacy.pres == 877.0)[0][0]])
    assert np.isfinite(np.asarray(
        native._sharpmod_native_analysis["arrays"]["wspd"], dtype=float)[
            np.where(np.asarray(native.pres) == 877.0)[0][0]])
    np.testing.assert_allclose(native.pblmaxwind, expected, atol=1e-10)


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


def test_internal_wind_interpolation_is_native_input_not_public_observation():
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _kwargs()
    gaps = np.array([5, 11, 18])
    kwargs["wdir"][gaps] = -9999.0
    kwargs["wspd"][gaps] = -9999.0

    legacy = sp_profile.BasicProfile(**_copied(kwargs))
    native = NativeConvectiveProfile(**_copied(kwargs))

    for name in (
            "pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg",
            "u", "v"):
        _assert_masked_array_equal(getattr(native, name), getattr(legacy, name))

    assert np.ma.getmaskarray(native.wdir)[gaps].all()
    assert np.ma.getmaskarray(native.wspd)[gaps].all()
    payload = native._sharpmod_native_analysis["arrays"]
    assert np.isfinite(np.asarray(payload["wdir"], dtype=float)[gaps]).all()
    assert np.isfinite(np.asarray(payload["wspd"], dtype=float)[gaps]).all()
    assert native._sharpmod_calculation_backend == "sharppyrs/sharprs"
    assert np.isfinite(float(native.right_srh1km[0]))


@pytest.mark.skipif(not OAX.exists(), reason="bundled OAX profile is unavailable")
def test_oax_leading_row_is_removed_without_publishing_filled_winds():
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    kwargs = _oax_kwargs()
    legacy = sp_profile.BasicProfile(**_copied(kwargs))
    native = NativeConvectiveProfile(**_copied(kwargs))

    assert legacy.sfc == 1
    assert native.sfc == 0
    assert len(native.pres) == len(legacy.pres) - legacy.sfc
    assert native.top == len(native.pres) - 1
    for name in (
            "pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg",
            "u", "v"):
        expected = np.ma.asarray(getattr(legacy, name))[legacy.sfc:]
        _assert_masked_array_equal(getattr(native, name), expected)

    public_missing = (
        np.ma.getmaskarray(native.wdir) | np.ma.getmaskarray(native.wspd))
    payload = native._sharpmod_native_analysis["arrays"]
    normalized_wdir = np.asarray(payload["wdir"], dtype=float)
    normalized_wspd = np.asarray(payload["wspd"], dtype=float)
    filled_for_rust = (
        public_missing & np.isfinite(normalized_wdir)
        & np.isfinite(normalized_wspd)
    )
    assert public_missing.any()
    assert filled_for_rust.any()
    assert len(native.vtmp) == len(native.pres)
    assert np.isfinite(float(native.right_srh1km[0]))


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
        "SARS-analog-databases",
        "PWV-station-climatology",
    )


def test_python_sars_subset_matches_full_vendored_method():
    """The remaining optimized Python SARS lookup retains legacy outputs."""
    from sharppy.sharptab import profile as sp_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    prof = NativeConvectiveProfile(**_kwargs())

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


def test_storm_motion_edit_refreshes_native_watch_labels(monkeypatch):
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    prof = NativeConvectiveProfile(**_kwargs())
    calls = []

    def classified(values):
        calls.append(dict(values))
        return "TOR" if len(calls) == 1 else "MRGL SVR"

    monkeypatch.setattr(native_analysis, "classify_watch", classified)
    current = tuple(prof.srwind)
    prof.set_srright(current[0] + 1.0, current[1] - 1.0)

    assert len(calls) == 2
    assert prof.right_watch_type == "TOR"
    assert prof.left_watch_type == "MRGL SVR"
    assert prof.watch_type == "TOR"


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
