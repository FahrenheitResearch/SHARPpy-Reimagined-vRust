"""Focused contracts for NativeConvectiveProfile's light input initializer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

from sharppy.io import qc_tools
from sharppy.sharptab import profile as sp_profile
from sharppy.sharptab import utils as sp_utils

from sharpmod.sharptab import native_analysis, native_profile


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "examples" / "soundings" / \
    "hrrr_point_36.68N_95.66W_f018.npz"
CORE_ARRAYS = (
    "pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg", "u", "v",
)
BASIC_DERIVED_METHODS = (
    "get_wetbulb_profile",
    "get_thetae_profile",
    "get_theta_profile",
    "get_wvmr_profile",
    "get_rh_profile",
)


def _kwargs(*, strict_qc=False):
    with np.load(SAMPLE, allow_pickle=True) as data:
        values = {
            name: np.asarray(data[name], dtype=float)
            for name in (
                "pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg",
            )
        }
        values.update(
            latitude=float(data["lat"]),
            location=str(data["loc"]),
            date=datetime.strptime(str(data["valid"]), "%Y-%m-%d %H:%M"),
            missing=-9999.0,
            strictQC=strict_qc,
        )
    return values


def _copied(values):
    return {
        name: value.copy() if isinstance(value, (np.ndarray, ma.MaskedArray))
        else value
        for name, value in values.items()
    }


def _light_profile(values):
    result = object.__new__(native_profile.NativeConvectiveProfile)
    native_profile._initialize_native_inputs(result, _copied(values))
    return result


def _assert_array_exact(actual, expected):
    actual = ma.asarray(actual)
    expected = ma.asarray(expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        ma.getmaskarray(actual), ma.getmaskarray(expected))
    np.testing.assert_allclose(
        actual.filled(np.nan), expected.filled(np.nan), equal_nan=True,
        rtol=0.0, atol=0.0,
    )
    assert actual.fill_value == expected.fill_value


def _assert_input_contract_exact(light, basic):
    for name in CORE_ARRAYS:
        _assert_array_exact(getattr(light, name), getattr(basic, name))
    for name in ("dew_stdev", "tmp_stdev"):
        actual = getattr(light, name)
        expected = getattr(basic, name)
        if expected is None:
            assert actual is None
        else:
            _assert_array_exact(actual, expected)
    for name in (
            "missing", "profile", "latitude", "strictQC", "location",
            "date", "sfc", "top"):
        np.testing.assert_equal(getattr(light, name), getattr(basic, name))


def test_light_initializer_matches_basic_profile_on_real_npz():
    values = _kwargs(strict_qc=False)
    light = _light_profile(values)
    basic = sp_profile.BasicProfile(**_copied(values))

    _assert_input_contract_exact(light, basic)
    assert not {
        "logp", "vtmp", "wetbulb", "thetae", "theta", "wvmr", "relh",
    } & light.__dict__.keys()


def test_light_initializer_matches_basic_component_wind_conversion():
    values = _kwargs(strict_qc=False)
    values["u"], values["v"] = sp_utils.vec2comp(
        values.pop("wdir"), values.pop("wspd"))
    values["u"][6] = values["missing"]
    values["v"][6] = values["missing"]

    _assert_input_contract_exact(
        _light_profile(values), sp_profile.BasicProfile(**_copied(values)))


def test_light_initializer_matches_missing_rows_winds_omega_and_stdev():
    values = _kwargs(strict_qc=False)
    missing = values["missing"]
    # SHARPpy permits a reported pressure/height row with no temperature (the
    # bundled OAX sounding has this shape), but its vtmp masking cannot handle
    # a missing pressure row. Exercise the supported leading-placeholder form.
    values["pres"] = np.concatenate(
        ([values["pres"][0] + 10.0], values["pres"]))
    values["hght"] = np.concatenate(
        ([values["hght"][0] - 100.0], values["hght"]))
    for name in ("tmpc", "dwpc", "wdir", "wspd"):
        values[name] = np.concatenate(([missing], values[name]))
    values["wdir"][[4, 12]] = missing
    values["wspd"][[4, 12]] = missing
    values["tmpc"] = ma.array(values["tmpc"], mask=False)
    values["tmpc"].mask[9] = True
    values["dwpc"] = ma.array(values["dwpc"], mask=False)
    values["dwpc"].mask[[7, 16]] = True
    values["omeg"] = None
    values["tmp_stdev"] = np.full(len(values["pres"]), 0.5)
    values["dew_stdev"] = np.full(len(values["pres"]), 1.0)
    values["tmp_stdev"][5] = missing
    values["dew_stdev"][6] = missing

    light = _light_profile(values)
    basic = sp_profile.BasicProfile(**_copied(values))

    _assert_input_contract_exact(light, basic)
    assert light.sfc == basic.sfc == 1
    assert ma.getmaskarray(light.omeg).all()


def test_light_initializer_preserves_strict_qc_behavior():
    values = _kwargs(strict_qc=True)
    _assert_input_contract_exact(
        _light_profile(values), sp_profile.BasicProfile(**_copied(values)))

    values["wspd"][3] = -1.0
    with pytest.raises(Exception) as light_error:
        _light_profile(values)
    with pytest.raises(Exception) as basic_error:
        sp_profile.BasicProfile(**_copied(values))
    assert light_error.value.args == basic_error.value.args
    assert light_error.value.args[0] is qc_tools.DataQualityException


def test_light_initializer_skips_basic_thermodynamic_columns(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("BasicProfile derived thermodynamics were called")

    for name in BASIC_DERIVED_METHODS:
        monkeypatch.setattr(sp_profile.BasicProfile, name, forbidden)
    monkeypatch.setattr(native_profile.sp_thermo, "virtemp", forbidden)

    light = _light_profile(_kwargs(strict_qc=False))
    assert light.sfc == 0
    assert light.top == len(light.pres) - 1


@pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")
def test_native_success_does_not_call_basic_derived_profile_methods(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("BasicProfile derived method was called")

    for name in BASIC_DERIVED_METHODS:
        monkeypatch.setattr(sp_profile.BasicProfile, name, forbidden)

    prof = native_profile.NativeConvectiveProfile(**_kwargs(strict_qc=False))
    assert prof._sharpmod_calculation_backend == "sharppyrs/sharprs"
    for name in ("vtmp", "wetbulb", "thetae", "theta", "wvmr", "relh"):
        assert len(getattr(prof, name)) == len(prof.pres)


def test_native_failure_restarts_complete_convective_profile(monkeypatch):
    values = _kwargs(strict_qc=False)
    calls = {name: 0 for name in BASIC_DERIVED_METHODS}

    for name in BASIC_DERIVED_METHODS:
        original = getattr(sp_profile.BasicProfile, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(sp_profile.BasicProfile, name, counted)
    expected = sp_profile.ConvectiveProfile(**_copied(values))
    expected_calls = calls.copy()
    calls.update({name: 0 for name in BASIC_DERIVED_METHODS})
    monkeypatch.setattr(
        native_profile.native_analysis, "try_analyze_profile",
        lambda *_args, **_kwargs: None,
    )

    fallback = native_profile.NativeConvectiveProfile(**_copied(values))

    assert fallback._sharpmod_calculation_backend == "python-fallback"
    assert fallback._sharpmod_python_fallbacks == ("all",)
    assert calls == expected_calls
    for name in CORE_ARRAYS + (
            "logp", "vtmp", "wetbulb", "thetae", "theta", "wvmr", "relh"):
        _assert_array_exact(getattr(fallback, name), getattr(expected, name))
    for name in ("mupcl", "mlpcl", "sfcpcl"):
        assert getattr(fallback, name).bplus == pytest.approx(
            getattr(expected, name).bplus, rel=0.0, abs=0.0)
