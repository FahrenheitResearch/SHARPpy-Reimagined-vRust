"""Effective-inflow edge contracts at the Rust/Python object boundary."""

from __future__ import annotations

import numpy as np
import numpy.ma as ma
import pytest

from sharpmod.sharptab import native_analysis
from sharpmod.tools.native_parity import _copy_kwargs, build_corpus


pytestmark = pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")


def _case(name):
    return next(case for case in build_corpus() if case.name == name)


def test_single_level_effective_layer_does_not_fall_back_to_surface_parcel():
    from sharppy.sharptab import profile as legacy_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    case = _case("bufkit-kbvo-2026062608")
    legacy = legacy_profile.ConvectiveProfile(**_copy_kwargs(case.kwargs))
    native = NativeConvectiveProfile(**_copy_kwargs(case.kwargs))

    assert native.ebottom == pytest.approx(951.9)
    assert native.etop == pytest.approx(951.9)
    assert native.effpcl.pres == pytest.approx(legacy.effpcl.pres, abs=1.0e-9)
    assert native.effpcl.tmpc == pytest.approx(legacy.effpcl.tmpc, abs=1.0e-9)
    assert native.effpcl.dwpc == pytest.approx(legacy.effpcl.dwpc, abs=1.0e-9)
    assert native.effpcl.bplus == pytest.approx(legacy.effpcl.bplus, abs=1.0e-6)
    assert native.effpcl.bminus == pytest.approx(legacy.effpcl.bminus, abs=1.0e-6)
    assert native.effpcl.lclpres == pytest.approx(
        legacy.effpcl.lclpres, abs=1.0e-6)
    assert native.effpcl.elpres == pytest.approx(legacy.effpcl.elpres, abs=1.0e-9)
    np.testing.assert_allclose(native.mean_eff, legacy.mean_eff, atol=1.0e-9)
    np.testing.assert_allclose(
        native.right_srw_eff, legacy.right_srw_eff, atol=1.0e-9)
    np.testing.assert_allclose(
        native.left_srw_eff, legacy.left_srw_eff, atol=1.0e-9)


@pytest.mark.parametrize(
    "case_name",
    ("bufkit-kbvo-2026062607", "synthetic-stable-dry"),
)
def test_no_effective_layer_preserves_legacy_public_missing_semantics(case_name):
    from sharppy.sharptab import profile as legacy_profile
    from sharpmod.sharptab import derived
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    case = _case(case_name)
    legacy = legacy_profile.ConvectiveProfile(**_copy_kwargs(case.kwargs))
    native = NativeConvectiveProfile(**_copy_kwargs(case.kwargs))

    assert ma.is_masked(native.ebottom)
    assert ma.is_masked(native.etop)
    assert native.right_stp_cin == 0.0
    assert native.left_stp_cin == 0.0
    assert ma.is_masked(native.right_critical_angle)
    assert ma.is_masked(native.left_critical_angle)

    for name in (
            "ebwd", "mean_eff", "mean_ebw", "right_srw_eff",
            "right_srw_ebw", "left_srw_eff", "left_srw_ebw"):
        value = np.asarray(getattr(native, name), dtype=float)
        expected = np.asarray(getattr(legacy, name), dtype=float)
        assert value.shape == (3,), name
        assert expected.shape == (3,), name
        np.testing.assert_array_equal(value, expected)

    expected_moshe = derived.modified_sherbe(legacy)
    assert native.modified_sherbe == pytest.approx(expected_moshe, abs=1.0e-3)
