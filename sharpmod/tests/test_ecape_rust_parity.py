"""Direct Rust ECAPE agreement tests against ``ecape-parcel-py``.

The ordinary public-API tests cannot prove that Rust produced the answer: a
native failure intentionally falls through to the Python oracle.  This module
therefore exercises the in-process ``sharpmod_native`` result directly, and
also checks the standalone helper on platforms where it is packaged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

from sharpmod.io import decoder as decoder_mod
from sharpmod.io.decoder import load_npz
from sharpmod.sharptab import ecape as ecape_mod
from sharpmod.sharptab import native_analysis, native_ecape
from sharpmod.tests.test_reference_agreement_property import SOUNDINGS


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "soundings"
REFERENCE_REL_TOL = 0.05
REFERENCE_ABS_TOL_JKG = 10.0
MIN_COMPARABLE_CASES = 50


def _collection_profiles(collection):
    for member, profiles in collection._profs.items():
        for index, profile in enumerate(profiles):
            yield f"{member}:{index}", profile


def _parity_cases():
    for index, profile in enumerate(SOUNDINGS, start=1):
        yield f"synthetic-{index:02d}", profile

    for decoder_name, filename in (
        ("spc", "14061619.OAX"),
        ("spc", "hrrr_point_36.68N_95.66W_f018.spc"),
        ("bufkit", "hrrr_kbvo_20260625_06z.buf"),
    ):
        path = EXAMPLES / filename
        decoder = decoder_mod.getDecoder(decoder_name)
        collection = decoder(str(path)).getProfiles()
        for suffix, profile in _collection_profiles(collection):
            yield f"{filename}:{suffix}", profile

    path = EXAMPLES / "hrrr_point_36.68N_95.66W_f018.npz"
    collection, _ = load_npz(str(path))
    _suffix, profile = next(_collection_profiles(collection))
    yield "hrrr-point-npz", profile


def _clean_arrays(profile):
    fields = [
        np.asarray(ma.asarray(getattr(profile, name), dtype=float).filled(np.nan))
        for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd")
    ]
    valid = np.logical_and.reduce([
        np.isfinite(values) & (values != -9999.0) for values in fields
    ])
    pres, hght, tmpc, dwpc, wdir, wspd = [
        values[valid] for values in fields
    ]
    radians = np.deg2rad(wdir)
    u_knots = -wspd * np.sin(radians)
    v_knots = -wspd * np.cos(radians)
    return pres, hght, tmpc, dwpc, u_knots, v_knots


def _native_ecape_arrays(result):
    """Return the exact normalized columns consumed by in-process ecape-rs."""
    arrays = result["arrays"]
    fields = [
        np.asarray(arrays[name], dtype=float)
        for name in ("pres", "hght", "tmpc", "dwpc", "u", "v")
    ]
    valid = np.logical_and.reduce([np.isfinite(values) for values in fields])
    pres, hght, tmpc, dwpc, u_knots, v_knots = [
        values[valid] for values in fields
    ]
    return pres, hght, tmpc, dwpc, u_knots, v_knots


def _assert_reference_agreement(case, actual, reference):
    tolerance = max(
        REFERENCE_ABS_TOL_JKG,
        REFERENCE_REL_TOL * abs(reference),
    )
    difference = abs(actual - reference)
    assert difference <= tolerance, (
        f"{case}: Rust ECAPE {actual:.6f} J/kg differs from "
        f"ecape-parcel-py {reference:.6f} J/kg by {difference:.6f} J/kg "
        f"(allowed {tolerance:.6f} J/kg)"
    )


@pytest.mark.skipif(
    not native_analysis.available(),
    reason="bulk native analysis extension has not been built",
)
def test_rust_ecape_matches_reference_across_synthetic_and_real_profiles():
    """Rust agrees with the maintained oracle without using Python fallback.

    The matrix includes 14 controlled synthetic environments, an observed OAX
    sounding, HRRR SPC and NPZ point soundings, and every forecast profile in a
    49-hour HRRR BUFKIT file.  Profiles for which the reference package itself
    cannot form ECAPE are recorded but not compared.
    """
    pytest.importorskip("ecape_parcel")

    compared = 0
    reference_unavailable = 0
    weak_instability_values = {}
    for case, profile in _parity_cases():
        arrays = _clean_arrays(profile)
        native = native_analysis.analyze_profile(profile)
        result = native["ecape"]
        display_ecape = float(native["derived"]["ecape"])
        native_mucape = max(
            0.0,
            float(native["parcels"]["most_unstable"]["bplus"]),
        )
        assert 0.0 <= display_ecape <= native_mucape + 1.0e-6, case
        if result is not None:
            actual = float(result["ecape"])
            assert np.isfinite(actual) and actual >= 0.0, case

        reference = ecape_mod._ecape_parcel_reference(*arrays)
        if reference is None:
            reference_unavailable += 1
            continue

        assert native["provenance"]["ecape"] == "ecape-rs", case
        assert result is not None, case
        _assert_reference_agreement(case, actual, reference)

        if native_ecape.available():
            # Exact engine equality requires exact input equality. The bulk
            # adapter removes a leading non-physical OAX row and interpolates
            # bracketed wind gaps for calculations; feed those same normalized
            # columns to the separately packaged helper.
            helper = native_ecape.analytic_ecape(*_native_ecape_arrays(native))
            assert helper is not None, case
            _assert_reference_agreement(case, helper.ecape, reference)
            assert helper.ecape == pytest.approx(actual, rel=1.0e-10, abs=1.0e-8)

        if case in {"synthetic-04", "synthetic-10", "hrrr_kbvo_20260625_06z.buf::32"}:
            weak_instability_values[case] = actual
        compared += 1

    assert compared >= MIN_COMPARABLE_CASES, (
        f"only {compared} profiles had comparable reference ECAPE values; "
        f"reference unavailable for {reference_unavailable}"
    )
    # These cases caught two former false-zero results and one rejected result
    # where analytic ECAPE legitimately exceeded the solver's internal CAPE.
    assert weak_instability_values.keys() == {
        "synthetic-04",
        "synthetic-10",
        "hrrr_kbvo_20260625_06z.buf::32",
    }
    assert all(value > 100.0 for value in weak_instability_values.values())


def test_standalone_bridge_accepts_weak_case_above_internal_cape():
    """Raw analytic ECAPE above internal CAPE is valid, not bridge corruption."""
    if not native_ecape.available():
        pytest.skip("standalone native ECAPE helper is unavailable")

    profile = SOUNDINGS[3]
    result = native_ecape.analytic_ecape(*_clean_arrays(profile))

    assert result is not None
    assert result.ecape > result.cape > 0.0
