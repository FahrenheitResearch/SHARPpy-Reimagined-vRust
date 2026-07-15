"""Focused contracts for the fixed-seed native parity stress matrix."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from sharpmod.sharptab import native_analysis
from sharpmod.tools import native_parity_fuzz


EXPECTED_MATRIX_SHA256 = (
    "E6BF88FF073EFA5DF6327203CE087B2A6E5B652F8D3E6AAFDC7A11DA9606BE68"
)

# Each ID caught a concrete edge seam in the 100-case release audit:
# downdraft source/trace selection, a later EL crossing retaining SHARPpy's
# stale MPL height, or bounded HGZ CAPE with missing moisture above the layer.
PARCEL_EDGE_CASE_IDS = (4, 7, 10, 15, 20, 40, 48, 52, 91, 92, 96, 97)
SINGLE_LEVEL_EFFECTIVE_WIND_CASE_ID = 61


def test_fixed_seed_fuzz_matrix_is_complete_and_reproducible():
    cases = native_parity_fuzz.build_fuzz_cases()

    assert len(cases) == 100
    assert Counter(case.metadata["family"] for case in cases) == {
        "hrrr_dense": 50,
        "bufkit_upper_dry": 25,
        "oax_sparse": 15,
        "hrrr_edge": 10,
    }
    assert Counter(
        case.metadata["scenario"]
        for case in cases if case.metadata["scenario"]
    ) == {
        "stable": 2,
        "near_saturated": 2,
        "hot_moist": 2,
        "dry_mixed": 2,
        "strong_shear": 2,
    }
    assert native_parity_fuzz.feature_summary(cases) == {
        "families": {
            "bufkit_upper_dry": 25,
            "hrrr_dense": 50,
            "hrrr_edge": 10,
            "oax_sparse": 15,
        },
        "northern_hemisphere": 50,
        "southern_hemisphere": 50,
        "missing_upper_moisture": 35,
        "internal_missing_wind": 14,
        "missing_omega": 13,
        "negative_elevation_perturbation": 16,
        "positive_elevation_perturbation": 84,
    }
    assert native_parity_fuzz.matrix_fingerprint(
        cases) == EXPECTED_MATRIX_SHA256

    # Every perturbation owns its arrays; auditing or mutating one case cannot
    # silently alter a later case in the deterministic matrix.
    for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd"):
        assert not np.shares_memory(cases[0].kwargs[name], cases[1].kwargs[name])


def test_fuzz_smoke_selection_spans_source_families_and_hemispheres():
    cases = native_parity_fuzz.build_fuzz_cases()
    selected = [
        case for case in cases
        if case.metadata["case_id"] in native_parity_fuzz.SMOKE_CASE_IDS
    ]

    assert len(selected) == len(native_parity_fuzz.SMOKE_CASE_IDS)
    assert {case.metadata["family"] for case in selected} == {
        "hrrr_dense", "bufkit_upper_dry", "oax_sparse", "hrrr_edge",
    }
    assert {case.metadata["latitude"] > 0.0 for case in selected} == {
        False, True,
    }
    assert any(case.metadata["missing_upper_moisture"] for case in selected)
    assert any(case.metadata["internal_missing_wind"] for case in selected)
    assert any(case.metadata["omega_missing"] for case in selected)


@pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")
def test_native_fixed_seed_smoke_matches_canonical_python_oracles():
    cases = native_parity_fuzz.build_fuzz_cases()
    selected_ids = set(native_parity_fuzz.SMOKE_CASE_IDS)
    selected = [
        case for case in cases
        if case.metadata["case_id"] in selected_ids
    ]
    comparisons, summary = native_parity_fuzz.audit_fuzz_cases(selected)
    failures = [comparison for comparison in comparisons if not comparison.passed]

    details = "\n".join(
        f"{item.case}: {item.group}.{item.field}: {item.status}; "
        f"legacy={item.legacy!r}; native={item.native!r}; "
        f"detail={item.detail!r}"
        for item in failures[:30]
    )
    assert comparisons, "canonical auditor returned no comparisons"
    assert summary["rust_owned_derived_fields"] == 84
    assert summary["tolerance_gated_derived_fields"] == 82
    assert not failures, details


@pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")
def test_native_downdraft_mpl_and_hgz_regression_cases_match_python():
    cases = native_parity_fuzz.build_fuzz_cases()
    selected_ids = set(PARCEL_EDGE_CASE_IDS)
    selected = [
        case for case in cases
        if case.metadata["case_id"] in selected_ids
    ]
    comparisons, _summary = native_parity_fuzz.audit_fuzz_cases(selected)
    failures = [comparison for comparison in comparisons if not comparison.passed]

    details = "\n".join(
        f"{item.case}: {item.group}.{item.field}: {item.status}; "
        f"legacy={item.legacy!r}; native={item.native!r}; "
        f"detail={item.detail!r}"
        for item in failures[:60]
    )
    assert comparisons, "canonical auditor returned no comparisons"
    assert not failures, details


@pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")
def test_single_level_effective_wind_matches_python():
    case = native_parity_fuzz.build_fuzz_cases()[
        SINGLE_LEVEL_EFFECTIVE_WIND_CASE_ID]
    assert case.name == "fuzz-061-bufkit_upper_dry"

    comparisons, _summary = native_parity_fuzz.audit_fuzz_cases([case])
    failures = [comparison for comparison in comparisons if not comparison.passed]

    details = "\n".join(
        f"{item.case}: {item.group}.{item.field}: {item.status}; "
        f"legacy={item.legacy!r}; native={item.native!r}; "
        f"detail={item.detail!r}"
        for item in failures[:30]
    )
    assert comparisons, "canonical auditor returned no comparisons"
    assert not failures, details
