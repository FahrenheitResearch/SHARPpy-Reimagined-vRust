"""Release-facing native parity schema and representative-corpus gates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sharpmod.sharptab import native_analysis
from sharpmod.tools import native_parity


pytestmark = pytest.mark.skipif(
    not native_analysis.available(), reason="native extension has not been built")


def test_native_parity_corpus_inventory_is_52_real_plus_5_synthetic():
    cases = native_parity.build_corpus()
    real = [case for case in cases if case.source != "deterministic synthetic transform"]
    synthetic = [
        case for case in cases if case.source == "deterministic synthetic transform"
    ]

    assert len(cases) == 57
    assert len(real) == 52
    assert len(synthetic) == 5
    assert len({case.name for case in cases}) == len(cases)
    assert sum(case.name.startswith("spc-oax-") for case in real) == 1
    assert sum(case.name.startswith("spc-hrrr-") for case in real) == 1
    assert sum(case.name.startswith("bufkit-kbvo-") for case in real) == 49
    assert sum(case.name == "npz-hrrr-point" for case in real) == 1
    assert {case.name for case in synthetic} == {
        "synthetic-stable-dry",
        "synthetic-moist-unstable",
        "synthetic-elevated",
        "synthetic-zero-wind",
        "synthetic-southern-hemisphere",
    }
    assert native_parity.QUICK_CASE_NAMES <= {case.name for case in real}
    assert len(native_parity.QUICK_CASE_NAMES) == 8
    assert {
        "spc-hrrr-2026062600",
        "npz-hrrr-point",
        "bufkit-kbvo-2026062623",
    } <= native_parity.QUICK_CASE_NAMES


def test_native_derived_schema_has_an_explicit_84_field_oracle_inventory():
    case = next(
        candidate for candidate in native_parity.build_corpus(synthetic=False)
        if candidate.name == "npz-hrrr-point"
    )
    from sharppy.sharptab import profile as legacy_profile

    raw = legacy_profile.create_profile(
        profile="raw", **native_parity._copy_kwargs(case.kwargs))
    result = native_analysis.analyze_profile(raw)

    assert len(native_parity.DERIVED_FIELD_COVERAGE) == 84
    assert set(result["derived"]) == set(native_parity.DERIVED_FIELD_COVERAGE)
    assert native_parity.EXTERNAL_ORACLE_FIELDS == {
        "ecape": "sharpmod/tests/test_ecape_rust_parity.py uses ecape-parcel-py",
    }
    assert native_parity.INFORMATIONAL_DERIVED_FIELDS == {
        "mmp": "legacy SHARPpy reads uninitialized np.empty cells",
    }
    assert (
        len(native_parity.DERIVED_FIELD_COVERAGE)
        - len(native_parity.EXTERNAL_ORACLE_FIELDS)
        - len(native_parity.INFORMATIONAL_DERIVED_FIELDS)
    ) == 82


def test_native_parity_helpers_enforce_missing_states_and_normalize_geometry():
    case = native_parity.Case("unit", "unit", {})
    missing = native_parity._compare_numeric(
        case,
        "unit",
        "missing",
        [None, 0.0, 1.0],
        [0.0, None, 1.0],
        native_parity.Tolerance(0.01),
    )
    assert [item.status for item in missing] == [
        "missing-mismatch",
        "missing-mismatch",
        "within-tolerance",
    ]

    complete = SimpleNamespace(
        pres=np.array([1000.0, 900.0, 800.0]),
        tmpc=np.array([20.0, 10.0, 0.0]),
    )
    missing_level = SimpleNamespace(
        pres=np.array([1000.0, 900.0, 800.0]),
        tmpc=np.ma.array([20.0, 10.0, 0.0], mask=[False, True, False]),
    )
    coverage = native_parity._compare_environment_on_pressure(
        case,
        complete,
        missing_level,
        "tmpc",
        native_parity.Tolerance(0.01),
    )
    assert coverage[0].status == "coverage-mismatch"
    assert not coverage[0].passed

    oax_case = native_parity.Case("spc-oax-unit", "unit", {})
    oax_coverage = native_parity._compare_environment_on_pressure(
        oax_case,
        SimpleNamespace(
            pres=np.array([1000.0, 900.0, 800.0]),
            tmpc=np.ma.array([0.0, 10.0, 0.0], mask=[True, False, False]),
            sfc=1,
        ),
        SimpleNamespace(
            pres=np.array([900.0, 800.0]),
            tmpc=np.array([10.0, 0.0]),
            sfc=0,
        ),
        "pres",
        native_parity.Tolerance(0.01),
    )
    assert oax_coverage[0].status == "normalized-sampling"
    assert oax_coverage[0].passed
    oax_missing_value = native_parity._compare_environment_on_pressure(
        oax_case,
        complete,
        missing_level,
        "tmpc",
        native_parity.Tolerance(0.01),
    )
    assert oax_missing_value[0].status == "coverage-mismatch"
    assert not oax_missing_value[0].passed

    old_pressure = np.array([100.0, 200.0, 500.0, 1000.0])
    new_pressure = np.array([100.0, 300.0, 700.0, 1000.0])
    trace = native_parity._compare_pressure_temperature_trace(
        case,
        "unit",
        "trace",
        old_pressure,
        10.0 * np.log(old_pressure),
        new_pressure,
        10.0 * np.log(new_pressure),
    )
    assert all(item.passed for item in trace)

    old_path = SimpleNamespace(
        slinky_traj=np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    )
    new_path = SimpleNamespace(
        slinky_traj=np.array(
            [[0.0, 0.0], [10.0, 0.0], [70.0, 0.0], [100.0, 0.0]]
        )
    )
    path = native_parity._compare_slinky_path(case, old_path, new_path)
    assert all(item.passed for item in path)


def test_watch_oracle_normalizes_only_undefined_legacy_mmp(monkeypatch):
    from sharppy.sharptab import watch_type

    legacy = SimpleNamespace(
        mmp=0.999999999999,
        right_watch_type="SVR",
        left_watch_type="SVR",
    )
    native = SimpleNamespace(mmp=0.50)

    def possible_watch(profile, use_left=False):
        assert profile is legacy
        return np.asarray([
            "SVR" if profile.mmp >= 0.6 else "MRGL SVR",
            "NONE",
        ])

    monkeypatch.setattr(watch_type, "possible_watch", possible_watch)
    normalized, changed = native_parity._deterministic_watch_oracle(
        legacy, native)

    assert normalized == {
        "right_watch_type": "MRGL SVR",
        "left_watch_type": "MRGL SVR",
    }
    assert changed == {
        "right_watch_type": ("SVR", "MRGL SVR"),
        "left_watch_type": ("SVR", "MRGL SVR"),
    }
    assert legacy.mmp == pytest.approx(0.999999999999)


def test_native_quick_representative_corpus_matches_python_oracles():
    cases = [
        case for case in native_parity.build_corpus(synthetic=False)
        if case.name in native_parity.QUICK_CASE_NAMES
    ]
    comparisons, summary = native_parity.audit(cases)
    failures = [comparison for comparison in comparisons if not comparison.passed]

    details = "\n".join(
        f"{item.case}: {item.field}: {item.status}: "
        f"legacy={item.legacy!r}, native={item.native!r}, detail={item.detail!r}"
        for item in failures[:30]
    )
    assert {case.name for case in cases} == native_parity.QUICK_CASE_NAMES
    assert len(cases) == 8
    assert summary["rust_owned_derived_fields"] == 84
    assert summary["tolerance_gated_derived_fields"] == 82
    assert not failures, details
