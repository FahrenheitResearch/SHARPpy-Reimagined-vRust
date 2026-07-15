"""Deterministic randomized stress audit for Rust/Python sounding parity.

The release-facing real/synthetic corpus in :mod:`sharpmod.tools.native_parity`
is the source of truth for comparisons and tolerances.  This module supplies a
larger, fixed-seed perturbation matrix and feeds every generated profile through
that same auditor.  It deliberately owns no second set of tolerances.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import numpy.ma as ma


SEED = 0x5A17C0DE
MISSING = -9999.0
FAMILY_COUNTS = {
    "hrrr_dense": 50,
    "bufkit_upper_dry": 25,
    "oax_sparse": 15,
    "hrrr_edge": 10,
}
EDGE_SCENARIOS = (
    "stable",
    "near_saturated",
    "hot_moist",
    "dry_mixed",
    "strong_shear",
)
SMOKE_CASE_IDS = (0, 1, 50, 75, 90)


@dataclass(frozen=True)
class FuzzCase:
    """One generated sounding plus the parameters required to reproduce it."""

    name: str
    source: str
    kwargs: dict
    metadata: dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _filled(value) -> np.ndarray:
    array = ma.asarray(value, dtype=float)
    return np.asarray(array.filled(MISSING), dtype=float).copy()


def _raw_profile(collection):
    profiles = collection._profs  # noqa: SLF001 - decoder compatibility store
    members = profiles[next(iter(profiles))]
    return members[0]


def _profile_kwargs(profile, *, include_omega: bool = True) -> dict:
    result = {
        name: _filled(getattr(profile, name))
        for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd")
    }
    if include_omega and getattr(profile, "omeg", None) is not None:
        omega = ma.asarray(getattr(profile, "omeg"), dtype=float)
        if omega.ndim == 1 and omega.size == result["pres"].size:
            result["omeg"] = np.asarray(
                omega.filled(MISSING), dtype=float).copy()
    result.update(
        latitude=float(getattr(profile, "latitude", 36.0) or 36.0),
        location=str(getattr(profile, "location", "FUZZ")),
        date=getattr(profile, "date", datetime(2026, 7, 13, 0, 0)),
        missing=MISSING,
        strictQC=False,
    )
    return result


def _load_bases(root: Path) -> tuple[dict, dict, list[dict]]:
    # Importing decoders lazily keeps the generator's simple metadata tests
    # independent of the optional native extension.
    from sharpmod.io import decoder as decoder_mod
    from sharpmod.io.decoder import load_npz

    examples = root / "examples" / "soundings"
    collection, _ = load_npz(
        str(examples / "hrrr_point_36.68N_95.66W_f018.npz"))
    hrrr = _profile_kwargs(_raw_profile(collection))

    spc_decoder = decoder_mod.getDecoder("spc")
    collection = spc_decoder(str(examples / "14061619.OAX")).getProfiles()
    oax = _profile_kwargs(_raw_profile(collection), include_omega=False)

    bufkit_decoder = decoder_mod.getDecoder("bufkit")
    collection = bufkit_decoder(
        str(examples / "hrrr_kbvo_20260625_06z.buf")).getProfiles()
    profiles = collection._profs  # noqa: SLF001 - decoder compatibility store
    bufkit = [
        _profile_kwargs(profile)
        for profile in profiles[next(iter(profiles))]
    ]
    return hrrr, oax, bufkit


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values > -9000.0)


def _copy_kwargs(kwargs: dict) -> dict:
    return {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in kwargs.items()
    }


def _perturb(
        base: dict,
        rng: np.random.Generator,
        case_id: int,
        family: str,
        *,
        lead: int | None = None,
        scenario: str | None = None,
) -> tuple[dict, dict]:
    kwargs = _copy_kwargs(base)
    pressure = kwargs["pres"]
    height = kwargs["hght"]
    temperature = kwargs["tmpc"]
    dewpoint = kwargs["dwpc"]
    wind_direction = kwargs["wdir"]
    wind_speed = kwargs["wspd"]

    thermo = _finite(pressure) & _finite(height) & _finite(temperature)
    moisture = thermo & _finite(dewpoint)
    wind = thermo & _finite(wind_direction) & _finite(wind_speed)
    surface_candidates = np.flatnonzero(thermo & moisture)
    if surface_candidates.size:
        surface = int(surface_candidates[0])
    else:
        surface = int(np.flatnonzero(thermo)[0])
    height_agl = np.maximum(0.0, height - height[surface])
    depth_fraction = np.clip(height_agl / 12000.0, 0.0, 1.0)

    temperature_bias = float(rng.uniform(-5.0, 5.0))
    lapse_delta = float(rng.uniform(-3.0, 3.0))
    dewpoint_depression_scale = float(rng.uniform(0.55, 1.65))
    wind_scale = float(rng.uniform(0.65, 1.45))
    turn_degrees = float(rng.uniform(-32.0, 32.0))
    added_u_6km = float(rng.uniform(-18.0, 18.0))
    added_v_6km = float(rng.uniform(-18.0, 18.0))
    elevation_offset = float(rng.uniform(-250.0, 1400.0))

    if scenario == "stable":
        temperature_bias = -3.0
        lapse_delta = 7.0
        dewpoint_depression_scale = 1.3
    elif scenario == "near_saturated":
        temperature_bias = 2.0
        lapse_delta = -1.0
        dewpoint_depression_scale = 0.12
    elif scenario == "hot_moist":
        temperature_bias = 7.0
        lapse_delta = -2.5
        dewpoint_depression_scale = 0.35
    elif scenario == "dry_mixed":
        temperature_bias = 4.0
        lapse_delta = 1.5
        dewpoint_depression_scale = 2.25
    elif scenario == "strong_shear":
        wind_scale = 1.5
        added_u_6km = 35.0
        added_v_6km = -25.0
        turn_degrees = 45.0

    original_temperature = temperature.copy()
    temperature[thermo] += (
        temperature_bias + lapse_delta * depth_fraction[thermo])
    original_depression = np.maximum(
        0.0, original_temperature[moisture] - dewpoint[moisture])
    new_depression = np.maximum(
        0.05, original_depression * dewpoint_depression_scale)
    dewpoint[moisture] = np.maximum(
        -110.0, temperature[moisture] - new_depression)

    if np.any(wind):
        angle = np.deg2rad(wind_direction[wind])
        u_wind = -wind_speed[wind] * np.sin(angle)
        v_wind = -wind_speed[wind] * np.cos(angle)
        layer_fraction = np.clip(height_agl[wind] / 6000.0, 0.0, 2.0)
        rotation = np.deg2rad(
            turn_degrees * np.clip(layer_fraction, 0.0, 1.0))
        rotated_u = u_wind * np.cos(rotation) - v_wind * np.sin(rotation)
        rotated_v = u_wind * np.sin(rotation) + v_wind * np.cos(rotation)
        rotated_u = wind_scale * rotated_u + added_u_6km * layer_fraction
        rotated_v = wind_scale * rotated_v + added_v_6km * layer_fraction
        wind_speed[wind] = np.hypot(rotated_u, rotated_v)
        wind_direction[wind] = np.mod(
            np.rad2deg(np.arctan2(-rotated_u, -rotated_v)), 360.0)

    height[thermo] += elevation_offset
    # Alternating signs guarantees equal northern/southern coverage while the
    # magnitude continues to vary randomly.
    kwargs["latitude"] = float(
        (1 if case_id % 2 == 0 else -1) * rng.uniform(12.0, 62.0))

    missing_upper_moisture = False
    internal_missing_wind = False
    omega_missing = False

    if family == "hrrr_dense" and case_id % 5 == 0:
        cutoff = float(rng.uniform(8500.0, 13000.0))
        mask = moisture & (height_agl >= cutoff)
        dewpoint[mask] = MISSING
        missing_upper_moisture = bool(np.any(mask))
    elif family == "bufkit_upper_dry":
        missing_upper_moisture = bool(np.any(dewpoint < -9000.0))

    if (
        (family == "hrrr_dense" and case_id % 7 == 0)
        or (family == "bufkit_upper_dry" and case_id % 4 == 0)
    ):
        center = float(rng.uniform(1800.0, 8500.0))
        half_width = float(rng.uniform(250.0, 900.0))
        mask = wind & (np.abs(height_agl - center) <= half_width)
        if np.any(mask):
            wind_direction[mask] = MISSING
            wind_speed[mask] = MISSING
            internal_missing_wind = True

    if "omeg" in kwargs:
        omega = kwargs["omeg"]
        if family == "hrrr_dense" and case_id % 4 == 0:
            omega[:] = MISSING
            omega_missing = True
        else:
            valid_omega = _finite(omega)
            omega[valid_omega] *= float(rng.uniform(0.5, 1.5))

    metadata = {
        "case_id": case_id,
        "family": family,
        "base_lead": lead,
        "scenario": scenario,
        "temperature_bias_c": round(temperature_bias, 4),
        "upper_temperature_delta_c": round(lapse_delta, 4),
        "dewpoint_depression_scale": round(dewpoint_depression_scale, 4),
        "wind_scale": round(wind_scale, 4),
        "turn_degrees": round(turn_degrees, 4),
        "added_u_6km_kt": round(added_u_6km, 4),
        "added_v_6km_kt": round(added_v_6km, 4),
        "elevation_offset_m": round(elevation_offset, 4),
        "latitude": round(float(kwargs["latitude"]), 4),
        "missing_upper_moisture": missing_upper_moisture,
        "internal_missing_wind": internal_missing_wind,
        "omega_missing": omega_missing,
    }
    return kwargs, metadata


def build_fuzz_cases(
        root: Path | None = None,
        *,
        seed: int = SEED,
) -> list[FuzzCase]:
    """Return the complete 100-profile deterministic stress matrix."""
    root = _repo_root() if root is None else Path(root)
    rng = np.random.default_rng(seed)
    hrrr, oax, bufkit = _load_bases(root)
    generated: list[FuzzCase] = []

    def append(base, family, *, lead=None, scenario=None):
        case_id = len(generated)
        kwargs, metadata = _perturb(
            base, rng, case_id, family, lead=lead, scenario=scenario)
        generated.append(FuzzCase(
            name=f"fuzz-{case_id:03d}-{family}",
            source=f"fixed-seed/{family}",
            kwargs=kwargs,
            metadata=metadata,
        ))

    for _ in range(FAMILY_COUNTS["hrrr_dense"]):
        append(hrrr, "hrrr_dense")
    for _ in range(FAMILY_COUNTS["bufkit_upper_dry"]):
        lead = int(rng.integers(0, len(bufkit)))
        append(bufkit[lead], "bufkit_upper_dry", lead=lead)
    for _ in range(FAMILY_COUNTS["oax_sparse"]):
        append(oax, "oax_sparse")
    for index in range(FAMILY_COUNTS["hrrr_edge"]):
        append(hrrr, "hrrr_edge", scenario=EDGE_SCENARIOS[index % 5])
    return generated


def matrix_fingerprint(cases: Sequence[FuzzCase]) -> str:
    """Hash all generated numerical inputs and metadata in stable order."""
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.name.encode("utf-8"))
        digest.update(case.source.encode("utf-8"))
        digest.update(json.dumps(
            case.metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        for key in sorted(case.kwargs):
            value = case.kwargs[key]
            digest.update(key.encode("utf-8"))
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value, dtype="<f8")
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest().upper()


def feature_summary(cases: Sequence[FuzzCase]) -> dict:
    family_counts = Counter(case.metadata["family"] for case in cases)
    return {
        "families": dict(sorted(family_counts.items())),
        "northern_hemisphere": sum(
            case.metadata["latitude"] > 0.0 for case in cases),
        "southern_hemisphere": sum(
            case.metadata["latitude"] < 0.0 for case in cases),
        "missing_upper_moisture": sum(
            case.metadata["missing_upper_moisture"] for case in cases),
        "internal_missing_wind": sum(
            case.metadata["internal_missing_wind"] for case in cases),
        "missing_omega": sum(
            case.metadata["omega_missing"] for case in cases),
        "negative_elevation_perturbation": sum(
            case.metadata["elevation_offset_m"] < 0.0 for case in cases),
        "positive_elevation_perturbation": sum(
            case.metadata["elevation_offset_m"] > 0.0 for case in cases),
    }


def _extension_identity() -> dict:
    from sharpmod import sharpmod_native
    from sharpmod.sharptab import native_analysis

    path = Path(sharpmod_native.__file__).resolve()
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "backend": native_analysis.backend_info(),
    }


def audit_fuzz_cases(cases: Sequence[FuzzCase]):
    """Audit generated cases with the canonical corpus comparison engine."""
    from sharpmod.tools import native_parity

    parity_cases = [
        native_parity.Case(
            name=case.name,
            source=case.source,
            kwargs=_copy_kwargs(case.kwargs),
        )
        for case in cases
    ]
    return native_parity.audit(parity_cases)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def report_payload(
        cases: Sequence[FuzzCase], comparisons: Sequence, summary: dict,
        *, seed: int = SEED, full_matrix_sha256: str | None = None,
) -> dict:
    from sharpmod.tools import native_parity

    failures = [comparison for comparison in comparisons if not comparison.passed]
    informational = [
        comparison for comparison in comparisons
        if comparison.status in {"informational", "normalized-sampling"}
    ]
    return {
        "schema": "sharpmod.native-parity-fuzz.v1",
        "seed": seed,
        "full_matrix_sha256": (
            full_matrix_sha256 or matrix_fingerprint(cases)),
        "selected_matrix_sha256": matrix_fingerprint(cases),
        "case_count": len(cases),
        "features": feature_summary(cases),
        "native_extension": _extension_identity(),
        "derived_field_coverage": native_parity.DERIVED_FIELD_COVERAGE,
        "external_oracles": native_parity.EXTERNAL_ORACLE_FIELDS,
        "intentional_non_gating": native_parity.INFORMATIONAL_DERIVED_FIELDS,
        "tolerance_rule": (
            "abs(error) <= max(field absolute tolerance, "
            "field relative tolerance * abs(legacy))"
        ),
        "summary": summary,
        "failures": [asdict(comparison) for comparison in failures],
        "informational": [
            asdict(comparison) for comparison in informational],
        "case_metadata": {
            case.name: case.metadata for case in cases
        },
    }


def _select_cases(cases: Sequence[FuzzCase], args) -> list[FuzzCase]:
    if args.smoke:
        selected_ids = set(SMOKE_CASE_IDS)
        return [
            case for case in cases
            if case.metadata["case_id"] in selected_ids
        ]
    if args.limit is not None:
        return list(cases[:args.limit])
    return list(cases)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=lambda value: int(value, 0), default=SEED,
        help="generator seed (default: fixed release seed 0x5A17C0DE)")
    parser.add_argument(
        "--smoke", action="store_true",
        help="audit the five representative smoke cases")
    parser.add_argument(
        "--limit", type=int,
        help="audit only the first N generated cases")
    parser.add_argument(
        "--json", type=Path,
        help="write a machine-readable report to this path")
    parser.add_argument(
        "--show-failures", type=int, default=20,
        help="maximum failing comparisons printed to stdout")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    cases = build_fuzz_cases(seed=args.seed)
    selected = _select_cases(cases, args)
    full_matrix_sha256 = matrix_fingerprint(cases)
    print(json.dumps({
        "event": "start",
        "seed": args.seed,
        "case_count": len(selected),
        "full_matrix_sha256": full_matrix_sha256,
        "features": feature_summary(selected),
        "native_extension": _extension_identity(),
    }, sort_keys=True, default=_jsonable), flush=True)

    comparisons, summary = audit_fuzz_cases(selected)
    failures = [comparison for comparison in comparisons if not comparison.passed]
    payload = report_payload(
        selected,
        comparisons,
        summary,
        seed=args.seed,
        full_matrix_sha256=full_matrix_sha256,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_jsonable)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "event": "summary",
        "case_count": len(selected),
        "comparison_count": len(comparisons),
        "failure_count": len(failures),
        "summary": summary,
    }, sort_keys=True, default=_jsonable), flush=True)
    for comparison in failures[:args.show_failures]:
        print(
            f"FAIL {comparison.case} {comparison.group}.{comparison.field}: "
            f"{comparison.status}; legacy={comparison.legacy!r}; "
            f"native={comparison.native!r}; {comparison.detail or ''}",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
