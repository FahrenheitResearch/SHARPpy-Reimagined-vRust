"""Reproducible Rust-versus-legacy SHARPpy numerical parity audit.

This module deliberately compares the *public profile objects* used by the
viewer.  That tests both the native calculation crates and the Python adapter
which translates their results back into SHARPpy's object model.

Run the complete bundled corpus from a source checkout with::

    python -m sharpmod.tools.native_parity

The command returns a non-zero status when a backend fails, a missing-value
state differs, a categorical result differs, or a numeric result exceeds its
field-specific tolerance.  ``--json`` writes the same evidence in a
machine-readable form suitable for release artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import numpy.ma as ma


@dataclass(frozen=True)
class Tolerance:
    """Symmetric ``abs + rel`` numerical agreement contract."""

    absolute: float
    relative: float = 0.05


@dataclass(frozen=True)
class FieldSpec:
    name: str
    tolerance: Tolerance
    group: str


@dataclass
class Case:
    name: str
    source: str
    kwargs: dict


@dataclass
class Comparison:
    case: str
    source: str
    group: str
    field: str
    status: str
    legacy: object = None
    native: object = None
    absolute_error: float | None = None
    relative_error: float | None = None
    allowed_error: float | None = None
    detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.status in {
            "within-tolerance", "both-missing", "equal", "informational",
            "normalized-sampling",
        }


_ENERGY = Tolerance(10.0, 0.05)
_HEIGHT = Tolerance(50.0, 0.05)
_PRESSURE = Tolerance(2.0, 0.01)
_TEMPERATURE = Tolerance(0.25, 0.02)
_INDEX = Tolerance(0.05, 0.05)
_WIND = Tolerance(0.5, 0.05)
_SRH = Tolerance(5.0, 0.05)


SCALAR_FIELDS = (
    FieldSpec("pwat", Tolerance(0.05), "thermodynamics"),
    FieldSpec("mean_mixr", Tolerance(0.10), "thermodynamics"),
    FieldSpec("low_rh", Tolerance(1.0), "thermodynamics"),
    FieldSpec("mid_rh", Tolerance(1.0), "thermodynamics"),
    FieldSpec("k_idx", Tolerance(0.25), "thermodynamics"),
    FieldSpec("totals_totals", Tolerance(0.25), "thermodynamics"),
    FieldSpec("convT", Tolerance(1.0, 0.02), "thermodynamics"),
    FieldSpec("maxT", Tolerance(1.0, 0.02), "thermodynamics"),
    FieldSpec("lapserate_3km", Tolerance(0.10), "thermodynamics"),
    FieldSpec("lapserate_3_6km", Tolerance(0.10), "thermodynamics"),
    FieldSpec("lapserate_700_500", Tolerance(0.10), "thermodynamics"),
    FieldSpec("lapserate_850_500", Tolerance(0.10), "thermodynamics"),
    FieldSpec("dcape", _ENERGY, "downdraft"),
    FieldSpec("drush", Tolerance(1.0, 0.02), "downdraft"),
    FieldSpec("mburst", _INDEX, "indices"),
    FieldSpec("ebottom", _PRESSURE, "effective-layer"),
    FieldSpec("etop", _PRESSURE, "effective-layer"),
    FieldSpec("ebotm", _HEIGHT, "effective-layer"),
    FieldSpec("etopm", _HEIGHT, "effective-layer"),
    FieldSpec("ebwspd", _WIND, "effective-layer"),
    FieldSpec("right_scp", _INDEX, "severe"),
    FieldSpec("left_scp", _INDEX, "severe"),
    FieldSpec("right_stp_cin", _INDEX, "severe"),
    FieldSpec("left_stp_cin", _INDEX, "severe"),
    FieldSpec("right_stp_fixed", _INDEX, "severe"),
    FieldSpec("left_stp_fixed", _INDEX, "severe"),
    FieldSpec("ship", _INDEX, "severe"),
    FieldSpec("sig_severe", Tolerance(10.0), "severe"),
    FieldSpec("wndg", _INDEX, "severe"),
    FieldSpec("esp", _INDEX, "severe"),
    FieldSpec("tei", Tolerance(0.25), "indices"),
    FieldSpec("sherbe", _INDEX, "indices"),
    FieldSpec("right_critical_angle", Tolerance(2.0), "kinematics"),
    FieldSpec("left_critical_angle", Tolerance(2.0), "kinematics"),
    FieldSpec("updraft_tilt", Tolerance(2.0), "trajectory"),
)


# Upstream SHARPpy's MMP implementation leaves elements of an ``np.empty``
# work array uninitialized whenever its ``if b < t`` branch is false, then
# applies ``nanmax``.  Its output can consequently vary with allocator state.
# Rust initializes every element and is deterministic; preserve the legacy
# comparison as evidence, but never use undefined oracle memory as a release
# gate.
INFORMATIONAL_FIELDS = (
    FieldSpec("mmp", _INDEX, "corrected-legacy-undefined"),
)


VECTOR_FIELDS = (
    FieldSpec("srwind", _WIND, "kinematics"),
    FieldSpec("sfc_1km_shear", _WIND, "kinematics"),
    FieldSpec("sfc_3km_shear", _WIND, "kinematics"),
    FieldSpec("sfc_6km_shear", _WIND, "kinematics"),
    FieldSpec("sfc_8km_shear", _WIND, "kinematics"),
    FieldSpec("sfc_9km_shear", _WIND, "kinematics"),
    FieldSpec("eff_shear", _WIND, "effective-layer"),
    FieldSpec("ebwd", _WIND, "effective-layer"),
    FieldSpec("lcl_el_shear", _WIND, "kinematics"),
    FieldSpec("mean_eff", _WIND, "effective-layer"),
    FieldSpec("mean_ebw", _WIND, "effective-layer"),
    FieldSpec("right_srw_0_2km", _WIND, "kinematics"),
    FieldSpec("right_srw_4_6km", _WIND, "kinematics"),
    FieldSpec("right_srw_9_11km", _WIND, "kinematics"),
    FieldSpec("right_srw_eff", _WIND, "effective-layer"),
    FieldSpec("right_srw_ebw", _WIND, "effective-layer"),
    FieldSpec("left_srw_0_2km", _WIND, "kinematics"),
    FieldSpec("left_srw_4_6km", _WIND, "kinematics"),
    FieldSpec("left_srw_9_11km", _WIND, "kinematics"),
    FieldSpec("left_srw_eff", _WIND, "effective-layer"),
    FieldSpec("left_srw_ebw", _WIND, "effective-layer"),
    FieldSpec("upshear_downshear", _WIND, "kinematics"),
    FieldSpec("right_srh1km", _SRH, "kinematics"),
    FieldSpec("left_srh1km", _SRH, "kinematics"),
    FieldSpec("right_srh3km", _SRH, "kinematics"),
    FieldSpec("left_srh3km", _SRH, "kinematics"),
    FieldSpec("right_esrh", _SRH, "effective-layer"),
    FieldSpec("left_esrh", _SRH, "effective-layer"),
)


# These public tuples are meteorological direction/speed rather than Cartesian
# components.  They are compared after conversion to u/v so 359 degrees and
# 1 degree are treated as two degrees apart, not 358.
POLAR_WIND_FIELDS = (
    "wind1km", "wind6km",
    "mean_1km", "mean_3km", "mean_6km", "mean_8km", "mean_lcl_el",
    "right_srw_1km", "right_srw_3km", "right_srw_6km", "right_srw_8km",
    "right_srw_4_5km", "right_srw_lcl_el",
    "left_srw_1km", "left_srw_3km", "left_srw_6km", "left_srw_8km",
    "left_srw_4_5km", "left_srw_lcl_el",
)


PARCEL_FIELDS = {
    "pres": _PRESSURE,
    "tmpc": _TEMPERATURE,
    "dwpc": _TEMPERATURE,
    "bplus": _ENERGY,
    "bminus": _ENERGY,
    "b3km": _ENERGY,
    "b6km": _ENERGY,
    "bfzl": _ENERGY,
    "lclpres": _PRESSURE,
    "lfcpres": _PRESSURE,
    "elpres": _PRESSURE,
    "lclhght": _HEIGHT,
    "lfchght": _HEIGHT,
    "elhght": _HEIGHT,
    "li5": _TEMPERATURE,
    "limax": _TEMPERATURE,
    "limaxpres": _PRESSURE,
    "cap": _ENERGY,
    "cappres": _PRESSURE,
    "mplpres": _PRESSURE,
    "mplhght": _HEIGHT,
    "bmin": _TEMPERATURE,
    "bminpres": _PRESSURE,
    "li3": _TEMPERATURE,
    "p0c": _PRESSURE,
    "pm10c": _PRESSURE,
    "pm20c": _PRESSURE,
    "pm30c": _PRESSURE,
    "hght0c": _HEIGHT,
    "hghtm10c": _HEIGHT,
    "hghtm20c": _HEIGHT,
    "hghtm30c": _HEIGHT,
    "wm10c": _ENERGY,
    "wm20c": _ENERGY,
    "wm30c": _ENERGY,
}


CATEGORICAL_FIELDS = (
    "right_watch_type",
    "left_watch_type",
    "precip_type",
)


ENVIRONMENT_ARRAYS = {
    "pres": Tolerance(1.0e-6, 1.0e-9),
    "hght": Tolerance(1.0e-6, 1.0e-9),
    "tmpc": Tolerance(1.0e-6, 1.0e-9),
    "dwpc": Tolerance(1.0e-6, 1.0e-9),
    "wdir": Tolerance(1.0e-6, 1.0e-9),
    "wspd": Tolerance(1.0e-6, 1.0e-9),
    "omeg": Tolerance(1.0e-6, 1.0e-9),
    "u": Tolerance(1.0e-6, 1.0e-9),
    "v": Tolerance(1.0e-6, 1.0e-9),
    "logp": Tolerance(1.0e-8, 1.0e-8),
    "vtmp": _TEMPERATURE,
    "theta": Tolerance(0.25, 0.01),
    "thetae": Tolerance(0.75, 0.01),
    "wvmr": Tolerance(0.10, 0.02),
    "relh": Tolerance(1.0, 0.02),
    "wetbulb": _TEMPERATURE,
}


COMPANION_FIELDS = (
    FieldSpec("dcp", _INDEX, "extended-derived"),
    FieldSpec("lapserate_sfc_500m", Tolerance(0.10), "extended-derived"),
    FieldSpec("lapserate_sfc_1km", Tolerance(0.10), "extended-derived"),
    FieldSpec("srh500", _SRH, "extended-derived"),
    FieldSpec("mean_wind_sfc_500m", _WIND, "extended-derived"),
    FieldSpec("srw_sfc_500m", _WIND, "extended-derived"),
    FieldSpec("vgp", _INDEX, "extended-derived"),
    FieldSpec("peskov", _INDEX, "extended-derived"),
    FieldSpec("mcs_index", _INDEX, "extended-derived"),
    FieldSpec("ncape", _INDEX, "extended-derived"),
    FieldSpec("lrghail", _INDEX, "extended-derived"),
    FieldSpec("lscp", _INDEX, "extended-derived"),
    FieldSpec("nstp", _INDEX, "extended-derived"),
    FieldSpec("hgz_cape", _ENERGY, "extended-derived"),
    FieldSpec("wbz_height", _HEIGHT, "extended-derived"),
    FieldSpec("modified_sherbe", _INDEX, "extended-derived"),
    FieldSpec("ehi_0_1km", _INDEX, "extended-derived"),
    FieldSpec("ehi_0_3km", _INDEX, "extended-derived"),
    FieldSpec("cape_0_6km", _ENERGY, "extended-derived"),
)


FIRE_FIELDS = {
    "fosberg": Tolerance(0.5),
    "haines_low": _INDEX,
    "haines_mid": _INDEX,
    "haines_high": _INDEX,
    "bplus_fire": _ENERGY,
}


WINTER_FIELDS = {
    "dgz_pbot": _PRESSURE,
    "dgz_ptop": _PRESSURE,
    "dgz_meanrh": Tolerance(1.0),
    "dgz_pw": Tolerance(0.05),
    "dgz_meanq": Tolerance(0.10),
}


# Every key returned by ``native["derived"]`` is named here.  The value is the
# public/adapted comparison label used above.  This makes additions to the Rust
# schema fail the audit until an oracle and tolerance are chosen; no field can
# silently fall outside the assurance claim.
DERIVED_FIELD_COVERAGE = {
    "pwat": "pwat",
    "k_idx": "k_idx",
    "tei": "tei",
    "esp": "esp",
    "mmp": "mmp (informational: undefined legacy memory)",
    "wndg": "wndg",
    "dcp": "Python-only companion dcp",
    "mburst": "mburst",
    "ship": "ship",
    "right_scp": "right_scp",
    "left_scp": "left_scp",
    "stp_cin": "hemisphere-selected stp_cin",
    "stp_fixed": "hemisphere-selected stp_fixed",
    "sweat": "sweat",
    "sig_severe": "sig_severe",
    "dcape": "dcape",
    "drush_f": "drush",
    "mean_mixr": "mean_mixr",
    "low_rh": "low_rh",
    "mid_rh": "mid_rh",
    "totals_totals": "totals_totals",
    "conv_t_f": "convT",
    "max_t_f": "maxT",
    "thetae_diff": "thetae_diff",
    "lapserate_3km": "lapserate_3km",
    "lapserate_3_6km": "lapserate_3_6km",
    "lapserate_850_500": "lapserate_850_500",
    "lapserate_700_500": "lapserate_700_500",
    "lapserate_sfc_500m": "Python-only companion lapserate_sfc_500m",
    "lapserate_sfc_1km": "Python-only companion lapserate_sfc_1km",
    "srh500": "Python-only companion srh500",
    "srh1km": "right_srh1km",
    "srh3km": "right_srh3km",
    "right_esrh": "right_esrh",
    "sfc_500m_shear": "Python-only companion shear_sfc_500m",
    "sfc_1km_shear": "sfc_1km_shear",
    "sfc_3km_shear": "sfc_3km_shear",
    "sfc_6km_shear": "sfc_6km_shear",
    "sfc_8km_shear": "sfc_8km_shear",
    "eff_shear": "eff_shear",
    "ebwd": "ebwd",
    "lcl_el_shear": "lcl_el_shear",
    "mean_wind_sfc_500m": "Python-only companion mean_wind_sfc_500m",
    "mean_1km": "mean_1km as u/v",
    "mean_3km": "mean_3km as u/v",
    "mean_6km": "mean_6km as u/v",
    "mean_8km": "mean_8km as u/v",
    "mean_eff": "mean_eff",
    "mean_ebw": "mean_ebw",
    "mean_lcl_el": "mean_lcl_el as u/v",
    "srw_sfc_500m": "Python-only companion srw_sfc_500m",
    "srw_1km": "right_srw_1km as u/v",
    "srw_3km": "right_srw_3km as u/v",
    "srw_6km": "right_srw_6km as u/v",
    "srw_8km": "right_srw_8km as u/v",
    "srw_4_5km": "right_srw_4_5km as u/v",
    "srw_eff": "right_srw_eff",
    "srw_ebw": "right_srw_ebw",
    "srw_lcl_el": "right_srw_lcl_el as u/v",
    "wind1km": "wind1km as u/v",
    "wind6km": "wind6km as u/v",
    "corfidi_up": "upshear_downshear[0:2]",
    "corfidi_dn": "upshear_downshear[2:4]",
    "right_critical_angle": "right_critical_angle",
    "brnshear": "mupcl.brnshear",
    "ehi_0_1km": "Python-only companion ehi_0_1km",
    "ehi_0_3km": "Python-only companion ehi_0_3km",
    "vgp": "Python-only companion vgp",
    "peskov": "Python-only companion peskov",
    "mcs_index": "Python-only companion mcs_index",
    "ncape": "Python-only companion normalized CAPE",
    "lrghail": "Python-only companion lrghail",
    "lscp": "Python-only companion lscp",
    "nstp": "Python-only companion nstp",
    "hgz_cape": "Python-only companion hgz_cape",
    "wbz_height": "Python-only companion wbz_height",
    "ecape": "dedicated ecape-parcel gate (not legacy SHARPpy)",
    "modified_sherbe": "Python-only companion modified_sherbe",
    "cape_0_3km": "Python-only layer_cape_agl(0, 3 km)",
    "cape_0_6km": "Python-only companion cape_0_6km",
    "temp_adv": "inf_temp_adv[0]",
    "temp_adv_bounds": "inf_temp_adv[1]",
    "slinky_traj": (
        "x/y geometry resampled by normalized path; legacy-only height column "
        "is not part of native renderer output"
    ),
    "slinky_tilt": "updraft_tilt",
}


EXTERNAL_ORACLE_FIELDS = {
    "ecape": "sharpmod/tests/test_ecape_rust_parity.py uses ecape-parcel-py",
}


INFORMATIONAL_DERIVED_FIELDS = {
    "mmp": "legacy SHARPpy reads uninitialized np.empty cells",
}


QUICK_CASE_NAMES = frozenset({
    "spc-oax-2014061619",
    "spc-hrrr-2026062600",
    "bufkit-kbvo-2026062506",
    "bufkit-kbvo-2026062519",
    "bufkit-kbvo-2026062523",
    # Sparse upper-level moisture: exercises missing-value propagation through
    # effective layers, parcel tops, storm motions, and trajectories.
    "bufkit-kbvo-2026062623",
    "bufkit-kbvo-2026062703",
    "npz-hrrr-point",
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _numpy_legacy_aliases() -> None:
    """Keep the legacy oracle runnable without importing the Qt shim."""

    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    # The vendored PWV climatology calls one-argument ``np.where`` on a scalar.
    # NumPy >=1.25 rejects that 0-d nonzero operation.  The production Qt shim
    # already promotes only this condition; repeat that narrowly here so an
    # observed SPC profile can serve as an oracle without requiring Qt.
    from sharppy.databases import pwv as pwv_module

    class _Where0dProxy:
        def __init__(self, module):
            self._module = module

        def __getattr__(self, name):
            return getattr(self._module, name)

        def where(self, condition, *args, **kwargs):
            return self._module.where(
                self._module.atleast_1d(condition), *args, **kwargs)

    if not isinstance(getattr(pwv_module, "np", None), _Where0dProxy):
        # Avoid nesting proxies when ``audit`` is called repeatedly in-process.
        underlying = getattr(pwv_module, "np", np)
        if hasattr(underlying, "_module"):
            underlying = underlying._module
        pwv_module.np = _Where0dProxy(underlying)


def _copy_kwargs(kwargs: dict) -> dict:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in kwargs.items()
    }


def _raw_array(raw, name: str) -> np.ndarray | None:
    value = getattr(raw, name, None)
    if value is None or np.ndim(value) == 0:
        return None
    return np.asarray(ma.asarray(value).filled(-9999.0), dtype=float).copy()


def _raw_kwargs(raw) -> dict:
    result = {
        name: _raw_array(raw, name)
        for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd")
    }
    omeg = _raw_array(raw, "omeg")
    result["omeg"] = (
        omeg if omeg is not None
        else np.full_like(result["pres"], -9999.0, dtype=float)
    )
    result.update(
        latitude=float(getattr(raw, "latitude", 35.0) or 35.0),
        location=str(getattr(raw, "location", "TST")),
        date=getattr(raw, "date", datetime(2020, 1, 1)),
        missing=-9999.0,
        strictQC=False,
    )
    return result


def _npz_kwargs(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {
            name: np.asarray(data[name], dtype=float).copy()
            for name in ("pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg")
        }
        result.update(
            latitude=float(data["lat"]),
            location=str(data["loc"]),
            date=datetime.strptime(str(data["valid"]), "%Y-%m-%d %H:%M"),
            missing=-9999.0,
            strictQC=False,
        )
    return result


def _synthetic_cases(base: dict) -> list[Case]:
    def transformed(name, **updates):
        values = _copy_kwargs(base)
        for key, value in updates.items():
            values[key] = np.asarray(value, dtype=float) if key in {
                "pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg"
            } else value
        values["location"] = name
        return Case(name, "deterministic synthetic transform", values)

    hght = np.asarray(base["hght"], dtype=float)
    agl_km = (hght - hght[0]) / 1000.0
    stable_t = 16.0 - 4.0 * np.minimum(agl_km, 12.0)
    stable_t = np.where(agl_km > 12.0, -32.0, stable_t)
    stable_td = stable_t - np.minimum(25.0, 10.0 + 1.2 * agl_km)

    moist_t = 31.0 - 7.8 * np.minimum(agl_km, 11.0)
    moist_t = np.where(agl_km > 11.0, -54.8, moist_t)
    moist_td = moist_t - np.minimum(15.0, 1.5 + 0.8 * agl_km)

    elevated_t = np.asarray(base["tmpc"], dtype=float).copy()
    elevated_td = np.asarray(base["dwpc"], dtype=float).copy()
    low = agl_km <= 1.5
    elevated_t[low] -= 5.0 * (1.0 - agl_km[low] / 1.5)
    elevated_td[low] -= 14.0 * (1.0 - agl_km[low] / 1.5)

    return [
        transformed("synthetic-stable-dry", tmpc=stable_t, dwpc=stable_td),
        transformed("synthetic-moist-unstable", tmpc=moist_t, dwpc=moist_td),
        transformed("synthetic-elevated", tmpc=elevated_t, dwpc=elevated_td),
        transformed(
            "synthetic-zero-wind",
            wdir=np.zeros_like(base["wdir"]),
            wspd=np.zeros_like(base["wspd"]),
        ),
        transformed("synthetic-southern-hemisphere", latitude=-36.68),
    ]


def build_corpus(root: Path | None = None, *, synthetic: bool = True) -> list[Case]:
    """Load every committed real sounding plus deterministic edge regimes."""

    from sharpmod.io import decoder

    root = Path(root or _repo_root())
    examples = root / "examples" / "soundings"
    sources = (
        ("spc-oax", "spc", examples / "14061619.OAX"),
        ("spc-hrrr", "spc", examples / "hrrr_point_36.68N_95.66W_f018.spc"),
        ("bufkit-kbvo", "bufkit", examples / "hrrr_kbvo_20260625_06z.buf"),
    )
    cases: list[Case] = []
    for prefix, decoder_name, path in sources:
        collection = decoder.getDecoder(decoder_name)(str(path)).getProfiles()
        profiles = next(iter(collection._profs.values()))
        for index, raw in enumerate(profiles):
            valid = getattr(raw, "date", None)
            suffix = valid.strftime("%Y%m%d%H") if valid else f"{index:02d}"
            cases.append(Case(
                f"{prefix}-{suffix}",
                str(path.relative_to(root)),
                _raw_kwargs(raw),
            ))

    npz = examples / "hrrr_point_36.68N_95.66W_f018.npz"
    base = _npz_kwargs(npz)
    cases.append(Case("npz-hrrr-point", str(npz.relative_to(root)), base))
    if synthetic:
        cases.extend(_synthetic_cases(base))
    return cases


def _scalar(value) -> float | None:
    if value is None or ma.is_masked(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= -9000.0:
        return None
    return number


def _numeric_values(value) -> list[float | None]:
    if value is None or ma.is_masked(value):
        return [None]
    array = ma.asarray(value)
    if array.ndim == 0:
        return [_scalar(value)]
    return [_scalar(item) for item in array.ravel()]


def _polar_components(value):
    values = _numeric_values(value)
    if len(values) != 2 or any(item is None for item in values):
        return (None, None)
    direction, speed = values
    radians = math.radians(direction)
    return (-speed * math.sin(radians), -speed * math.cos(radians))


def _vector_magnitude(value):
    values = _numeric_values(value)
    if len(values) != 2 or any(item is None for item in values):
        return None
    return math.hypot(*values)


def _python_companion(legacy):
    """Build the pre-Rust local derived-profile oracle explicitly."""

    from sharpmod.sharptab import profile as local_profile

    meta = {
        "date": getattr(legacy, "date", None),
        "valid": getattr(legacy, "date", None),
        "location": getattr(legacy, "location", None),
        "lat": getattr(legacy, "latitude", None),
        "lon": getattr(legacy, "longitude", None),
    }
    return local_profile.create_profile(
        pres=legacy.pres,
        hght=legacy.hght,
        tmpc=legacy.tmpc,
        dwpc=legacy.dwpc,
        wdir=legacy.wdir,
        wspd=legacy.wspd,
        omeg=getattr(legacy, "omeg", None),
        wetbulb=getattr(legacy, "wetbulb", None),
        meta={key: value for key, value in meta.items() if value is not None},
    )


def _safe_oracle(function):
    try:
        return function()
    except Exception:
        return ma.masked


def _extended_python_oracles(legacy) -> dict:
    """Run pre-Rust sharpmod formulas on the full physical legacy profile."""

    from sharpmod.sharptab import derived, params, winds

    sfc500 = _safe_oracle(lambda: winds.sfc_500m_kinematics(legacy))
    normalized = _safe_oracle(lambda: derived.normalized_cape_cin(legacy))

    def group(index):
        if not isinstance(sfc500, (tuple, list)) or len(sfc500) != 4:
            return ma.masked
        return sfc500[index]

    def normalized_cape():
        if not isinstance(normalized, (tuple, list)) or len(normalized) != 2:
            return ma.masked
        return normalized[0]

    return {
        "dcp": _safe_oracle(lambda: derived.dcp(legacy)),
        "lapserate_sfc_500m": _safe_oracle(
            lambda: params.lapse_rate(legacy, 0.0, 500.0, agl=True)),
        "lapserate_sfc_1km": _safe_oracle(
            lambda: params.lapse_rate(legacy, 0.0, 1000.0, agl=True)),
        "srh500": group(0),
        "sfc_500m_shear": group(1),
        "mean_wind_sfc_500m": group(2),
        "srw_sfc_500m": group(3),
        "vgp": _safe_oracle(lambda: derived.vorticity_generation_parameter(legacy)),
        "peskov": _safe_oracle(lambda: derived.peskov_index(legacy)),
        "mcs_index": _safe_oracle(lambda: derived.mcs_index(legacy)),
        "ncape": normalized_cape(),
        "lrghail": _safe_oracle(lambda: derived.large_hail_parameter(legacy)),
        "lscp": _safe_oracle(lambda: derived.left_supercell_composite(legacy)),
        "nstp": _safe_oracle(
            lambda: derived.non_supercell_tornado_parameter(legacy)),
        "hgz_cape": _safe_oracle(
            lambda: params.layer_cape_isotherm(legacy, -10.0, -30.0)),
        "wbz_height": _safe_oracle(lambda: derived.wet_bulb_zero_height(legacy)),
        "modified_sherbe": _safe_oracle(lambda: derived.modified_sherbe(legacy)),
        "ehi_0_1km": _safe_oracle(lambda: derived.ehi(legacy, 1000.0)),
        "ehi_0_3km": _safe_oracle(lambda: derived.ehi(legacy, 3000.0)),
        "cape_0_3km": _safe_oracle(
            lambda: params.layer_cape_agl(legacy, 0.0, 3000.0)),
        "cape_0_6km": _safe_oracle(
            lambda: params.layer_cape_agl(legacy, 0.0, 6000.0)),
    }


def _compare_numeric(
    case: Case,
    group: str,
    field: str,
    legacy,
    native,
    tolerance: Tolerance,
) -> list[Comparison]:
    expected = _numeric_values(legacy)
    actual = _numeric_values(native)
    if len(expected) != len(actual):
        return [Comparison(
            case.name, case.source, group, field, "shape-mismatch",
            len(expected), len(actual), detail="flattened value counts differ",
        )]
    results = []
    for index, (left, right) in enumerate(zip(expected, actual)):
        name = field if len(expected) == 1 else f"{field}[{index}]"
        if left is None and right is None:
            results.append(Comparison(
                case.name, case.source, group, name, "both-missing"))
            continue
        if left is None or right is None:
            results.append(Comparison(
                case.name, case.source, group, name, "missing-mismatch",
                left, right,
            ))
            continue
        error = abs(right - left)
        allowed = max(tolerance.absolute, tolerance.relative * abs(left))
        results.append(Comparison(
            case.name,
            case.source,
            group,
            name,
            "within-tolerance" if error <= allowed else "outside-tolerance",
            left,
            right,
            error,
            error / max(abs(left), 1.0e-12),
            allowed,
        ))
    return results


def _pressure_value_map(profile, field: str) -> dict[float, float]:
    pressure = ma.asarray(getattr(profile, "pres", ()), dtype=float)
    values = pressure if field == "pres" else ma.asarray(
        getattr(profile, field, ()), dtype=float)
    if pressure.ndim != 1 or values.ndim != 1:
        return {}
    count = min(len(pressure), len(values))
    p = np.asarray(pressure[:count].filled(np.nan), dtype=float)
    v = np.asarray(values[:count].filled(np.nan), dtype=float)
    valid = (
        np.isfinite(p) & np.isfinite(v) & (p > 0.0)
        & (p > -9000.0) & (v > -9000.0)
    )
    # Reported pressures are carried to 0.01 hPa in the bundled corpus.  Six
    # decimal places preserves them while treating binary round-off as equal.
    return {round(float(pi), 6): float(vi) for pi, vi in zip(p[valid], v[valid])}


def _is_leading_presurface_normalization(legacy, native) -> bool:
    """Recognize intentional removal of empty rows before the physical SFC.

    SPC/OAX-style inputs can retain pressure placeholders before the first
    valid temperature. ``BasicProfile.sfc`` points past those rows; the native
    adapter publishes the same physical suffix starting at index zero so its
    arrays remain aligned. Detect that invariant from the two profiles rather
    than special-casing a fixture name, which also covers generated variants.
    """

    try:
        old_sfc = int(getattr(legacy, "sfc", 0))
        new_sfc = int(getattr(native, "sfc", 0))
        old_pres = np.asarray(
            ma.asarray(getattr(legacy, "pres"), dtype=float).filled(np.nan))
        new_pres = np.asarray(
            ma.asarray(getattr(native, "pres"), dtype=float).filled(np.nan))
        old_tmpc = np.asarray(
            ma.asarray(getattr(legacy, "tmpc"), dtype=float).filled(np.nan))
    except (AttributeError, TypeError, ValueError):
        return False
    if old_sfc <= 0 or new_sfc != 0:
        return False
    if len(new_pres) != len(old_pres) - old_sfc:
        return False
    if np.isfinite(old_tmpc[:old_sfc]).any():
        return False
    return bool(np.allclose(
        old_pres[old_sfc:], new_pres,
        rtol=0.0, atol=1.0e-9, equal_nan=True,
    ))


def _compare_environment_on_pressure(
    case: Case,
    legacy,
    native,
    field: str,
    tolerance: Tolerance,
) -> list[Comparison]:
    old = _pressure_value_map(legacy, field)
    new = _pressure_value_map(native, field)
    old_levels = set(old)
    new_levels = set(new)
    excluded_upper = set()
    if field in {"vtmp", "theta", "thetae", "wvmr", "relh", "wetbulb"}:
        # SHARPpy's sounding-analysis contract ends at 100 hPa.  Saturation
        # quantities in sparse upper-stratospheric rows (some observed files
        # extend to ~8 hPa) are not inputs to the displayed diagnostics and are
        # numerically undefined in parts of the legacy formulas.
        excluded_upper = {
            pressure for pressure in old_levels | new_levels
            if pressure < 100.0
        }
        old_levels -= excluded_upper
        new_levels -= excluded_upper
    common = sorted(old_levels & new_levels, reverse=True)
    old_only = sorted(old_levels - new_levels, reverse=True)
    new_only = sorted(new_levels - old_levels, reverse=True)

    # Keep missing-value parity strict at every in-domain pressure.  The sole
    # sampling exception is the observed OAX file: its first row has no
    # thermodynamic data and its internal wind gaps are deliberately
    # interpolated by the native input normalizer before Rust receives it.
    known_oax_normalization = (
        _is_leading_presurface_normalization(legacy, native)
        and field in {"pres", "hght", "logp"}
    )
    status = (
        "equal" if not old_only and not new_only
        else "normalized-sampling"
        if known_oax_normalization and len(common) >= 2
        else "coverage-mismatch"
    )
    results = [Comparison(
        case.name,
        case.source,
        "environment-arrays",
        f"{field}.pressure-level-coverage",
        status,
        len(old),
        len(new),
        detail=(
            f"compared={len(common)}, "
            f"legacy_only={old_only[:5]}, native_only={new_only[:5]}"
            + (
                "; known OAX input-normalization correction"
                if status == "normalized-sampling" else ""
            )
        ),
    )]
    if excluded_upper:
        results.append(Comparison(
            case.name,
            case.source,
            "environment-arrays",
            f"{field}.upper-stratosphere-outside-analysis-domain",
            "informational",
            len(excluded_upper),
            len(excluded_upper),
            detail=(
                "pressure <100 hPa is outside the SHARPpy analysis domain; "
                f"excluded range={min(excluded_upper):g}-{max(excluded_upper):g} hPa"
            ),
        ))
    if not common:
        return results

    if field == "wdir":
        for pressure in common:
            left, right = old[pressure], new[pressure]
            angular_error = abs((right - left + 180.0) % 360.0 - 180.0)
            results.append(Comparison(
                case.name,
                case.source,
                "environment-arrays",
                f"wdir@{pressure:g}hPa",
                "within-tolerance" if angular_error <= 0.01
                else "outside-tolerance",
                left,
                right,
                angular_error,
                angular_error / 180.0,
                0.01,
            ))
        return results

    return results + _compare_numeric(
        case,
        "environment-arrays",
        f"{field}@common-pressure-levels",
        [old[pressure] for pressure in common],
        [new[pressure] for pressure in common],
        tolerance,
    )


def _clean_trace(pressure, temperature):
    p = np.asarray(ma.asarray(pressure, dtype=float).filled(np.nan), dtype=float).ravel()
    t = np.asarray(ma.asarray(temperature, dtype=float).filled(np.nan), dtype=float).ravel()
    count = min(len(p), len(t))
    valid = (
        np.isfinite(p[:count]) & np.isfinite(t[:count])
        & (p[:count] > 0.0) & (p[:count] > -9000.0) & (t[:count] > -9000.0)
    )
    p, t = p[:count][valid], t[:count][valid]
    if len(p) == 0:
        return p, t
    order = np.argsort(p)
    p, t = p[order], t[order]
    unique, index = np.unique(p, return_index=True)
    return unique, t[index]


def _compare_pressure_temperature_trace(
    case: Case,
    group: str,
    name: str,
    legacy_pressure,
    legacy_temperature,
    native_pressure,
    native_temperature,
) -> list[Comparison]:
    old_p, old_t = _clean_trace(legacy_pressure, legacy_temperature)
    new_p, new_t = _clean_trace(native_pressure, native_temperature)
    if len(old_p) == 0 and len(new_p) == 0:
        return [Comparison(case.name, case.source, group, name, "both-missing")]
    if len(old_p) < 2 or len(new_p) < 2:
        return [Comparison(
            case.name, case.source, group, name, "missing-mismatch",
            len(old_p), len(new_p), detail="trace has fewer than two valid points")]

    results = []
    for label, old_value, new_value in (
        ("bottom-pressure", old_p[-1], new_p[-1]),
        ("top-pressure", old_p[0], new_p[0]),
    ):
        results.extend(_compare_numeric(
            case, group, f"{name}.{label}", old_value, new_value, _PRESSURE))

    lower = max(old_p[0], new_p[0])
    upper = min(old_p[-1], new_p[-1])
    if lower >= upper:
        results.append(Comparison(
            case.name, case.source, group, name, "shape-mismatch",
            detail="traces have no overlapping pressure interval"))
        return results
    # A fixed log-pressure grid compares the physical curve independently of
    # either solver's internal integration levels or ordering.
    grid = np.geomspace(lower, upper, 41)
    old_interp = np.interp(np.log(grid), np.log(old_p), old_t)
    new_interp = np.interp(np.log(grid), np.log(new_p), new_t)
    results.extend(_compare_numeric(
        case, group, f"{name}.temperature-on-common-pressure",
        old_interp, new_interp, _TEMPERATURE))
    return results


def _xy_path(value) -> np.ndarray:
    array = np.asarray(ma.asarray(value, dtype=float).filled(np.nan), dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    xy = array[:, :2]
    return xy[np.isfinite(xy).all(axis=1)]


def _resample_xy_path(path: np.ndarray, count: int = 41) -> np.ndarray:
    if len(path) < 2:
        return np.empty((0, 2), dtype=float)
    distance = np.concatenate(([0.0], np.cumsum(np.hypot(
        np.diff(path[:, 0]), np.diff(path[:, 1])))))
    if distance[-1] <= 0.0:
        parameter = np.linspace(0.0, 1.0, len(path))
    else:
        parameter = distance / distance[-1]
    target = np.linspace(0.0, 1.0, count)
    return np.column_stack((
        np.interp(target, parameter, path[:, 0]),
        np.interp(target, parameter, path[:, 1]),
    ))


def _compare_slinky_path(case: Case, legacy, native) -> list[Comparison]:
    old = _resample_xy_path(_xy_path(getattr(legacy, "slinky_traj", None)))
    new = _resample_xy_path(_xy_path(getattr(native, "slinky_traj", None)))
    if len(old) == 0 and len(new) == 0:
        return [Comparison(
            case.name, case.source, "trajectory", "slinky_traj.xy",
            "both-missing")]
    if len(old) == 0 or len(new) == 0:
        return [Comparison(
            case.name, case.source, "trajectory", "slinky_traj.xy",
            "missing-mismatch", len(old), len(new))]
    return _compare_numeric(
        case,
        "trajectory",
        "slinky_traj.xy-normalized-path",
        old,
        new,
        _HEIGHT,
    )


def _deterministic_watch_oracle(legacy, native):
    """Re-run legacy watch logic with deterministic native MMP.

    Legacy SHARPpy's MMP routine can read uninitialized ``np.empty`` cells,
    making both MMP and its ``>= 0.6`` watch branch process-memory dependent.
    All other watch ingredients remain the tolerance-gated legacy values.
    """

    raw = {
        "right_watch_type": getattr(legacy, "right_watch_type", None),
        "left_watch_type": getattr(legacy, "left_watch_type", None),
    }
    replacement = _scalar(getattr(native, "mmp", None))
    if replacement is None:
        return raw, {}

    from sharppy.sharptab import watch_type

    original = getattr(legacy, "mmp", None)
    try:
        legacy.mmp = replacement
        normalized = {
            "right_watch_type": watch_type.possible_watch(
                legacy, use_left=False)[0],
            "left_watch_type": watch_type.possible_watch(
                legacy, use_left=True)[0],
        }
    except Exception:
        normalized = raw
    finally:
        legacy.mmp = original
    changed = {
        field: (raw[field], normalized[field])
        for field in raw if raw[field] != normalized[field]
    }
    return normalized, changed


def compare_case(case: Case) -> list[Comparison]:
    """Return all public-output comparisons for one sounding case."""

    from sharppy.sharptab import profile as legacy_profile
    from sharpmod.sharptab.native_profile import NativeConvectiveProfile

    try:
        legacy = legacy_profile.ConvectiveProfile(**_copy_kwargs(case.kwargs))
    except Exception as exc:
        return [Comparison(
            case.name, case.source, "backend", "legacy-construction",
            "backend-error", detail=f"{type(exc).__name__}: {exc}",
        )]
    try:
        native = NativeConvectiveProfile(**_copy_kwargs(case.kwargs))
    except Exception as exc:
        return [Comparison(
            case.name, case.source, "backend", "native-construction",
            "backend-error", detail=f"{type(exc).__name__}: {exc}",
        )]

    backend = getattr(native, "_sharpmod_calculation_backend", None)
    results = [Comparison(
        case.name,
        case.source,
        "backend",
        "native-backend",
        "equal" if backend == "sharppyrs/sharprs" else "backend-error",
        "sharppyrs/sharprs",
        backend,
    )]
    native_payload = getattr(native, "_sharpmod_native_analysis", {}) or {}
    actual_derived = set((native_payload.get("derived") or {}).keys())
    expected_derived = set(DERIVED_FIELD_COVERAGE)
    missing_derived = sorted(expected_derived - actual_derived)
    unexpected_derived = sorted(actual_derived - expected_derived)
    results.append(Comparison(
        case.name,
        case.source,
        "coverage",
        "84-field-derived-schema",
        "equal" if not missing_derived and not unexpected_derived
        else "schema-mismatch",
        len(expected_derived),
        len(actual_derived),
        detail=(
            f"missing={missing_derived}, unexpected={unexpected_derived}"
            if missing_derived or unexpected_derived else None
        ),
    ))

    for spec in SCALAR_FIELDS + VECTOR_FIELDS:
        results.extend(_compare_numeric(
            case,
            spec.group,
            spec.name,
            getattr(legacy, spec.name, None),
            getattr(native, spec.name, None),
            spec.tolerance,
        ))

    for field in POLAR_WIND_FIELDS:
        results.extend(_compare_numeric(
            case,
            "kinematics-polar-as-components",
            field,
            _polar_components(getattr(legacy, field, None)),
            _polar_components(getattr(native, field, None)),
            _WIND,
        ))

    # SHARPpy hard-wires its public aggregate fields to the right mover.  The
    # native adapter deliberately selects the left mover south of the equator.
    # Compare against the appropriate side-specific legacy ingredient rather
    # than treating corrected Southern Hemisphere selection as a mismatch.
    side = "left" if float(case.kwargs.get("latitude", 0.0)) < 0.0 else "right"
    for public_name, side_suffix in (
        ("stp_cin", "stp_cin"),
        ("stp_fixed", "stp_fixed"),
        ("scp", "scp"),
    ):
        results.extend(_compare_numeric(
            case,
            "severe-selected-hemisphere",
            public_name,
            getattr(legacy, f"{side}_{side_suffix}", None),
            getattr(native, public_name, None),
            _INDEX,
        ))

    for spec in INFORMATIONAL_FIELDS:
        expected = _scalar(getattr(legacy, spec.name, None))
        actual = _scalar(getattr(native, spec.name, None))
        error = (
            abs(actual - expected)
            if expected is not None and actual is not None else None
        )
        results.append(Comparison(
            case.name,
            case.source,
            spec.group,
            spec.name,
            "informational",
            expected,
            actual,
            error,
            (
                error / max(abs(expected), 1.0e-12)
                if error is not None else None
            ),
            detail=(
                "legacy MMP reads uninitialized np.empty cells; native result "
                "is initialized and deterministic"
            ),
        ))

    from sharppy.sharptab import params as legacy_params
    results.extend(_compare_numeric(
        case, "indices", "sweat",
        _safe_oracle(lambda: legacy_params.sweat(legacy)),
        getattr(native, "sweat", None), Tolerance(1.0)))
    results.extend(_compare_numeric(
        case, "thermodynamics", "thetae_diff",
        _safe_oracle(lambda: legacy_params.thetae_diff(legacy)),
        getattr(native, "thetae_diff", None), Tolerance(0.5)))

    oracles = _extended_python_oracles(legacy)
    companion = _python_companion(legacy)
    for spec in COMPANION_FIELDS:
        results.extend(_compare_numeric(
            case,
            spec.group,
            spec.name,
            oracles[spec.name],
            getattr(native, spec.name, None),
            spec.tolerance,
        ))

    # Same local Python oracle, with public-name seams made explicit.
    results.extend(_compare_numeric(
        case,
        "extended-derived",
        "sfc_500m_shear",
        oracles["sfc_500m_shear"],
        _vector_magnitude(getattr(native, "sfc_500m_shear", None)),
        _WIND,
    ))
    results.extend(_compare_numeric(
        case,
        "extended-derived",
        "cape_0_3km",
        oracles["cape_0_3km"],
        getattr(native, "cape_0_3km", None),
        _ENERGY,
    ))

    # Transparently inventory the one intentional behavior seam: the old
    # stripped companion hard-coded sfc=0 and lost sparse OAX wind context.
    # The native bridge now normalizes to the full physical profile, so its
    # authoritative comparison is the same Python formula on ``legacy`` above.
    companion_values = {
        spec.name: getattr(companion, spec.name, None)
        for spec in COMPANION_FIELDS
    }
    companion_values["sfc_500m_shear"] = getattr(
        companion, "shear_sfc_500m", None)
    from sharpmod.sharptab import params as local_params
    companion_values["cape_0_3km"] = _safe_oracle(
        lambda: local_params.layer_cape_agl(companion, 0.0, 3000.0))
    seam_specs = {
        spec.name: spec.tolerance for spec in COMPANION_FIELDS
    } | {"sfc_500m_shear": _WIND, "cape_0_3km": _ENERGY}
    for field, tolerance in seam_specs.items():
        checks = _compare_numeric(
            case, "legacy-companion-seam", field,
            oracles[field], companion_values[field], tolerance)
        if all(check.passed for check in checks):
            continue
        intentional = case.name.startswith("spc-oax-")
        results.append(Comparison(
            case.name,
            case.source,
            "legacy-companion-seam",
            f"pre-rust-companion.{field}",
            "informational",
            repr(oracles[field]),
            repr(companion_values[field]),
            detail=(
                "intentional OAX input-normalization correction: the old "
                "stripped companion lost the physical surface/wind context"
                if intentional else
                "pre-existing lightweight-companion input-context "
                "approximation differs from the full-profile SHARPpy formula"
            ),
        ))
    results.extend(_compare_numeric(
        case,
        "parcel-mupcl",
        "mupcl.brnshear",
        getattr(getattr(legacy, "mupcl", None), "brnshear", None),
        getattr(getattr(native, "mupcl", None), "brnshear", None),
        _ENERGY,
    ))

    corfidi = getattr(legacy, "upshear_downshear", ())
    results.extend(_compare_numeric(
        case, "kinematics", "corfidi_up", corfidi[:2],
        getattr(native, "corfidi_up", None), _WIND))
    results.extend(_compare_numeric(
        case, "kinematics", "corfidi_dn", corfidi[2:4],
        getattr(native, "corfidi_dn", None), _WIND))

    legacy_advection = getattr(legacy, "inf_temp_adv", (None, None))
    native_advection = getattr(native, "inf_temp_adv", (None, None))
    results.extend(_compare_numeric(
        case, "temperature-advection", "temp_adv",
        legacy_advection[0], native_advection[0], Tolerance(0.25)))
    results.extend(_compare_numeric(
        case, "temperature-advection", "temp_adv_bounds",
        legacy_advection[1], native_advection[1], _PRESSURE))

    results.extend(_compare_slinky_path(case, legacy, native))

    old_max_lapse = _numeric_values(getattr(legacy, "max_lapse_rate_2_6", None))
    new_max_lapse = _numeric_values(getattr(native, "max_lapse_rate_2_6", None))
    for index, tolerance in enumerate((Tolerance(0.10), _PRESSURE, _PRESSURE)):
        old_value = old_max_lapse[index] if index < len(old_max_lapse) else None
        new_value = new_max_lapse[index] if index < len(new_max_lapse) else None
        results.extend(_compare_numeric(
            case, "profile", f"max_lapse_rate_2_6[{index}]",
            old_value, new_value, tolerance))
    results.extend(_compare_pressure_temperature_trace(
        case,
        "profile",
        "dpcl_trace",
        getattr(legacy, "dpcl_ptrace", None),
        getattr(legacy, "dpcl_ttrace", None),
        getattr(native, "dpcl_ptrace", None),
        getattr(native, "dpcl_ttrace", None),
    ))
    old_sfc = int(getattr(legacy, "sfc", 0))
    new_sfc = int(getattr(native, "sfc", 0))
    old_top = int(getattr(legacy, "top", -1))
    new_top = int(getattr(native, "top", -1))
    indices_equal = (old_sfc, old_top) == (new_sfc, new_top)
    known_oax_normalization = _is_leading_presurface_normalization(
        legacy, native)
    results.append(Comparison(
        case.name,
        case.source,
        "profile",
        "normalized-surface/top-index",
        "equal" if indices_equal else (
            "normalized-sampling" if known_oax_normalization
            else "coverage-mismatch"
        ),
        (old_sfc, old_top),
        (new_sfc, new_top),
        detail=(
            "known OAX normalization: indices shift when its all-missing "
            "first SPC row is removed"
            if not indices_equal and known_oax_normalization else None
        ),
    ))
    for label, old_index, new_index in (
        ("surface-pressure", old_sfc, new_sfc),
        ("top-pressure", old_top, new_top),
    ):
        results.extend(_compare_numeric(
            case,
            "profile",
            label,
            ma.asarray(legacy.pres)[old_index],
            ma.asarray(native.pres)[new_index],
            _PRESSURE,
        ))

    for parcel_name in ("sfcpcl", "fcstpcl", "mupcl", "mlpcl", "effpcl"):
        old_parcel = getattr(legacy, parcel_name, None)
        new_parcel = getattr(native, parcel_name, None)
        for field, tolerance in PARCEL_FIELDS.items():
            results.extend(_compare_numeric(
                case,
                f"parcel-{parcel_name}",
                f"{parcel_name}.{field}",
                getattr(old_parcel, field, None),
                getattr(new_parcel, field, None),
                tolerance,
            ))
        results.extend(_compare_pressure_temperature_trace(
            case,
            f"parcel-{parcel_name}",
            f"{parcel_name}.trace",
            getattr(old_parcel, "ptrace", None),
            getattr(old_parcel, "ttrace", None),
            getattr(new_parcel, "ptrace", None),
            getattr(new_parcel, "ttrace", None),
        ))

    for field, tolerance in ENVIRONMENT_ARRAYS.items():
        results.extend(_compare_environment_on_pressure(
            case, legacy, native, field, tolerance))

    for field, tolerance in FIRE_FIELDS.items():
        results.extend(_compare_numeric(
            case, "fire", field,
            getattr(legacy, field, None), getattr(native, field, None), tolerance))
    for field, tolerance in WINTER_FIELDS.items():
        results.extend(_compare_numeric(
            case, "winter", field,
            getattr(legacy, field, None), getattr(native, field, None), tolerance))

    old_haines = _scalar(getattr(legacy, "haines_hght", None))
    new_haines = _scalar(getattr(native, "haines_hght", None))
    results.append(Comparison(
        case.name,
        case.source,
        "fire",
        "haines_hght",
        "equal" if old_haines == new_haines else "categorical-mismatch",
        old_haines,
        new_haines,
    ))

    watch_oracle, watch_seams = _deterministic_watch_oracle(legacy, native)
    for field, (raw_watch, normalized_watch) in watch_seams.items():
        results.append(Comparison(
            case.name,
            case.source,
            "undefined-legacy-mmp-seam",
            f"{field}.mmp-normalized-oracle",
            "informational",
            str(raw_watch),
            str(normalized_watch),
            detail=(
                "legacy MMP uses uninitialized np.empty cells; watch oracle "
                "re-evaluated with deterministic Rust MMP"
            ),
        ))

    for field in CATEGORICAL_FIELDS:
        expected = watch_oracle.get(field, getattr(legacy, field, None))
        actual = getattr(native, field, None)
        results.append(Comparison(
            case.name,
            case.source,
            "categorical",
            field,
            "equal" if expected == actual else "categorical-mismatch",
            str(expected),
            str(actual),
        ))
    expected_watch = watch_oracle.get(
        f"{side}_watch_type", getattr(legacy, f"{side}_watch_type", None))
    actual_watch = getattr(native, "watch_type", None)
    results.append(Comparison(
        case.name,
        case.source,
        "categorical-selected-hemisphere",
        "watch_type",
        "equal" if expected_watch == actual_watch else "categorical-mismatch",
        str(expected_watch),
        str(actual_watch),
    ))
    return results


def audit(cases: list[Case]) -> tuple[list[Comparison], dict]:
    """Run the audit and return element-level evidence plus summary counts."""

    _numpy_legacy_aliases()
    warnings.filterwarnings(
        "ignore", message="Input line .* contained no data", category=UserWarning)
    comparisons = [result for case in cases for result in compare_case(case)]
    failures = [result for result in comparisons if not result.passed]
    summary = {
        "cases": len(cases),
        "comparisons": len(comparisons),
        "passed": len(comparisons) - len(failures),
        "failed": len(failures),
        "numeric_outside_tolerance": sum(
            result.status == "outside-tolerance" for result in failures),
        "missing_state_mismatches": sum(
            result.status == "missing-mismatch" for result in failures),
        "categorical_mismatches": sum(
            result.status == "categorical-mismatch" for result in failures),
        "backend_or_shape_errors": sum(
            result.status in {
                "backend-error", "shape-mismatch", "schema-mismatch",
                "coverage-mismatch",
            }
            for result in failures),
        "informational_non_gating": sum(
            result.status == "informational" for result in comparisons),
        "legacy_companion_seams": sum(
            result.status == "informational"
            and result.group == "legacy-companion-seam"
            for result in comparisons),
        "upper_stratosphere_exclusions": sum(
            result.status == "informational"
            and result.field.endswith(
                ".upper-stratosphere-outside-analysis-domain")
            for result in comparisons),
        "mmp_informationals": sum(
            result.status == "informational" and result.field == "mmp"
            for result in comparisons),
        "mmp_watch_oracle_normalizations": sum(
            result.status == "informational"
            and result.group == "undefined-legacy-mmp-seam"
            for result in comparisons),
        "rust_owned_derived_fields": len(DERIVED_FIELD_COVERAGE),
        "tolerance_gated_derived_fields": (
            len(DERIVED_FIELD_COVERAGE)
            - len(EXTERNAL_ORACLE_FIELDS)
            - len(INFORMATIONAL_DERIVED_FIELDS)
        ),
        "dedicated_external_oracle_fields": len(EXTERNAL_ORACLE_FIELDS),
        "intentional_non_gating_derived_fields": len(
            INFORMATIONAL_DERIVED_FIELDS),
    }
    return comparisons, summary


def _failure_sort_key(result: Comparison):
    if result.absolute_error is None or result.allowed_error in {None, 0.0}:
        return (math.inf, result.case, result.field)
    return (
        result.absolute_error / result.allowed_error,
        result.case,
        result.field,
    )


def _print_report(comparisons: list[Comparison], summary: dict, maximum: int) -> None:
    from sharpmod.sharptab import native_analysis

    info = native_analysis.backend_info()
    print("SHARPpy Reimagined vRust native parity audit")
    print(
        "backend revisions: "
        f"sharppyrs={info.get('sharppyrs_revision')} "
        f"sharprs={info.get('sharprs_revision')} "
        f"ecape-rs={info.get('ecape_rs_revision')}"
    )
    print(
        f"cases={summary['cases']} comparisons={summary['comparisons']} "
        f"passed={summary['passed']} failed={summary['failed']}"
    )
    print(
        "failure classes: "
        f"numeric={summary['numeric_outside_tolerance']} "
        f"missing-state={summary['missing_state_mismatches']} "
        f"categorical={summary['categorical_mismatches']} "
        f"backend/shape={summary['backend_or_shape_errors']}"
    )
    print(
        f"informational non-gating comparisons="
        f"{summary['informational_non_gating']} "
        f"(MMP={summary['mmp_informationals']}, "
        f"pre-existing companion seams={summary['legacy_companion_seams']}, "
        f"upper-stratosphere exclusions="
        f"{summary['upper_stratosphere_exclusions']})"
    )
    print(
        "Rust-derived field coverage: "
        f"total={summary['rust_owned_derived_fields']} "
        f"tolerance-gated={summary['tolerance_gated_derived_fields']} "
        f"dedicated-external-oracle={summary['dedicated_external_oracle_fields']} "
        f"intentional-non-gating={summary['intentional_non_gating_derived_fields']}"
    )

    failures = sorted(
        (result for result in comparisons if not result.passed),
        key=_failure_sort_key,
        reverse=True,
    )
    if not failures:
        print("RESULT: PASS")
        return
    print(f"worst failures (showing {min(maximum, len(failures))}):")
    for result in failures[:maximum]:
        if result.absolute_error is not None:
            detail = (
                f"legacy={result.legacy:.8g} native={result.native:.8g} "
                f"abs={result.absolute_error:.5g} allowed={result.allowed_error:.5g}"
            )
        else:
            detail = (
                result.detail
                or f"legacy={result.legacy!r} native={result.native!r}"
            )
        print(
            f"  {result.case} | {result.field} | {result.status} | {detail}"
        )
    print("RESULT: FAIL")


def _json_payload(comparisons: list[Comparison], summary: dict) -> dict:
    from sharpmod.sharptab import native_analysis

    failures = [asdict(result) for result in comparisons if not result.passed]
    informational = [
        asdict(result) for result in comparisons
        if result.status in {"informational", "normalized-sampling"}
    ]
    return {
        "schema": "sharpmod.native-parity.v1",
        "backend": native_analysis.backend_info(),
        "derived_field_coverage": DERIVED_FIELD_COVERAGE,
        "external_oracles": EXTERNAL_ORACLE_FIELDS,
        "intentional_non_gating": INFORMATIONAL_DERIVED_FIELDS,
        "tolerance_rule": (
            "abs(error) <= max(field absolute tolerance, "
            "field relative tolerance * abs(legacy))"
        ),
        "summary": summary,
        "failures": failures,
        "informational": informational,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Audit eight representative real cases instead of the full corpus.",
    )
    parser.add_argument(
        "--no-synthetic", action="store_true", help="Exclude synthetic regimes."
    )
    parser.add_argument(
        "--json", type=Path, help="Write machine-readable results to this path."
    )
    parser.add_argument(
        "--max-failures", type=int, default=30, help="Maximum failures to print."
    )
    args = parser.parse_args(argv)

    cases = build_corpus(synthetic=not args.no_synthetic)
    if args.quick:
        cases = [case for case in cases if case.name in QUICK_CASE_NAMES]

    comparisons, summary = audit(cases)
    _print_report(comparisons, summary, max(1, args.max_failures))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(_json_payload(comparisons, summary), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json.resolve()}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
