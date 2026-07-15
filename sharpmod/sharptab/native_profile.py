"""SHARPpy-compatible convective profile backed by one native Rust analysis.

The Qt widgets continue to receive the exact object shape they expect. Only
the analysis implementation changes: every field covered by sharppyrs/sharprs
is populated from one cached native result, while Python is retained for the
few database/climatology/precipitation-source features without a Rust input
contract.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import numpy.ma as ma

from sharppy.sharptab import constants as sp_constants
from sharppy.sharptab import interp as sp_interp
from sharppy.sharptab import params as sp_params
from sharppy.sharptab import profile as sp_profile
from sharppy.sharptab import thermo as sp_thermo
from sharppy.sharptab import utils as sp_utils
from sharppy.sharptab import watch_type as sp_watch_type
from sharppy.sharptab import winds as sp_winds

from . import native_analysis, sars_cache


_LOGGER = logging.getLogger(__name__)


def _masked_scalar(value):
    if value is None or value is ma.masked:
        return ma.masked
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return value if np.isfinite(value) else ma.masked


def _tuple(value):
    if value is None:
        return (ma.masked, ma.masked)
    converted = tuple(_masked_scalar(item) for item in value)
    return converted


def _array(value):
    array = ma.masked_invalid(np.asarray(value, dtype=float))
    array.set_fill_value(sp_constants.MISSING)
    return array


def _parcel(values):
    parcel = sp_params.Parcel()
    for name, value in values.items():
        if name in {"ptrace", "ttrace"}:
            setattr(parcel, name, _array(value))
        else:
            setattr(parcel, name, _masked_scalar(value))
    parcel.lplvals = SimpleNamespace(
        pres=parcel.pres, tmpc=parcel.tmpc, dwpc=parcel.dwpc)
    return parcel


class NativeConvectiveProfile(sp_profile.ConvectiveProfile):
    """Drop-in ``ConvectiveProfile`` with native-first calculations."""

    def __init__(self, **kwargs):
        # BasicProfile establishes SHARPpy's validated/masked object contract.
        # The native result immediately replaces its calculated profile arrays.
        sp_profile.BasicProfile.__init__(self, **kwargs)
        self.user_srwind = None
        native = native_analysis.try_analyze_profile(self)
        if native is None:
            _LOGGER.warning("Falling back to the complete Python ConvectiveProfile")
            sp_profile.ConvectiveProfile.__init__(self, **kwargs)
            self._sharpmod_calculation_backend = "python-fallback"
            self._sharpmod_native_analysis = None
            self._sharpmod_python_fallbacks = ("all",)
            return

        self._sharpmod_native_analysis = native
        self._sharpmod_calculation_backend = "sharppyrs/sharprs"
        self._sharpmod_python_fallbacks = (
            "fire-pbl-details",
            "precipitation-source/layer-energies",
            "SARS-analog-databases",
            "PWV-station-climatology",
        )
        self._apply_native(native, reset_bunkers=True)
        self._run_python_only_features()
        _LOGGER.info(
            "Sounding analysis backend=%s fallbacks=%s",
            self._sharpmod_calculation_backend,
            ",".join(self._sharpmod_python_fallbacks),
        )

    def _run_python_only_features(self):
        # Calculate only the fields that the pinned Rust API does not supply.
        # The vendored methods also recompute native fire, winter, SHIP, and
        # parcel values, which are immediately overwritten and cost tens of
        # milliseconds on every cached map click.
        self._run_python_fire_details()
        self._run_python_precip_details()
        self._apply_native_precip_type()
        self._run_python_sars_matches()

        try:
            sp_profile.ConvectiveProfile.get_PWV_loc(self)
        except Exception as exc:  # legacy station table/NumPy compatibility
            _LOGGER.warning("PWV climatology unavailable: %s", exc)
            self.pwv_flag = 0
        self._apply_native_watch()

    def _run_python_fire_details(self):
        """Populate the PBL fields not yet exposed by sharprs."""
        self.ppbl_top = sp_params.pbl_top(self)
        self.sfc_rh = sp_thermo.relh(
            self.pres[self.sfc], self.tmpc[self.sfc], self.dwpc[self.sfc])
        pres_sfc = self.pres[self.sfc]
        pres_1km = sp_interp.pres(self, sp_interp.to_msl(self, 1000.0))
        self.pbl_h = sp_interp.to_agl(
            self, sp_interp.hght(self, self.ppbl_top))
        self.rh01km = sp_params.mean_relh(
            self, pbot=pres_sfc, ptop=pres_1km)
        self.pblrh = sp_params.mean_relh(
            self, pbot=pres_sfc, ptop=self.ppbl_top)
        self.meanwind01km = sp_winds.mean_wind(
            self, pbot=pres_sfc, ptop=pres_1km)
        self.meanwindpbl = sp_winds.mean_wind(
            self, pbot=pres_sfc, ptop=self.ppbl_top)
        self.pblmaxwind = sp_winds.max_wind(
            self, lower=0, upper=self.pbl_h)

    def _run_python_precip_details(self):
        """Populate precip-source and warm/cold-layer fields absent in Rust."""
        # Preserve the established mean-omega result by using the vendored DGZ
        # bounds; all displayed DGZ thermodynamic values remain native.
        dgz_pbot, dgz_ptop = sp_params.dgz(self)
        self.dgz_meanomeg = sp_params.mean_omega(
            self, pbot=dgz_pbot, ptop=dgz_ptop) * 10
        self.oprh = self.dgz_meanomeg * self.dgz_pw * (
            self.dgz_meanrh / 100.0)

        self.plevel, self.phase, self.tmp, self.st = \
            sp_watch_type.init_phase(self)
        self.tpos, self.tneg, self.ttop, self.tbot = \
            sp_watch_type.posneg_temperature(self, start=self.plevel)
        self.wpos, self.wneg, self.wtop, self.wbot = \
            sp_watch_type.posneg_wetbulb(self, start=self.plevel)
        self.precip_type = sp_watch_type.best_guess_precip(
            self, self.phase, self.plevel, self.tmp, self.tpos, self.tneg)

    def _run_python_sars_matches(self):
        """Populate analog matches while reusing parsed immutable databases."""
        sfc_6km_shear = sp_utils.KTS2MS(sp_utils.mag(
            self.sfc_6km_shear[0], self.sfc_6km_shear[1]))
        sfc_3km_shear = sp_utils.KTS2MS(sp_utils.mag(
            self.sfc_3km_shear[0], self.sfc_3km_shear[1]))
        sfc_9km_shear = sp_utils.KTS2MS(sp_utils.mag(
            self.sfc_9km_shear[0], self.sfc_9km_shear[1]))
        h500t = sp_interp.temp(self, 500.0)
        lapse_rate = sp_params.lapse_rate(self, 700.0, 500.0, pres=True)
        mumr = sp_thermo.mixratio(self.mupcl.pres, self.mupcl.dwpc)

        self.hail_database = "sars_hail.txt"
        self.supercell_database = "sars_supercell.txt"
        hail_args = (
            mumr, self.mupcl.bplus, h500t, lapse_rate, sfc_6km_shear,
            sfc_9km_shear, sfc_3km_shear)
        supercell_args = (
            self.mlpcl.bplus, self.mlpcl.lclhght, h500t, lapse_rate,
            sp_utils.MS2KTS(sfc_6km_shear))

        try:
            self.right_matches = sars_cache.hail(
                self.hail_database, *hail_args, self.right_srh3km[0])
        except Exception:
            self.right_matches = ([], [], 0, 0, 0)
        try:
            self.left_matches = sars_cache.hail(
                self.hail_database, *hail_args, -self.left_srh3km[0])
        except Exception:
            self.left_matches = ([], [], 0, 0, 0)

        common_supercell = (
            sp_utils.MS2KTS(sfc_3km_shear),
            sp_utils.MS2KTS(sfc_9km_shear),
        )
        try:
            self.right_supercell_matches = sars_cache.supercell(
                self.supercell_database, *supercell_args,
                self.right_srh1km[0], *common_supercell,
                self.right_srh3km[0])
        except Exception:
            self.right_supercell_matches = ([], [], 0, 0, 0)
        try:
            self.left_supercell_matches = sars_cache.supercell(
                self.supercell_database, *supercell_args,
                -self.left_srh1km[0], *common_supercell,
                -self.left_srh3km[0])
        except Exception:
            self.left_supercell_matches = ([], [], 0, 0, 0)

        if self.latitude < 0:
            self.supercell_matches = self.left_supercell_matches
            self.matches = self.left_matches
        else:
            self.supercell_matches = self.right_supercell_matches
            self.matches = self.right_matches

    def _apply_native(self, native, *, reset_bunkers=False):
        arrays = native["arrays"]
        for name in (
                "pres", "hght", "tmpc", "dwpc", "wdir", "wspd", "omeg",
                "u", "v", "logp", "vtmp", "theta", "thetae", "wvmr",
                "relh", "wetbulb"):
            setattr(self, name, _array(arrays[name]))

        profile_values = native["profile"]
        self.sfc = int(profile_values["sfc"])
        self.top = int(profile_values["top"])
        self.ebottom = _masked_scalar(profile_values["ebottom"])
        self.etop = _masked_scalar(profile_values["etop"])
        self.ebotm = _masked_scalar(profile_values["ebotm"])
        self.etopm = _masked_scalar(profile_values["etopm"])
        self.max_lapse_rate_2_6 = _tuple(
            profile_values["max_lapse_rate_2_6"])

        parcels = native["parcels"]
        self.sfcpcl = _parcel(parcels["surface"])
        self.fcstpcl = _parcel(parcels["forecast"])
        self.mupcl = _parcel(parcels["most_unstable"])
        self.mlpcl = _parcel(parcels["mixed_layer"])
        self.effpcl = _parcel(parcels["effective"])
        self.usrpcl = sp_params.Parcel()

        derived = native["derived"]
        for name, value in derived.items():
            if isinstance(value, (list, tuple)):
                setattr(self, name, tuple(value))
            else:
                setattr(self, name, _masked_scalar(value))
        self.mupcl.brnshear = _masked_scalar(derived["brnshear"])
        if self.mupcl.brnshear is not ma.masked and self.mupcl.brnshear > 0:
            self.mupcl.brn = self.mupcl.bplus / self.mupcl.brnshear

        self.convT = _masked_scalar(derived["conv_t_f"])
        self.maxT = _masked_scalar(derived["max_t_f"])
        self.inf_temp_adv = (
            _array(derived["temp_adv"]),
            _array(derived["temp_adv_bounds"]),
        )

        self.dcape = _masked_scalar(derived["dcape"])
        self.dpcl_ttrace = _array(profile_values["dpcl_ttrace"])
        self.dpcl_ptrace = _array(profile_values["dpcl_ptrace"])
        self.drush = _masked_scalar(derived["drush_f"])
        self.slinky_traj = _array(derived["slinky_traj"])
        self.updraft_tilt = _masked_scalar(derived["slinky_tilt"])

        self._apply_native_kinematics(native, reset_bunkers=reset_bunkers)
        self._apply_native_severe(native)
        self._apply_native_fire(native)
        self._apply_native_winter(native)

        # The local IndexBoard reads this cached companion; force it to be
        # rebuilt from the new native result after edits/storm-motion changes.
        self.__dict__.pop("_sharpmod_derived_profile", None)

    def _apply_native_kinematics(self, native, *, reset_bunkers=False):
        derived = native["derived"]
        extra = native["extra"]
        motion = _tuple(native["profile"]["srwind"])
        if reset_bunkers or not hasattr(self, "bunkers"):
            self.bunkers = motion
        self.user_srwind = motion
        self.srwind = motion

        for name in (
                "wind1km", "wind6km", "sfc_500m_shear", "sfc_1km_shear",
                "sfc_3km_shear", "sfc_6km_shear", "sfc_8km_shear",
                "lcl_el_shear", "eff_shear", "ebwd", "mean_wind_sfc_500m",
                "mean_1km", "mean_3km", "mean_6km", "mean_8km", "mean_eff",
                "mean_ebw", "mean_lcl_el"):
            setattr(self, name, _tuple(derived[name]))
        self.sfc_9km_shear = _tuple(extra["sfc_9km_shear"])
        self.ebwspd = _masked_scalar(extra["ebwspd"])

        self.right_esrh = _tuple(extra["right_esrh"])
        self.left_esrh = _tuple(extra["left_esrh"])
        self.right_srh1km = _tuple(extra["right_srh1km"])
        self.right_srh3km = _tuple(extra["right_srh3km"])
        self.left_srh1km = _tuple(extra["left_srh1km"])
        self.left_srh3km = _tuple(extra["left_srh3km"])
        self.right_critical_angle = _masked_scalar(
            extra["right_critical_angle"])
        self.left_critical_angle = _masked_scalar(extra["left_critical_angle"])

        for side in ("right", "left"):
            for layer in ("1km", "3km", "6km", "8km"):
                setattr(self, f"{side}_srw_{layer}",
                        _tuple(extra[f"{side}_srw_{layer}"]))
            for layer in ("0_2km", "4_6km", "9_11km", "eff", "ebw"):
                setattr(self, f"{side}_srw_{layer}",
                        _tuple(extra[f"{side}_srw_{layer}"]))
            for layer in ("4_5km", "lcl_el"):
                comp = _tuple(extra[f"{side}_srw_{layer}_comp"])
                setattr(self, f"{side}_srw_{layer}",
                        _tuple(sp_utils.comp2vec(*comp)))

        self.upshear_downshear = _tuple(extra["upshear_downshear"])
        if float(getattr(self, "latitude", 0.0) or 0.0) < 0:
            prefix = "left"
        else:
            prefix = "right"
        self.srw_eff = getattr(self, f"{prefix}_srw_eff")
        self.srw_ebw = getattr(self, f"{prefix}_srw_ebw")
        self.esrh = getattr(self, f"{prefix}_esrh")
        self.critical_angle = getattr(self, f"{prefix}_critical_angle")
        for layer in (
                "1km", "3km", "6km", "8km", "4_5km", "lcl_el",
                "0_2km", "4_6km", "9_11km"):
            setattr(self, f"srw_{layer}", getattr(self, f"{prefix}_srw_{layer}"))
        self.srh1km = getattr(self, f"{prefix}_srh1km")
        self.srh3km = getattr(self, f"{prefix}_srh3km")

    def _apply_native_severe(self, native):
        derived = native["derived"]
        extra = native["extra"]
        for name in (
                "right_stp_fixed", "left_stp_fixed", "right_stp_cin",
                "left_stp_cin", "sherbe"):
            setattr(self, name, _masked_scalar(extra[name]))
        self.right_scp = _masked_scalar(derived["right_scp"])
        self.left_scp = _masked_scalar(derived["left_scp"])
        if float(getattr(self, "latitude", 0.0) or 0.0) < 0:
            self.stp_fixed = self.left_stp_fixed
            self.stp_cin = self.left_stp_cin
            self.scp = self.left_scp
        else:
            self.stp_fixed = self.right_stp_fixed
            self.stp_cin = self.right_stp_cin
            self.scp = self.right_scp

    def _apply_native_fire(self, native):
        fire = native["extra"]["fire"]
        self.fosberg = _masked_scalar(fire["fosberg"])
        self.haines_hght = {
            "Low": sp_constants.HAINES_LOW,
            "Mid": sp_constants.HAINES_MID,
            "High": sp_constants.HAINES_HIGH,
        }.get(fire["haines_hght"], ma.masked)
        for name in ("haines_low", "haines_mid", "haines_high", "bplus_fire"):
            setattr(self, name, _masked_scalar(fire[name]))

    def _apply_native_winter(self, native):
        winter = native["extra"]["winter"]
        for name in ("dgz_pbot", "dgz_ptop", "dgz_meanrh", "dgz_pw", "dgz_meanq"):
            setattr(self, name, _masked_scalar(winter[name]))
        try:
            self.oprh = self.dgz_meanomeg * self.dgz_pw * (self.dgz_meanrh / 100.0)
        except (AttributeError, TypeError, ValueError):
            self.oprh = ma.masked

    def _apply_native_precip_type(self):
        try:
            phase = int(self.phase)
            if phase < 0:
                init_temp = 0.0
                init_level = 0.0
            else:
                init_temp = float(self.tmp)
                init_level = float(sp_interp.to_agl(
                    self, sp_interp.hght(self, self.plevel)))
            self.precip_type = native_analysis.best_guess_precip(
                phase,
                init_temp,
                init_level,
                float(self.tpos),
                float(self.tneg),
                float(self.tmpc[self.sfc]),
            )
        except (RuntimeError, TypeError, ValueError):
            # The Python source/layer solver already produced a valid fallback.
            pass

    def _watch_inputs(self, side):
        sign = -1.0 if float(getattr(self, "latitude", 0.0) or 0.0) < 0 else 1.0
        srw = getattr(self, f"{side}_srw_4_6km")
        esrh = getattr(self, f"{side}_esrh")[0] * sign
        srh1km = getattr(self, f"{side}_srh1km")[0] * sign
        upshear = self.upshear_downshear
        return {
            "stp_eff": float(getattr(self, f"{side}_stp_cin")) * sign,
            "stp_fixed": float(getattr(self, f"{side}_stp_fixed")) * sign,
            "srw_4_6km": float(np.hypot(*srw)),
            "esrh": float(esrh),
            "srh1km": float(srh1km),
            "sfc_8km_shear": float(np.hypot(*self.sfc_8km_shear)),
            "lr1": float(self.lapserate_sfc_1km),
            "sfcpcl_lclhght": float(self.sfcpcl.lclhght),
            "mlpcl_lclhght": float(self.mlpcl.lclhght),
            "mlpcl_bminus": float(self.mlpcl.bminus),
            "mupcl_bminus": float(self.mupcl.bminus),
            "ebotm": float(self.ebotm),
            "scp": float(getattr(self, f"{side}_scp")),
            "ship": float(self.ship),
            "sig_severe": float(self.sig_severe),
            "mmp": float(self.mmp),
            "wndg": float(self.wndg),
            "dcape": float(self.dcape),
            "pwat": float(self.pwat),
            "pwv_flag": int(self.pwv_flag),
            "low_rh": float(self.low_rh),
            "mid_rh": float(self.mid_rh),
            "upshear_wspd": float(np.hypot(upshear[0], upshear[1])),
            "sfc_tmpc": float(self.tmpc[self.sfc]),
            "sfc_dwpc": float(self.dwpc[self.sfc]),
            "sfc_pres": float(self.pres[self.sfc]),
            "sfc_wspd_kts": float(self.wspd[self.sfc]),
            "precip_type": str(self.precip_type),
        }

    def _apply_native_watch(self):
        try:
            self.right_watch_type = native_analysis.classify_watch(
                self._watch_inputs("right"))
            self.left_watch_type = native_analysis.classify_watch(
                self._watch_inputs("left"))
        except (RuntimeError, TypeError, ValueError):
            sp_profile.ConvectiveProfile.get_watch(self)
            return
        if float(getattr(self, "latitude", 0.0) or 0.0) < 0:
            self.watch_type = self.left_watch_type
        else:
            self.watch_type = self.right_watch_type

    def _reanalyze_for_storm_motion(self, storm_motion):
        native = native_analysis.try_analyze_profile(
            self, storm_motion=tuple(storm_motion))
        if native is None:
            # The native package became unavailable after construction. Keep
            # the established SHARPpy behavior as the explicit safety net.
            self.user_srwind = tuple(storm_motion)
            sp_profile.ConvectiveProfile.get_kinematics(self)
            sp_profile.ConvectiveProfile.get_severe(self)
            self._sharpmod_calculation_backend = "python-fallback-after-native"
            return
        self._sharpmod_native_analysis = native
        self._apply_native(native, reset_bunkers=False)

    def set_srright(self, rm_u, rm_v):
        current = tuple(self.user_srwind or self.bunkers)
        self._reanalyze_for_storm_motion((rm_u, rm_v, current[2], current[3]))

    def set_srleft(self, lm_u, lm_v):
        current = tuple(self.user_srwind or self.bunkers)
        self._reanalyze_for_storm_motion((current[0], current[1], lm_u, lm_v))

    def reset_srm(self):
        self._reanalyze_for_storm_motion(tuple(self.bunkers))


def target_profile_type():
    """Return the native adapter when packaged, otherwise legacy SHARPpy."""
    if native_analysis.available():
        return NativeConvectiveProfile
    return sp_profile.ConvectiveProfile


def configure_profile_collection(collection):
    """Set one existing ProfCollection to the best available target type."""
    collection._target_type = target_profile_type()
    return collection


_ORIGINAL_PARCELX = getattr(
    sp_params.parcelx, "_sharpmod_original_parcelx", sp_params.parcelx)


def _native_parcelx(prof, pbot=None, ptop=None, dp=-1, **kwargs):
    """Route interactive/user parcel lifts through the same sharprs solver."""
    if not isinstance(prof, NativeConvectiveProfile):
        return _ORIGINAL_PARCELX(
            prof, pbot=pbot, ptop=ptop, dp=dp, **kwargs)
    if getattr(prof, "_sharpmod_native_analysis", None) is None:
        # A NativeConvectiveProfile whose native construction failed is being
        # initialized by the complete legacy ConvectiveProfile path. Its
        # parcels do not exist yet, so every request must delegate until that
        # fallback initialization has finished.
        return _ORIGINAL_PARCELX(
            prof, pbot=pbot, ptop=ptop, dp=dp, **kwargs)

    flag = int(kwargs.get("flag", 5))
    cached = {
        1: "sfcpcl", 2: "fcstpcl", 3: "mupcl", 4: "mlpcl",
    }
    if (flag in cached and pbot is None and ptop is None
            and "lplvals" not in kwargs and "pres" not in kwargs):
        return getattr(prof, cached[flag])

    if flag == 5:
        level = kwargs.get("lplvals")
        pres = kwargs.get("pres", getattr(level, "pres", None))
        tmpc = kwargs.get("tmpc", getattr(level, "tmpc", None))
        dwpc = kwargs.get("dwpc", getattr(level, "dwpc", None))
        if pres is not None and tmpc is not None and dwpc is not None:
            try:
                result = _parcel(native_analysis.lift_user_parcel(
                    prof, pres, tmpc, dwpc, pbot=pbot, ptop=ptop))
                result._sharpmod_calculation_backend = "sharprs"
                return result
            except (native_analysis.NativeAnalysisUnavailable,
                    RuntimeError, TypeError, ValueError) as exc:
                _LOGGER.warning("Native user parcel lift unavailable: %s", exc)

    return _ORIGINAL_PARCELX(
        prof, pbot=pbot, ptop=ptop, dp=dp, **kwargs)


_native_parcelx._sharpmod_original_parcelx = _ORIGINAL_PARCELX
sp_params.parcelx = _native_parcelx
