//! One-call native sounding analysis for SHARPpy Reimagined vRust.
//!
//! The public `sharprs` Python module exposes a deliberately small legacy
//! surface.  This private extension instead calls the real `sharppyrs`
//! analysis view-model (and therefore the real `sharprs` core) once per
//! sounding, outside the GIL, and returns plain Python containers.

use ecape_rs::{CapeType, ParcelOptions, StormMotionType, calc_ecape_ncape};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use sharppyrs::sharprs::fire;
use sharppyrs::sharprs::params::cape::{
    self, LiftedParcelLevel, ParcelResult, ParcelType as SharprsParcelType, parcelx,
};
use sharppyrs::sharprs::params::{composites, indices};
use sharppyrs::sharprs::watch_type::{self, PrecipPhase, WatchParams};
use sharppyrs::sharprs::{thermo, winds};
use sharppyrs::{DerivedParams, Profile, SoundingData};

const KTS_TO_MS: f64 = 0.514_444_444_444_444_5;

#[derive(Clone, Copy, Debug)]
struct EcapeResult {
    ecape: f64,
    ncape: f64,
    cape: f64,
    lfc_m_msl: Option<f64>,
    el_m_msl: Option<f64>,
}

fn specific_humidity(pressure_pa: f64, dewpoint_c: f64) -> f64 {
    let vapor_pressure = 611.2 * ((17.67 * dewpoint_c) / (dewpoint_c + 243.5)).exp();
    0.62197 * vapor_pressure / (pressure_pa - 0.37803 * vapor_pressure)
}

fn analytic_mu_ecape(profile: &Profile) -> Option<EcapeResult> {
    let inner = &profile.inner;
    let mut height = Vec::new();
    let mut pressure = Vec::new();
    let mut temperature = Vec::new();
    let mut humidity = Vec::new();
    let mut u = Vec::new();
    let mut v = Vec::new();

    for index in 0..inner.pres.len() {
        let values = [
            inner.hght[index],
            inner.pres[index],
            inner.tmpc[index],
            inner.dwpc[index],
            inner.u[index],
            inner.v[index],
        ];
        if !values.iter().all(|value| value.is_finite()) {
            continue;
        }
        let pressure_pa = inner.pres[index] * 100.0;
        height.push(inner.hght[index]);
        pressure.push(pressure_pa);
        temperature.push(inner.tmpc[index] + 273.15);
        humidity.push(specific_humidity(pressure_pa, inner.dwpc[index]));
        u.push(inner.u[index] * KTS_TO_MS);
        v.push(inner.v[index] * KTS_TO_MS);
    }
    if pressure.len() < 3 {
        return None;
    }

    let options = ParcelOptions {
        cape_type: CapeType::MostUnstable,
        storm_motion_type: StormMotionType::RightMoving,
        pseudoadiabatic: Some(true),
        ..ParcelOptions::default()
    };
    let result = calc_ecape_ncape(
        &height,
        &pressure,
        &temperature,
        &humidity,
        &u,
        &v,
        &options,
    )
    .ok()?;
    let values = [result.ecape_jkg, result.ncape_jkg, result.cape_jkg];
    if !values.iter().all(|value| value.is_finite())
        || result.ecape_jkg < 0.0
        || result.cape_jkg < 0.0
        || result.ecape_jkg > result.cape_jkg + 1.0e-6
    {
        return None;
    }
    Some(EcapeResult {
        ecape: result.ecape_jkg,
        ncape: result.ncape_jkg,
        cape: result.cape_jkg,
        lfc_m_msl: result.lfc_m,
        el_m_msl: result.el_m,
    })
}

fn parcel_dict<'py>(py: Python<'py>, parcel: &ParcelResult) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    macro_rules! fields {
        ($($name:ident),+ $(,)?) => {
            $(result.set_item(stringify!($name), parcel.$name.clone())?;)+
        };
    }
    fields!(
        pres, tmpc, dwpc, bplus, bminus, lclpres, lclhght, lfcpres, lfchght, elpres, elhght,
        mplpres, mplhght, bfzl, b3km, b6km, li5, li3, limax, limaxpres, cap, cappres, bmin,
        bminpres, p0c, pm10c, pm20c, pm30c, hght0c, hghtm10c, hghtm20c, hghtm30c, wm10c, wm20c,
        wm30c, ptrace, ttrace,
    );
    Ok(result)
}

fn derived_dict<'py>(py: Python<'py>, derived: &DerivedParams) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    macro_rules! fields {
        ($($name:ident),+ $(,)?) => {
            $(result.set_item(stringify!($name), derived.$name.clone())?;)+
        };
    }
    fields!(
        pwat,
        k_idx,
        tei,
        esp,
        mmp,
        wndg,
        dcp,
        mburst,
        ship,
        right_scp,
        left_scp,
        stp_cin,
        stp_fixed,
        sweat,
        sig_severe,
        dcape,
        drush_f,
        mean_mixr,
        low_rh,
        mid_rh,
        totals_totals,
        conv_t_f,
        max_t_f,
        thetae_diff,
        lapserate_3km,
        lapserate_3_6km,
        lapserate_850_500,
        lapserate_700_500,
        lapserate_sfc_500m,
        lapserate_sfc_1km,
        srh500,
        srh1km,
        srh3km,
        right_esrh,
        sfc_500m_shear,
        sfc_1km_shear,
        sfc_3km_shear,
        sfc_6km_shear,
        sfc_8km_shear,
        eff_shear,
        ebwd,
        lcl_el_shear,
        mean_wind_sfc_500m,
        mean_1km,
        mean_3km,
        mean_6km,
        mean_8km,
        mean_eff,
        mean_ebw,
        mean_lcl_el,
        srw_sfc_500m,
        srw_1km,
        srw_3km,
        srw_6km,
        srw_8km,
        srw_4_5km,
        srw_eff,
        srw_ebw,
        srw_lcl_el,
        wind1km,
        wind6km,
        corfidi_up,
        corfidi_dn,
        right_critical_angle,
        brnshear,
        ehi_0_1km,
        ehi_0_3km,
        vgp,
        peskov,
        mcs_index,
        ncape,
        lrghail,
        lscp,
        nstp,
        hgz_cape,
        wbz_height,
        ecape,
        modified_sherbe,
        cape_0_3km,
        cape_0_6km,
        temp_adv,
        temp_adv_bounds,
        slinky_traj,
        slinky_tilt,
    );
    Ok(result)
}

fn comp_or_nan(value: Result<(f64, f64), impl std::fmt::Display>) -> (f64, f64) {
    value.unwrap_or((f64::NAN, f64::NAN))
}

fn helicity_or_nan(
    profile: &sharppyrs::sharprs::Profile,
    lower: f64,
    upper: f64,
    stu: f64,
    stv: f64,
) -> (f64, f64, f64) {
    winds::helicity(profile, lower, upper, stu, stv, -1.0, true).unwrap_or((
        f64::NAN,
        f64::NAN,
        f64::NAN,
    ))
}

fn sr_wind_or_nan(
    profile: &sharppyrs::sharprs::Profile,
    pbot: f64,
    ptop: f64,
    stu: f64,
    stv: f64,
) -> (f64, f64) {
    comp_or_nan(winds::sr_wind(profile, pbot, ptop, stu, stv, -1.0))
}

fn effective_parcel(profile: &Profile) -> ParcelResult {
    if !profile.ebottom.is_finite() || !profile.etop.is_finite() {
        return profile.sfcpcl.clone();
    }
    let inner = &profile.inner;
    let Some(mean_theta) = indices::mean_theta(inner, Some(profile.ebottom), Some(profile.etop))
    else {
        return profile.sfcpcl.clone();
    };
    let Some(mean_mixratio) =
        indices::mean_mixratio(inner, Some(profile.ebottom), Some(profile.etop))
    else {
        return profile.sfcpcl.clone();
    };
    let pres = (profile.ebottom + profile.etop) / 2.0;
    let tmpc = thermo::theta(1000.0, mean_theta, pres);
    let dwpc = thermo::temp_at_mixrat(mean_mixratio, pres);
    let cape_profile = cape::Profile::new(
        inner.pres.clone(),
        inner.hght.clone(),
        inner.tmpc.clone(),
        inner.dwpc.clone(),
        inner.sfc,
    );
    let level = LiftedParcelLevel {
        pres,
        tmpc,
        dwpc,
        parcel_type: SharprsParcelType::UserDefined { pres, tmpc, dwpc },
    };
    parcelx(&cape_profile, &level, None, None)
}

fn extra_dict<'py>(
    py: Python<'py>,
    profile: &Profile,
    derived: &DerivedParams,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let inner = &profile.inner;
    let (rstu, rstv, lstu, lstv) = profile.srwind;
    let p_at = |height_agl: f64| inner.pres_at_height(inner.to_msl(height_agl));
    let surface = inner.sfc_pressure();
    let p1km = p_at(1_000.0);
    let p2km = p_at(2_000.0);
    let p3km = p_at(3_000.0);
    let p4km = p_at(4_000.0);
    let p5km = p_at(5_000.0);
    let p6km = p_at(6_000.0);
    let p8km = p_at(8_000.0);
    let p9km = p_at(9_000.0);
    let p11km = p_at(11_000.0);

    let right_srh500 = helicity_or_nan(inner, 0.0, 500.0, rstu, rstv);
    let right_srh1km = helicity_or_nan(inner, 0.0, 1_000.0, rstu, rstv);
    let right_srh3km = helicity_or_nan(inner, 0.0, 3_000.0, rstu, rstv);
    let left_srh500 = helicity_or_nan(inner, 0.0, 500.0, lstu, lstv);
    let left_srh1km = helicity_or_nan(inner, 0.0, 1_000.0, lstu, lstv);
    let left_srh3km = helicity_or_nan(inner, 0.0, 3_000.0, lstu, lstv);
    let right_esrh = if profile.ebotm.is_finite() && profile.etopm.is_finite() {
        helicity_or_nan(inner, profile.ebotm, profile.etopm, rstu, rstv)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };
    let left_esrh = if profile.ebotm.is_finite() && profile.etopm.is_finite() {
        helicity_or_nan(inner, profile.ebotm, profile.etopm, lstu, lstv)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    result.set_item("bunkers", profile.srwind)?;
    result.set_item("right_srh500", right_srh500)?;
    result.set_item("right_srh1km", right_srh1km)?;
    result.set_item("right_srh3km", right_srh3km)?;
    result.set_item("left_srh500", left_srh500)?;
    result.set_item("left_srh1km", left_srh1km)?;
    result.set_item("left_srh3km", left_srh3km)?;
    result.set_item("right_esrh", right_esrh)?;
    result.set_item("left_esrh", left_esrh)?;
    result.set_item(
        "right_critical_angle",
        winds::critical_angle(inner, rstu, rstv).unwrap_or(f64::NAN),
    )?;
    result.set_item(
        "left_critical_angle",
        winds::critical_angle(inner, lstu, lstv).unwrap_or(f64::NAN),
    )?;
    result.set_item(
        "sfc_9km_shear",
        comp_or_nan(winds::wind_shear(inner, surface, p9km)),
    )?;

    for (prefix, stu, stv) in [("right", rstu, rstv), ("left", lstu, lstv)] {
        for (layer, ptop) in [("1km", p1km), ("3km", p3km), ("6km", p6km), ("8km", p8km)] {
            let comp = sr_wind_or_nan(inner, surface, ptop, stu, stv);
            result.set_item(
                format!("{prefix}_srw_{layer}"),
                sharppyrs::sharprs::profile::comp2vec(comp.0, comp.1),
            )?;
        }
        result.set_item(
            format!("{prefix}_srw_0_2km"),
            sr_wind_or_nan(inner, surface, p2km, stu, stv),
        )?;
        result.set_item(
            format!("{prefix}_srw_4_6km"),
            sr_wind_or_nan(inner, p4km, p6km, stu, stv),
        )?;
        result.set_item(
            format!("{prefix}_srw_9_11km"),
            sr_wind_or_nan(inner, p9km, p11km, stu, stv),
        )?;
        result.set_item(
            format!("{prefix}_srw_4_5km_comp"),
            sr_wind_or_nan(inner, p4km, p5km, stu, stv),
        )?;
        result.set_item(
            format!("{prefix}_srw_lcl_el_comp"),
            sr_wind_or_nan(inner, profile.mupcl.lclpres, profile.mupcl.elpres, stu, stv),
        )?;
        result.set_item(
            format!("{prefix}_srw_eff"),
            sr_wind_or_nan(inner, profile.ebottom, profile.etop, stu, stv),
        )?;
        let effective_depth = (profile.mupcl.elhght - profile.ebotm) / 2.0;
        let effective_mid = p_at(profile.ebotm + effective_depth);
        result.set_item(
            format!("{prefix}_srw_ebw"),
            sr_wind_or_nan(inner, profile.ebottom, effective_mid, stu, stv),
        )?;
    }

    let ebwspd = (derived.ebwd.0.powi(2) + derived.ebwd.1.powi(2)).sqrt();
    let shear6_ms =
        (derived.sfc_6km_shear.0.powi(2) + derived.sfc_6km_shear.1.powi(2)).sqrt() * KTS_TO_MS;
    let ebwd_ms = ebwspd * KTS_TO_MS;
    let right_stp_fixed = composites::stp_fixed(
        profile.sfcpcl.bplus,
        profile.sfcpcl.lclhght,
        right_srh1km.0,
        shear6_ms,
    )
    .unwrap_or(f64::NAN);
    let left_stp_fixed = composites::stp_fixed(
        profile.sfcpcl.bplus,
        profile.sfcpcl.lclhght,
        left_srh1km.0,
        shear6_ms,
    )
    .unwrap_or(f64::NAN);
    let right_stp_cin = composites::stp_cin(
        profile.mlpcl.bplus,
        right_esrh.0,
        ebwd_ms,
        profile.mlpcl.lclhght,
        profile.mlpcl.bminus,
    )
    .unwrap_or(f64::NAN);
    let left_stp_cin = composites::stp_cin(
        profile.mlpcl.bplus,
        left_esrh.0,
        ebwd_ms,
        profile.mlpcl.lclhght,
        profile.mlpcl.bminus,
    )
    .unwrap_or(f64::NAN);
    result.set_item("ebwspd", ebwspd)?;
    result.set_item("right_stp_fixed", right_stp_fixed)?;
    result.set_item("left_stp_fixed", left_stp_fixed)?;
    result.set_item("right_stp_cin", right_stp_cin)?;
    result.set_item("left_stp_cin", left_stp_cin)?;
    result.set_item(
        "sherbe",
        composites::sherb(
            (derived.eff_shear.0.powi(2) + derived.eff_shear.1.powi(2)).sqrt() * KTS_TO_MS,
            derived.lapserate_3km,
            derived.lapserate_700_500,
            true,
        )
        .unwrap_or(f64::NAN),
    )?;
    result.set_item(
        "upshear_downshear",
        winds::mbe_vectors(inner).unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN)),
    )?;

    let sfc = inner.sfc;
    let t950 = inner.interp_tmpc(950.0);
    let t850 = inner.interp_tmpc(850.0);
    let t700 = inner.interp_tmpc(700.0);
    let t500 = inner.interp_tmpc(500.0);
    let td850 = inner.interp_dwpc(850.0);
    let td700 = inner.interp_dwpc(700.0);
    let fire_values = PyDict::new(py);
    fire_values.set_item(
        "fosberg",
        fire::fosberg(inner.tmpc[sfc], inner.dwpc[sfc], inner.wspd[sfc]),
    )?;
    fire_values.set_item(
        "haines_hght",
        format!("{:?}", fire::haines_height(inner.sfc_height())),
    )?;
    fire_values.set_item("haines_low", fire::haines_low(t950, t850, td850))?;
    fire_values.set_item("haines_mid", fire::haines_mid(t850, t700, td850))?;
    fire_values.set_item("haines_high", fire::haines_high(t700, t500, td700))?;
    let cape_profile = cape::Profile::new(
        inner.pres.clone(),
        inner.hght.clone(),
        inner.tmpc.clone(),
        inner.dwpc.clone(),
        inner.sfc,
    );
    let fire_level = cape::define_parcel(
        &cape_profile,
        SharprsParcelType::MostUnstable { depth_hpa: 500.0 },
    );
    fire_values.set_item(
        "bplus_fire",
        cape::cape(&cape_profile, &fire_level, None, None).bplus,
    )?;
    result.set_item("fire", fire_values)?;

    let (dgz_pbot, dgz_ptop) = indices::dgz(inner);
    let winter = PyDict::new(py);
    winter.set_item("dgz_pbot", dgz_pbot)?;
    winter.set_item("dgz_ptop", dgz_ptop)?;
    winter.set_item(
        "dgz_meanrh",
        indices::mean_relh(inner, Some(dgz_pbot), Some(dgz_ptop)),
    )?;
    winter.set_item(
        "dgz_pw",
        indices::precip_water(inner, Some(dgz_pbot), Some(dgz_ptop)),
    )?;
    winter.set_item(
        "dgz_meanq",
        indices::mean_mixratio(inner, Some(dgz_pbot), Some(dgz_ptop)),
    )?;
    result.set_item("winter", winter)?;
    Ok(result)
}

fn analysis_dict<'py>(
    py: Python<'py>,
    profile: &Profile,
    derived: &DerivedParams,
    ecape: Option<EcapeResult>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("schema", "sharpmod.native-analysis.v1")?;
    result.set_item("engine", "sharppyrs/sharprs")?;
    result.set_item(
        "sharppyrs_revision",
        "958bcd685b1e28b8fce0ab5c7b8daea3cdd993aa",
    )?;
    result.set_item(
        "sharprs_revision",
        "1601674e8be0a07eaa48a50ddf6b2cedc035324f",
    )?;

    let arrays = PyDict::new(py);
    let inner = &profile.inner;
    arrays.set_item("pres", &inner.pres)?;
    arrays.set_item("hght", &inner.hght)?;
    arrays.set_item("tmpc", &inner.tmpc)?;
    arrays.set_item("dwpc", &inner.dwpc)?;
    arrays.set_item("wdir", &inner.wdir)?;
    arrays.set_item("wspd", &inner.wspd)?;
    arrays.set_item("omeg", &inner.omeg)?;
    arrays.set_item("u", &inner.u)?;
    arrays.set_item("v", &inner.v)?;
    arrays.set_item("logp", &inner.logp)?;
    arrays.set_item("vtmp", &inner.vtmp)?;
    arrays.set_item("theta", &inner.theta)?;
    arrays.set_item("thetae", &inner.thetae)?;
    arrays.set_item("wvmr", &inner.wvmr)?;
    arrays.set_item("relh", &inner.relh)?;
    arrays.set_item("wetbulb", &inner.wetbulb)?;
    result.set_item("arrays", arrays)?;

    let profile_values = PyDict::new(py);
    profile_values.set_item("sfc", inner.sfc)?;
    profile_values.set_item("top", inner.top)?;
    profile_values.set_item("latitude", inner.station.latitude)?;
    profile_values.set_item("longitude", inner.station.longitude)?;
    profile_values.set_item("ebottom", profile.ebottom)?;
    profile_values.set_item("etop", profile.etop)?;
    profile_values.set_item("ebotm", profile.ebotm)?;
    profile_values.set_item("etopm", profile.etopm)?;
    profile_values.set_item("srwind", profile.srwind)?;
    profile_values.set_item("right_esrh", profile.right_esrh)?;
    profile_values.set_item("max_lapse_rate_2_6", profile.max_lapse_rate_2_6)?;
    profile_values.set_item("dcape", profile.dcape)?;
    profile_values.set_item("dpcl_ttrace", &profile.dpcl_ttrace)?;
    profile_values.set_item("dpcl_ptrace", &profile.dpcl_ptrace)?;
    result.set_item("profile", profile_values)?;

    let parcels = PyDict::new(py);
    parcels.set_item("surface", parcel_dict(py, &profile.sfcpcl)?)?;
    parcels.set_item("forecast", parcel_dict(py, &profile.fcstpcl)?)?;
    parcels.set_item("most_unstable", parcel_dict(py, &profile.mupcl)?)?;
    parcels.set_item("mixed_layer", parcel_dict(py, &profile.mlpcl)?)?;
    parcels.set_item("effective", parcel_dict(py, &effective_parcel(profile))?)?;
    result.set_item("parcels", parcels)?;
    result.set_item("derived", derived_dict(py, derived)?)?;
    result.set_item("extra", extra_dict(py, profile, derived)?)?;

    let provenance = PyDict::new(py);
    provenance.set_item("profile", "sharprs-core")?;
    provenance.set_item("parcels", "sharprs-core")?;
    provenance.set_item("derived", "sharppyrs-rust")?;
    provenance.set_item(
        "ecape",
        if ecape.is_some() {
            "ecape-rs"
        } else {
            "unavailable"
        },
    )?;
    result.set_item("provenance", provenance)?;

    if let Some(ecape) = ecape {
        let values = PyDict::new(py);
        values.set_item("method", "ecape-rs analytic most-unstable pseudoadiabatic")?;
        values.set_item("ecape", ecape.ecape)?;
        values.set_item("ncape", ecape.ncape)?;
        values.set_item("cape", ecape.cape)?;
        values.set_item("lfc_m_msl", ecape.lfc_m_msl)?;
        values.set_item("el_m_msl", ecape.el_m_msl)?;
        result.set_item("ecape", values)?;
    } else {
        result.set_item("ecape", py.None())?;
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (
    pres,
    hght,
    tmpc,
    dwpc,
    wdir,
    wspd,
    omeg=None,
    latitude=None,
    longitude=None,
    missing=-9999.0,
    storm_motion=None,
))]
#[allow(clippy::too_many_arguments)]
fn analyze(
    py: Python<'_>,
    pres: Vec<f64>,
    hght: Vec<f64>,
    tmpc: Vec<f64>,
    dwpc: Vec<f64>,
    wdir: Vec<f64>,
    wspd: Vec<f64>,
    omeg: Option<Vec<f64>>,
    latitude: Option<f64>,
    longitude: Option<f64>,
    missing: f64,
    storm_motion: Option<(f64, f64, f64, f64)>,
) -> PyResult<Py<PyDict>> {
    let lengths = [hght.len(), tmpc.len(), dwpc.len(), wdir.len(), wspd.len()];
    if pres.len() < 2 || lengths.iter().any(|length| *length != pres.len()) {
        return Err(PyValueError::new_err(
            "pres, hght, tmpc, dwpc, wdir, and wspd must have the same length >= 2",
        ));
    }
    if omeg
        .as_ref()
        .is_some_and(|values| values.len() != pres.len())
    {
        return Err(PyValueError::new_err(
            "omeg must be omitted or have the same length as pres",
        ));
    }

    let computed = py.allow_threads(move || {
        let data = SoundingData {
            pres,
            hght,
            tmpc,
            dwpc,
            wdir,
            wspd,
            omeg,
            latitude,
            longitude,
            missing: Some(missing),
        };
        let mut profile = Profile::new(data)
            .ok_or_else(|| "sharppyrs rejected the sounding input".to_string())?;
        if let Some(srwind) = storm_motion {
            if ![srwind.0, srwind.1, srwind.2, srwind.3]
                .iter()
                .all(|value| value.is_finite())
            {
                return Err("storm_motion contains a non-finite component".to_string());
            }
            profile.srwind = srwind;
            if profile.ebotm.is_finite() && profile.etopm.is_finite() {
                profile.right_esrh = sharppyrs::extras::helicity(
                    &profile.inner,
                    profile.ebotm,
                    profile.etopm,
                    srwind.0,
                    srwind.1,
                )
                .0;
            }
        }
        let ecape = analytic_mu_ecape(&profile);
        let mut derived = DerivedParams::compute(&profile);
        if let Some(native) = ecape {
            // The authoritative vRust values come from the independently
            // parity-tested ecape-rs path, not sharppyrs' display formula.
            derived.ecape = native.ecape;
            derived.ncape = native.ncape;
        }
        Ok::<_, String>((profile, derived, ecape))
    });
    let (profile, derived, ecape) = computed.map_err(PyRuntimeError::new_err)?;
    Ok(analysis_dict(py, &profile, &derived, ecape)?.unbind())
}

#[pyfunction]
#[pyo3(signature = (
    pres,
    hght,
    tmpc,
    dwpc,
    wdir,
    wspd,
    parcel_pres,
    parcel_tmpc,
    parcel_dwpc,
    pbot=None,
    ptop=None,
    missing=-9999.0,
))]
#[allow(clippy::too_many_arguments)]
fn lift_parcel(
    py: Python<'_>,
    pres: Vec<f64>,
    hght: Vec<f64>,
    tmpc: Vec<f64>,
    dwpc: Vec<f64>,
    wdir: Vec<f64>,
    wspd: Vec<f64>,
    parcel_pres: f64,
    parcel_tmpc: f64,
    parcel_dwpc: f64,
    pbot: Option<f64>,
    ptop: Option<f64>,
    missing: f64,
) -> PyResult<Py<PyDict>> {
    let n = pres.len();
    if n < 2
        || [hght.len(), tmpc.len(), dwpc.len(), wdir.len(), wspd.len()]
            .iter()
            .any(|length| *length != n)
    {
        return Err(PyValueError::new_err(
            "all sounding arrays must have the same length >= 2",
        ));
    }
    if ![parcel_pres, parcel_tmpc, parcel_dwpc]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(PyValueError::new_err(
            "user parcel pressure, temperature, and dewpoint must be finite",
        ));
    }
    let parcel = py.allow_threads(move || {
        let analyzed = Profile::new(SoundingData {
            pres,
            hght,
            tmpc,
            dwpc,
            wdir,
            wspd,
            omeg: None,
            latitude: None,
            longitude: None,
            missing: Some(missing),
        })
        .ok_or_else(|| "sharppyrs rejected the sounding input".to_string())?;
        let inner = &analyzed.inner;
        let cape_profile = cape::Profile::new(
            inner.pres.clone(),
            inner.hght.clone(),
            inner.tmpc.clone(),
            inner.dwpc.clone(),
            inner.sfc,
        );
        let level = LiftedParcelLevel {
            pres: parcel_pres,
            tmpc: parcel_tmpc,
            dwpc: parcel_dwpc,
            parcel_type: SharprsParcelType::UserDefined {
                pres: parcel_pres,
                tmpc: parcel_tmpc,
                dwpc: parcel_dwpc,
            },
        };
        Ok::<_, String>(parcelx(&cape_profile, &level, pbot, ptop))
    });
    let parcel = parcel.map_err(PyRuntimeError::new_err)?;
    Ok(parcel_dict(py, &parcel)?.unbind())
}

#[pyfunction]
fn backend_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("schema", "sharpmod.native-analysis.v1")?;
    result.set_item("abi", "abi3-py311")?;
    result.set_item(
        "sharppyrs_revision",
        "958bcd685b1e28b8fce0ab5c7b8daea3cdd993aa",
    )?;
    result.set_item(
        "sharprs_revision",
        "1601674e8be0a07eaa48a50ddf6b2cedc035324f",
    )?;
    result.set_item(
        "ecape_rs_revision",
        "82922534c02a888e773c50463b5a49d535606276",
    )?;
    result.set_item("gil_released", true)?;
    Ok(result.unbind())
}

#[pyfunction]
fn runtime_check() -> bool {
    // Exercise the module through a real exported call. The release launcher
    // invokes this after PyInstaller extraction on every supported platform.
    true
}

#[pyfunction]
fn best_guess_precip(
    phase: i8,
    init_temp_c: f64,
    init_level_agl_m: f64,
    positive_area: f64,
    negative_area: f64,
    surface_temp_c: f64,
) -> PyResult<&'static str> {
    let phase = match phase {
        -1 => PrecipPhase::None,
        0 => PrecipPhase::Rain,
        1 => PrecipPhase::FreezingOrMix,
        3 => PrecipPhase::Snow,
        value => {
            return Err(PyValueError::new_err(format!(
                "unsupported SHARPpy precipitation phase {value}"
            )));
        }
    };
    Ok(watch_type::best_guess_precip(
        phase,
        init_temp_c,
        init_level_agl_m,
        positive_area,
        negative_area,
        surface_temp_c,
    ))
}

#[pyfunction]
fn classify_watch(values: &Bound<'_, PyDict>) -> PyResult<&'static str> {
    let get_f64 = |name: &str| -> PyResult<f64> {
        values
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing watch field {name}")))?
            .extract::<f64>()
    };
    let get_u8 = |name: &str| -> PyResult<u8> {
        values
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing watch field {name}")))?
            .extract::<u8>()
    };
    let get_string = |name: &str| -> PyResult<String> {
        values
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing watch field {name}")))?
            .extract::<String>()
    };
    let params = WatchParams {
        stp_eff: get_f64("stp_eff")?,
        stp_fixed: get_f64("stp_fixed")?,
        srw_4_6km: get_f64("srw_4_6km")?,
        esrh: get_f64("esrh")?,
        srh1km: get_f64("srh1km")?,
        sfc_8km_shear: get_f64("sfc_8km_shear")?,
        lr1: get_f64("lr1")?,
        sfcpcl_lclhght: get_f64("sfcpcl_lclhght")?,
        mlpcl_lclhght: get_f64("mlpcl_lclhght")?,
        mlpcl_bminus: get_f64("mlpcl_bminus")?,
        mupcl_bminus: get_f64("mupcl_bminus")?,
        ebotm: get_f64("ebotm")?,
        scp: get_f64("scp")?,
        ship: get_f64("ship")?,
        sig_severe: get_f64("sig_severe")?,
        mmp: get_f64("mmp")?,
        wndg: get_f64("wndg")?,
        dcape: get_f64("dcape")?,
        pwat: get_f64("pwat")?,
        pwv_flag: get_u8("pwv_flag")?,
        low_rh: get_f64("low_rh")?,
        mid_rh: get_f64("mid_rh")?,
        upshear_wspd: get_f64("upshear_wspd")?,
        sfc_tmpc: get_f64("sfc_tmpc")?,
        sfc_dwpc: get_f64("sfc_dwpc")?,
        sfc_pres: get_f64("sfc_pres")?,
        sfc_wspd_kts: get_f64("sfc_wspd_kts")?,
        precip_type: get_string("precip_type")?,
    };
    Ok(watch_type::best_watch(&params).label())
}

#[pymodule]
fn sharpmod_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze, module)?)?;
    module.add_function(wrap_pyfunction!(lift_parcel, module)?)?;
    module.add_function(wrap_pyfunction!(backend_info, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_check, module)?)?;
    module.add_function(wrap_pyfunction!(best_guess_precip, module)?)?;
    module.add_function(wrap_pyfunction!(classify_watch, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
