//! One-call native sounding analysis for SHARPpy Reimagined vRust.
//!
//! The public `sharprs` Python module exposes a deliberately small legacy
//! surface.  This private extension instead calls the real `sharppyrs`
//! analysis view-model (and therefore the real `sharprs` core) once per
//! sounding, outside the GIL, and returns plain Python containers.

mod precip_compat;

use ecape_rs::{CapeType, ParcelOptions, StormMotionType, calc_ecape_ncape};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use sharppyrs::sharprs::fire;
use sharppyrs::sharprs::params::cape::{
    self, LiftedParcelLevel, ParcelResult, ParcelType as SharprsParcelType,
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

#[derive(Clone, Debug)]
struct StreamwisenessResult {
    height_m: Vec<f64>,
    percent: Vec<f64>,
    signed_percent: Vec<f64>,
}

fn linear_interpolate(x: f64, xs: &[f64], values: &[f64], cursor: &mut usize) -> f64 {
    if x <= xs[0] {
        return values[0];
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        return values[last];
    }
    while *cursor + 1 < xs.len() && xs[*cursor + 1] < x {
        *cursor += 1;
    }
    let x0 = xs[*cursor];
    let x1 = xs[*cursor + 1];
    let fraction = (x - x0) / (x1 - x0);
    values[*cursor] + (values[*cursor + 1] - values[*cursor]) * fraction
}

/// Rust implementation of the streamwiseness inset's complete numeric path.
///
/// Wind and storm-motion inputs use knots; all derivatives and thresholds are
/// evaluated in SI units. Equal heights retain the first sample after a
/// stable sort, matching NumPy's ``unique(..., return_index=True)`` behavior.
fn streamwiseness_core(
    height_msl_m: &[f64],
    u_kts: &[f64],
    v_kts: &[f64],
    sfc: usize,
    storm_u_kts: f64,
    storm_v_kts: f64,
    max_height_m: f64,
    step_m: f64,
) -> Option<StreamwisenessResult> {
    if height_msl_m.len() != u_kts.len()
        || height_msl_m.len() != v_kts.len()
        || height_msl_m.len() < 2
        || sfc >= height_msl_m.len()
        || !height_msl_m[sfc].is_finite()
        || !storm_u_kts.is_finite()
        || !storm_v_kts.is_finite()
        || !max_height_m.is_finite()
        || !step_m.is_finite()
        || max_height_m <= 0.0
        || step_m <= 0.0
    {
        return None;
    }

    let surface_height = height_msl_m[sfc];
    let mut samples: Vec<(f64, f64, f64)> = height_msl_m[sfc..]
        .iter()
        .zip(&u_kts[sfc..])
        .zip(&v_kts[sfc..])
        .filter_map(|((&height, &u), &v)| {
            let height_agl = height - surface_height;
            (height_agl.is_finite() && u.is_finite() && v.is_finite()).then_some((height_agl, u, v))
        })
        .collect();
    if samples.len() < 2 {
        return None;
    }
    samples.sort_by(|left, right| left.0.total_cmp(&right.0));
    samples.dedup_by(|left, right| left.0 == right.0);
    if samples.len() < 2 {
        return None;
    }

    let height: Vec<f64> = samples.iter().map(|sample| sample.0).collect();
    let u: Vec<f64> = samples.iter().map(|sample| sample.1).collect();
    let v: Vec<f64> = samples.iter().map(|sample| sample.2).collect();
    let top = max_height_m.min(*height.last()?);
    if top < step_m {
        return None;
    }
    let step_count = (top / step_m).floor() as usize;
    if step_count < 1 {
        return None;
    }
    let grid: Vec<f64> = (0..=step_count)
        .map(|index| index as f64 * step_m)
        .collect();

    let mut u_cursor = 0usize;
    let mut v_cursor = 0usize;
    let u_ms: Vec<f64> = grid
        .iter()
        .map(|&target| linear_interpolate(target, &height, &u, &mut u_cursor) * KTS_TO_MS)
        .collect();
    let v_ms: Vec<f64> = grid
        .iter()
        .map(|&target| linear_interpolate(target, &height, &v, &mut v_cursor) * KTS_TO_MS)
        .collect();
    let storm_u_ms = storm_u_kts * KTS_TO_MS;
    let storm_v_ms = storm_v_kts * KTS_TO_MS;

    let mut dudz = vec![f64::NAN; grid.len()];
    let mut dvdz = vec![f64::NAN; grid.len()];
    let last = grid.len() - 1;
    dudz[0] = (u_ms[1] - u_ms[0]) / step_m;
    dvdz[0] = (v_ms[1] - v_ms[0]) / step_m;
    dudz[last] = (u_ms[last] - u_ms[last - 1]) / step_m;
    dvdz[last] = (v_ms[last] - v_ms[last - 1]) / step_m;
    for index in 1..last {
        dudz[index] = (u_ms[index + 1] - u_ms[index - 1]) / (2.0 * step_m);
        dvdz[index] = (v_ms[index + 1] - v_ms[index - 1]) / (2.0 * step_m);
    }

    let mut any_usable = false;
    let mut percent = vec![f64::NAN; grid.len()];
    let mut signed_percent = vec![f64::NAN; grid.len()];
    for index in 0..grid.len() {
        let omega_u = -dvdz[index];
        let omega_v = dudz[index];
        let omega_mag = omega_u.hypot(omega_v);
        let u_sr = u_ms[index] - storm_u_ms;
        let v_sr = v_ms[index] - storm_v_ms;
        let sr_speed = u_sr.hypot(v_sr);
        if omega_mag <= 1.0e-6 || sr_speed <= 0.1 {
            continue;
        }
        let omega_streamwise = omega_u * (u_sr / sr_speed) + omega_v * (v_sr / sr_speed);
        let ratio = omega_streamwise / omega_mag;
        let value = (ratio * ratio * 100.0).clamp(0.0, 100.0);
        let sign = if omega_streamwise > 0.0 {
            1.0
        } else if omega_streamwise < 0.0 {
            -1.0
        } else {
            0.0
        };
        percent[index] = value;
        signed_percent[index] = sign * value;
        any_usable = true;
    }
    any_usable.then_some(StreamwisenessResult {
        height_m: grid,
        percent,
        signed_percent,
    })
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
        mplpres, bfzl, b3km, b6km, li5, li3, limax, limaxpres, cap, cappres, bmin, bminpres, p0c,
        pm10c, pm20c, pm30c, hght0c, hghtm10c, hghtm20c, hghtm30c, wm10c, wm20c, wm30c, ptrace,
        ttrace,
    );
    // SHARPpy can retain a previously diagnosed MPL height after a later EL
    // crossing masks only MPL pressure.  It is an observable legacy quirk,
    // so expose the two fields independently rather than sanitizing it here.
    result.set_item("mplhght", parcel.mplhght)?;
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
    if (!stu.is_finite() || !stv.is_finite()) && sharppyrs::extras::has_constant_wind(profile) {
        return (0.0, 0.0, 0.0);
    }
    sharppyrs::extras::helicity(profile, lower, upper, stu, stv)
}

fn sr_wind_or_nan(
    profile: &sharppyrs::sharprs::Profile,
    pbot: f64,
    ptop: f64,
    stu: f64,
    stv: f64,
) -> (f64, f64) {
    // SHARPpy's exclusive np.arange stop leaves a zero-depth effective layer
    // with exactly its one reported pressure sample.  sharprs's inclusive
    // loop otherwise adds ptop - 1 hPa and biases both mover diagnostics.
    if (pbot - ptop).abs() <= 1.0e-9 {
        let (u, v) = profile.interp_wind(pbot);
        if u.is_finite() && v.is_finite() && stu.is_finite() && stv.is_finite() {
            return (u - stu, v - stv);
        }
        return (f64::NAN, f64::NAN);
    }
    comp_or_nan(winds::sr_wind(profile, pbot, ptop, stu, stv, -1.0))
}

/// SHARPpy's pressure interpolator removes masked samples before applying
/// `numpy.interp` in log-pressure space.  Keep this local compatibility path
/// for fields whose public values are compared directly with legacy SHARPpy.
fn interp_pressure_sharppy(
    profile: &sharppyrs::sharprs::Profile,
    field: &[f64],
    target: f64,
) -> f64 {
    if !target.is_finite() || target <= 0.0 {
        return f64::NAN;
    }
    let valid: Vec<(f64, f64)> = profile
        .pres
        .iter()
        .copied()
        .zip(field.iter().copied())
        .filter(|(pressure, value)| pressure.is_finite() && *pressure > 0.0 && value.is_finite())
        .collect();
    if valid.is_empty() || target > valid[0].0 || target < valid[valid.len() - 1].0 {
        return f64::NAN;
    }
    for &(pressure, value) in &valid {
        if target == pressure {
            return value;
        }
    }
    for pair in valid.windows(2) {
        let (p0, v0) = pair[0];
        let (p1, v1) = pair[1];
        if target <= p0 && target >= p1 {
            let fraction = (target.ln() - p1.ln()) / (p0.ln() - p1.ln());
            return v1 + (v0 - v1) * fraction;
        }
    }
    f64::NAN
}

/// Exact `np.arange(pbot, ptop - 1, -1)` grid used by SHARPpy's fast
/// pressure-layer means.  Fractional bounds intentionally keep their
/// fractional offset instead of snapping the final point to `ptop`.
fn sharppy_pressure_grid(pbot: f64, ptop: f64) -> Vec<f64> {
    if !pbot.is_finite() || !ptop.is_finite() || pbot < ptop {
        return Vec::new();
    }
    let count = (pbot - ptop + 1.0).ceil().max(0.0) as usize;
    (0..count).map(|index| pbot - index as f64).collect()
}

fn sharppy_pressure_weighted_mean(
    profile: &sharppyrs::sharprs::Profile,
    field: &[f64],
    pbot: f64,
    ptop: f64,
) -> f64 {
    let mut weighted = 0.0;
    let mut weights = 0.0;
    for pressure in sharppy_pressure_grid(pbot, ptop) {
        let value = interp_pressure_sharppy(profile, field, pressure);
        if value.is_finite() {
            weighted += value * pressure;
            weights += pressure;
        }
    }
    if weights > 0.0 {
        weighted / weights
    } else {
        f64::NAN
    }
}

fn sharppy_mean_relh(profile: &sharppyrs::sharprs::Profile, mut pbot: f64, ptop: f64) -> f64 {
    if !interp_pressure_sharppy(profile, &profile.tmpc, pbot).is_finite() {
        pbot = profile.sfc_pressure();
    }
    if !interp_pressure_sharppy(profile, &profile.tmpc, ptop).is_finite() {
        return f64::NAN;
    }
    let mut weighted = 0.0;
    let mut weights = 0.0;
    for pressure in sharppy_pressure_grid(pbot, ptop) {
        let temp = interp_pressure_sharppy(profile, &profile.tmpc, pressure);
        let dewpoint = interp_pressure_sharppy(profile, &profile.dwpc, pressure);
        if temp.is_finite() && dewpoint.is_finite() {
            let rh = thermo::relh(pressure, temp, dewpoint);
            if rh.is_finite() {
                weighted += rh * pressure;
                weights += pressure;
            }
        }
    }
    if weights > 0.0 {
        weighted / weights
    } else {
        f64::NAN
    }
}

fn sharppy_pbl_top(profile: &sharppyrs::sharprs::Profile) -> f64 {
    let theta_v = |index: usize| {
        let pressure = profile.pres[index];
        let temp = profile.tmpc[index];
        let dewpoint = profile.dwpc[index];
        if pressure.is_finite() && temp.is_finite() && dewpoint.is_finite() {
            thermo::theta(
                pressure,
                thermo::virtemp(pressure, temp, Some(dewpoint)),
                1000.0,
            )
        } else {
            f64::NAN
        }
    };
    let surface_theta_v = theta_v(profile.sfc);
    if surface_theta_v.is_finite() {
        for index in 0..profile.pres.len() {
            let value = theta_v(index);
            if value.is_finite() && value > surface_theta_v + 0.5 {
                return profile.pres[index];
            }
        }
    }
    profile.pres.last().copied().unwrap_or(f64::NAN)
}

fn sharppy_observed_max_wind(
    profile: &sharppyrs::sharprs::Profile,
    observed_wind_valid: Option<&[bool]>,
    lower: f64,
    upper: f64,
) -> (f64, f64, f64) {
    let Some(observed) = observed_wind_valid else {
        return winds::max_wind(profile, lower, upper).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    };
    if observed.len() != profile.pres.len() || !lower.is_finite() || !upper.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let plower = profile.pres_at_height(profile.to_msl(lower));
    let pupper = profile.pres_at_height(profile.to_msl(upper));
    if !plower.is_finite() || !pupper.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let is_close = |left: f64, right: f64| (left - right).abs() <= 1.0e-8 + 1.0e-5 * right.abs();
    let mut best: Option<(f64, f64, f64, f64)> = None;
    for (index, was_observed) in observed.iter().copied().enumerate() {
        if !was_observed {
            continue;
        }
        let pressure = profile.pres[index];
        let u = profile.u[index];
        let v = profile.v[index];
        if !pressure.is_finite() || !u.is_finite() || !v.is_finite() {
            continue;
        }
        let below_lower = pressure < plower || is_close(pressure, plower);
        let above_upper = pressure > pupper || is_close(pressure, pupper);
        if !below_lower || !above_upper {
            continue;
        }
        let speed = u.hypot(v);
        // Strictly greater preserves the first (lowest-altitude) observation
        // when equal maxima occur, matching SHARPpy's documented contract.
        if best.is_none_or(|current| speed > current.0) {
            best = Some((speed, u, v, pressure));
        }
    }
    best.map(|(_, u, v, pressure)| (u, v, pressure))
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN))
}

fn effective_parcel(profile: &Profile) -> ParcelResult {
    if !profile.ebottom.is_finite() || !profile.etop.is_finite() {
        return profile.sfcpcl.clone();
    }
    let inner = &profile.inner;
    let Some(mean_theta) =
        sharppyrs::extras::sharppy_mean_theta(inner, profile.ebottom, profile.etop)
    else {
        return profile.sfcpcl.clone();
    };
    let Some(mean_mixratio) =
        sharppyrs::extras::sharppy_mean_mixratio(inner, profile.ebottom, profile.etop)
    else {
        return profile.sfcpcl.clone();
    };
    let pres = (profile.ebottom + profile.etop) / 2.0;
    let tmpc = thermo::theta(1000.0, mean_theta, pres);
    let dwpc = thermo::temp_at_mixrat(mean_mixratio, pres);
    let cape_profile = sharppyrs::extras::cape_profile(inner);
    let level = LiftedParcelLevel {
        pres,
        tmpc,
        dwpc,
        parcel_type: SharprsParcelType::UserDefined { pres, tmpc, dwpc },
    };
    sharppyrs::extras::parcelx_sharppy(&cape_profile, &level, None, None)
}

fn extra_dict<'py>(
    py: Python<'py>,
    profile: &Profile,
    derived: &DerivedParams,
    observed_wind_valid: Option<&[bool]>,
    missing: f64,
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
    let has_effective_layer = profile.ebottom.is_finite() && profile.etop.is_finite();
    result.set_item(
        "right_critical_angle",
        if has_effective_layer {
            winds::critical_angle(inner, rstu, rstv).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        },
    )?;
    result.set_item(
        "left_critical_angle",
        if has_effective_layer {
            winds::critical_angle(inner, lstu, lstv).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        },
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
    let southern = inner.station.latitude < 0.0;
    let mut right_stp_cin = if has_effective_layer {
        composites::stp_cin(
            profile.mlpcl.bplus,
            if southern {
                -right_esrh.0
            } else {
                right_esrh.0
            },
            ebwd_ms,
            profile.mlpcl.lclhght,
            profile.mlpcl.bminus,
        )
        .unwrap_or(f64::NAN)
    } else {
        0.0
    };
    let mut left_stp_cin = if has_effective_layer {
        composites::stp_cin(
            profile.mlpcl.bplus,
            if southern { -left_esrh.0 } else { left_esrh.0 },
            ebwd_ms,
            profile.mlpcl.lclhght,
            profile.mlpcl.bminus,
        )
        .unwrap_or(f64::NAN)
    } else {
        0.0
    };
    if southern {
        right_stp_cin = -right_stp_cin;
        left_stp_cin = -left_stp_cin;
    }
    result.set_item("ebwspd", ebwspd)?;
    result.set_item("right_stp_fixed", right_stp_fixed)?;
    result.set_item("left_stp_fixed", left_stp_fixed)?;
    result.set_item("right_stp_cin", right_stp_cin)?;
    result.set_item("left_stp_cin", left_stp_cin)?;
    result.set_item(
        "sherbe",
        composites::sherb(
            ebwspd * KTS_TO_MS,
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
    let cape_profile = sharppyrs::extras::cape_profile(inner);
    let fire_level = cape::define_parcel(
        &cape_profile,
        SharprsParcelType::MostUnstable { depth_hpa: 500.0 },
    );
    fire_values.set_item(
        "bplus_fire",
        cape::parcelx(&cape_profile, &fire_level, None, None).bplus,
    )?;
    let ppbl_top = sharppy_pbl_top(inner);
    let pbl_h = inner.to_agl(interp_pressure_sharppy(inner, &inner.hght, ppbl_top));
    let p1km = inner.pres_at_height(inner.to_msl(1_000.0));
    let meanwind01km = (
        sharppy_pressure_weighted_mean(inner, &inner.u, surface, p1km),
        sharppy_pressure_weighted_mean(inner, &inner.v, surface, p1km),
    );
    let meanwindpbl = (
        sharppy_pressure_weighted_mean(inner, &inner.u, surface, ppbl_top),
        sharppy_pressure_weighted_mean(inner, &inner.v, surface, ppbl_top),
    );
    fire_values.set_item("ppbl_top", ppbl_top)?;
    fire_values.set_item("pbl_h", pbl_h)?;
    fire_values.set_item(
        "sfc_rh",
        thermo::relh(inner.pres[sfc], inner.tmpc[sfc], inner.dwpc[sfc]),
    )?;
    fire_values.set_item("rh01km", sharppy_mean_relh(inner, surface, p1km))?;
    fire_values.set_item("pblrh", sharppy_mean_relh(inner, surface, ppbl_top))?;
    fire_values.set_item("meanwind01km", meanwind01km)?;
    fire_values.set_item("meanwindpbl", meanwindpbl)?;
    fire_values.set_item(
        "pblmaxwind",
        sharppy_observed_max_wind(inner, observed_wind_valid, 0.0, pbl_h),
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
    let precip = precip_compat::compute(inner, missing);
    winter.set_item("dgz_meanomeg", precip.dgz_meanomeg)?;
    winter.set_item("oprh", precip.oprh)?;
    winter.set_item("plevel", precip.plevel)?;
    winter.set_item("phase", precip.phase)?;
    winter.set_item("tmp", precip.tmp)?;
    winter.set_item("st", precip.st)?;
    winter.set_item("tpos", precip.tpos)?;
    winter.set_item("tneg", precip.tneg)?;
    winter.set_item("ttop", precip.ttop)?;
    winter.set_item("tbot", precip.tbot)?;
    winter.set_item("wpos", precip.wpos)?;
    winter.set_item("wneg", precip.wneg)?;
    winter.set_item("wtop", precip.wtop)?;
    winter.set_item("wbot", precip.wbot)?;
    winter.set_item("precip_type", precip.precip_type)?;
    result.set_item("winter", winter)?;
    Ok(result)
}

fn analysis_dict<'py>(
    py: Python<'py>,
    profile: &Profile,
    derived: &DerivedParams,
    ecape: Option<EcapeResult>,
    observed_wind_valid: Option<&[bool]>,
    missing: f64,
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
    // sharprs::Profile caches a Bolton theta-e array, while SHARPpy's public
    // BasicProfile contract uses its Wobus lift and stores kelvin.  The
    // sharprs thermo function is the Wobus port; expose that reference array
    // without changing internal algorithms that intentionally use Bolton.
    let thetae: Vec<f64> = inner
        .pres
        .iter()
        .zip(&inner.tmpc)
        .zip(&inner.dwpc)
        .map(|((&p, &t), &td)| {
            if p.is_finite() && t.is_finite() && td.is_finite() {
                thermo::thetae(p, t, td) + 273.15
            } else {
                f64::NAN
            }
        })
        .collect();
    arrays.set_item("thetae", thetae)?;
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
    result.set_item(
        "extra",
        extra_dict(py, profile, derived, observed_wind_valid, missing)?,
    )?;

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
    observed_wind_valid=None,
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
    observed_wind_valid: Option<Vec<bool>>,
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
    if observed_wind_valid
        .as_ref()
        .is_some_and(|values| values.len() != pres.len())
    {
        return Err(PyValueError::new_err(
            "observed_wind_valid must be omitted or have the same length as pres",
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
            // Keep the raw value in the dedicated `ecape` result for oracle
            // comparisons, but enforce the public/display contract against
            // the native most-unstable parcel CAPE.
            derived.ecape = if profile.mupcl.bplus.is_finite() && profile.mupcl.bplus > 0.0 {
                native.ecape.clamp(0.0, profile.mupcl.bplus)
            } else {
                0.0
            };
        }
        Ok::<_, String>((profile, derived, ecape, observed_wind_valid, missing))
    });
    let (profile, derived, ecape, observed_wind_valid, missing) =
        computed.map_err(PyRuntimeError::new_err)?;
    Ok(analysis_dict(
        py,
        &profile,
        &derived,
        ecape,
        observed_wind_valid.as_deref(),
        missing,
    )?
    .unbind())
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
        let cape_profile = sharppyrs::extras::cape_profile(inner);
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
        Ok::<_, String>(sharppyrs::extras::parcelx_sharppy(
            &cape_profile,
            &level,
            pbot,
            ptop,
        ))
    });
    let parcel = parcel.map_err(PyRuntimeError::new_err)?;
    Ok(parcel_dict(py, &parcel)?.unbind())
}

#[pyfunction]
#[pyo3(signature = (
    height_msl_m,
    u_kts,
    v_kts,
    sfc,
    storm_u_kts,
    storm_v_kts,
    max_height_m=6000.0,
    step_m=100.0,
))]
#[allow(clippy::too_many_arguments)]
fn streamwiseness(
    py: Python<'_>,
    height_msl_m: Vec<f64>,
    u_kts: Vec<f64>,
    v_kts: Vec<f64>,
    sfc: usize,
    storm_u_kts: f64,
    storm_v_kts: f64,
    max_height_m: f64,
    step_m: f64,
) -> PyResult<Option<(Vec<f64>, Vec<f64>, Vec<f64>)>> {
    if height_msl_m.len() != u_kts.len() || height_msl_m.len() != v_kts.len() {
        return Err(PyValueError::new_err(
            "height_msl_m, u_kts, and v_kts must have identical lengths",
        ));
    }
    let result = py.allow_threads(move || {
        streamwiseness_core(
            &height_msl_m,
            &u_kts,
            &v_kts,
            sfc,
            storm_u_kts,
            storm_v_kts,
            max_height_m,
            step_m,
        )
    });
    Ok(result.map(|values| (values.height_m, values.percent, values.signed_percent)))
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
        "414cac67ce1ce4bff64c7c74449ed4ccddb3ebc0",
    )?;
    result.set_item("streamwiseness", "rust")?;
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

/// Literal `sharppy.sharptab.watch_type.possible_watch` priority semantics.
///
/// The upstream Rust port intentionally corrected the Python routine's final
/// marginal-tornado operator precedence and its historical heat-index RH
/// unit mix-up. This bridge promises SHARPpy equivalence, so preserve those
/// established classifier results here instead of silently changing labels.
fn legacy_best_watch(p: &WatchParams) -> &'static str {
    if p.stp_eff >= 3.0
        && p.stp_fixed >= 3.0
        && p.srh1km >= 200.0
        && p.esrh >= 200.0
        && p.srw_4_6km >= 15.0
        && p.sfc_8km_shear > 45.0
        && p.sfcpcl_lclhght < 1000.0
        && p.mlpcl_lclhght < 1200.0
        && p.lr1 >= 5.0
        && p.mlpcl_bminus > -50.0
        && p.ebotm == 0.0
    {
        return "PDS TOR";
    }
    if (p.stp_eff >= 3.0 || p.stp_fixed >= 4.0) && p.mlpcl_bminus > -125.0 && p.ebotm == 0.0 {
        return "TOR";
    }
    if (p.stp_eff >= 1.0 || p.stp_fixed >= 1.0)
        && (p.srw_4_6km >= 15.0 || p.sfc_8km_shear >= 40.0)
        && p.mlpcl_bminus > -50.0
        && p.ebotm == 0.0
    {
        return "TOR";
    }
    if (p.stp_eff >= 1.0 || p.stp_fixed >= 1.0)
        && (p.low_rh + p.mid_rh) / 2.0 >= 60.0
        && p.lr1 >= 5.0
        && p.mlpcl_bminus > -50.0
        && p.ebotm == 0.0
    {
        return "TOR";
    }
    if (p.stp_eff >= 1.0 || p.stp_fixed >= 1.0) && p.mlpcl_bminus > -150.0 && p.ebotm == 0.0 {
        return "MRGL TOR";
    }
    // Python's `and` binds more tightly than `or`: the CIN/surface gates
    // apply only to the fixed-STP arm of this particular condition.
    if (p.stp_eff >= 0.5 && p.esrh >= 150.0)
        || (p.stp_fixed >= 0.5 && p.srh1km >= 150.0 && p.mlpcl_bminus > -50.0 && p.ebotm == 0.0)
    {
        return "MRGL TOR";
    }

    if (p.stp_fixed >= 1.0 || p.scp >= 4.0 || p.stp_eff >= 1.0) && p.mupcl_bminus >= -50.0 {
        return "SVR";
    }
    if p.scp >= 2.0 && (p.ship >= 1.0 || p.dcape >= 750.0) && p.mupcl_bminus >= -50.0 {
        return "SVR";
    }
    if p.sig_severe >= 30_000.0 && p.mmp >= 0.6 && p.mupcl_bminus >= -50.0 {
        return "SVR";
    }
    if p.mupcl_bminus >= -75.0 && (p.wndg >= 0.5 || p.ship >= 0.5 || p.scp >= 0.5) {
        return "MRGL SVR";
    }
    if p.pwv_flag >= 2 && p.upshear_wspd < 25.0 {
        return "FLASH FLOOD";
    }
    if p.sfc_wspd_kts * 1.150_78 > 35.0 && p.sfc_tmpc <= 0.0 && p.precip_type.contains("Snow") {
        return "BLIZZARD";
    }

    let temp_f = thermo::ctof(p.sfc_tmpc);
    // Preserve SHARPpy's historical call, which passes Fahrenheit `temp_f`
    // alongside a Celsius dewpoint to thermo.relh.
    let rh = thermo::relh(p.sfc_pres, temp_f, p.sfc_dwpc);
    if watch_type::heat_index(temp_f, rh) > 105.0 {
        return "EXCESSIVE HEAT";
    }
    "NONE"
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
    Ok(legacy_best_watch(&params))
}

#[pymodule]
fn sharpmod_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze, module)?)?;
    module.add_function(wrap_pyfunction!(lift_parcel, module)?)?;
    module.add_function(wrap_pyfunction!(streamwiseness, module)?)?;
    module.add_function(wrap_pyfunction!(backend_info, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_check, module)?)?;
    module.add_function(wrap_pyfunction!(best_guess_precip, module)?)?;
    module.add_function(wrap_pyfunction!(classify_watch, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamwiseness_squares_the_signed_projection_ratio() {
        // omega=(3,4)*1e-3 has magnitude 5e-3. Construct a unit storm-
        // relative vector whose projection on omega is exactly half of that
        // magnitude: the corrected squared definition is 25%, not 50%.
        let root_three_over_two = (0.75_f64).sqrt();
        let sr_u_ms = 0.5 * 0.6 + root_three_over_two * -0.8;
        let sr_v_ms = 0.5 * 0.8 + root_three_over_two * 0.6;
        let result = streamwiseness_core(
            &[0.0, 100.0],
            &[0.0, 0.4 / KTS_TO_MS],
            &[0.0, -0.3 / KTS_TO_MS],
            0,
            -sr_u_ms / KTS_TO_MS,
            -sr_v_ms / KTS_TO_MS,
            100.0,
            100.0,
        )
        .expect("3-4-5 vorticity profile should be usable");
        assert!((result.percent[0] - 25.0).abs() < 1.0e-10);
        assert!((result.signed_percent[0] - 25.0).abs() < 1.0e-10);
    }

    #[test]
    fn streamwiseness_normalizes_levels_and_rejects_zero_vorticity() {
        let normalized = streamwiseness_core(
            &[999.0, 1300.0, 1100.0, 1100.0, f64::NAN],
            &[0.0, 3.0, 1.0, 99.0, 0.0],
            &[0.0, 0.0, 0.0, 99.0, 0.0],
            0,
            0.0,
            -10.0,
            250.0,
            100.0,
        )
        .expect("sorted, deduplicated profile should be usable");
        assert_eq!(normalized.height_m, vec![0.0, 100.0, 200.0]);
        assert!(normalized.percent.iter().all(|value| value.is_finite()));

        assert!(
            streamwiseness_core(
                &[0.0, 100.0],
                &[1.0, 1.0],
                &[2.0, 2.0],
                0,
                0.0,
                0.0,
                100.0,
                100.0,
            )
            .is_none()
        );
    }
}
