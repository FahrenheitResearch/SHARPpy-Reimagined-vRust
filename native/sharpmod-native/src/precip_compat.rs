//! Exact compatibility path for SHARPpy's precipitation-source diagnostics.

use sharppyrs::sharprs::params::indices;
use sharppyrs::sharprs::watch_type::{self, PrecipPhase};
use sharppyrs::sharprs::{Profile, interp, thermo};

#[derive(Clone, Debug, PartialEq)]
pub struct PrecipDetails {
    pub dgz_meanomeg: f64,
    pub oprh: f64,
    pub plevel: f64,
    pub phase: i8,
    pub tmp: f64,
    pub st: String,
    pub tpos: f64,
    pub tneg: f64,
    pub ttop: f64,
    pub tbot: f64,
    pub wpos: f64,
    pub wneg: f64,
    pub wtop: f64,
    pub wbot: f64,
    pub precip_type: String,
}

fn pressure_grid(pbot: f64, ptop: f64) -> Vec<f64> {
    if !pbot.is_finite() || !ptop.is_finite() || pbot < ptop {
        return Vec::new();
    }
    let count = (pbot - ptop + 1.0).ceil().max(0.0) as usize;
    (0..count).map(|index| pbot - index as f64).collect()
}

fn field_at(profile: &Profile, field: &[f64], pressure: f64) -> f64 {
    interp::generic_interp_pres(pressure, &profile.pres, field).unwrap_or(f64::NAN)
}

fn mean_omega(profile: &Profile, mut pbot: f64, ptop: f64, missing: f64) -> f64 {
    if !profile.omeg.iter().any(|value| value.is_finite()) {
        return missing;
    }
    if !field_at(profile, &profile.omeg, pbot).is_finite() {
        pbot = profile.sfc_pressure();
    }
    if !field_at(profile, &profile.omeg, ptop).is_finite() {
        return f64::NAN;
    }
    let mut total = 0.0;
    let mut weights = 0.0;
    for pressure in pressure_grid(pbot, ptop) {
        let value = field_at(profile, &profile.omeg, pressure);
        if value.is_finite() {
            total += value * pressure;
            weights += pressure;
        }
    }
    if weights > 0.0 {
        total / weights
    } else {
        f64::NAN
    }
}

fn classify_phase(temp: f64) -> (PrecipPhase, &'static str) {
    if temp > 0.0 {
        (PrecipPhase::Rain, "Rain")
    } else if temp <= 0.0 && temp > -5.0 {
        (PrecipPhase::FreezingOrMix, "Freezing Rain")
    } else if temp <= -5.0 && temp > -9.0 {
        (PrecipPhase::FreezingOrMix, "ZR/S Mix")
    } else if temp <= -9.0 {
        (PrecipPhase::Snow, "Snow")
    } else {
        (PrecipPhase::None, "N/A")
    }
}

fn init_phase(profile: &Profile, missing: f64) -> (f64, PrecipPhase, f64, &'static str) {
    let omega_available = profile.omeg.iter().filter(|&&value| value < 0.1).count() >= 5;
    let surface_height = profile.hght[profile.sfc];
    let mut candidates = Vec::new();
    for index in 0..profile.pres.len() {
        let agl = profile.hght[index] - surface_height;
        if !agl.is_finite() || !(0.0..5000.0).contains(&agl) {
            continue;
        }
        if omega_available && !(profile.omeg[index] <= 0.0) {
            continue;
        }
        let rh = thermo::relh(
            profile.pres[index],
            profile.tmpc[index],
            profile.dwpc[index],
        );
        if rh > 80.0 {
            let lower_pressure = profile.pres[index] + 50.0;
            let temp = interp::temp(lower_pressure, &profile.pres, &profile.tmpc);
            let dewpoint = interp::dwpt(lower_pressure, &profile.pres, &profile.dwpc);
            if let (Some(temp), Some(dewpoint)) = (temp, dewpoint) {
                if thermo::relh(lower_pressure, temp, dewpoint) > 80.0 {
                    candidates.push(profile.pres[index] + 25.0);
                }
            }
        }
    }
    let Some(&plevel) = candidates.last() else {
        return (missing, PrecipPhase::None, missing, "N/A");
    };
    let temp = interp::temp(plevel, &profile.pres, &profile.tmpc).unwrap_or(f64::NAN);
    let (phase, text) = classify_phase(temp);
    (plevel, phase, temp, text)
}

fn layer_energy(profile: &Profile, start: f64, wetbulb: bool) -> (f64, f64, f64, f64) {
    if interp::temp(500.0, &profile.pres, &profile.tmpc).is_none()
        && interp::temp(850.0, &profile.pres, &profile.tmpc).is_none()
    {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let value_at = |pressure: f64| -> f64 {
        let temp = interp::temp(pressure, &profile.pres, &profile.tmpc).unwrap_or(f64::NAN);
        if wetbulb {
            let dewpoint = interp::dwpt(pressure, &profile.pres, &profile.dwpc).unwrap_or(f64::NAN);
            thermo::wetbulb(pressure, temp, dewpoint)
        } else {
            temp
        }
    };
    let uptr = profile
        .pres
        .iter()
        .enumerate()
        .filter(|(_, pressure)| pressure.is_finite() && **pressure > start)
        .map(|(index, _)| index)
        .next_back()
        .unwrap_or(0);
    let mut h1 = interp::hght(start, &profile.pres, &profile.hght).unwrap_or(f64::NAN);
    let mut t1 = value_at(start);
    let mut warm = false;
    let mut cold = false;
    let mut positive = 0.0;
    let mut negative = 0.0;
    let mut top = 0.0;
    let mut bottom = 0.0;
    if uptr >= profile.sfc {
        for index in (profile.sfc..=uptr).rev() {
            let pressure = profile.pres[index];
            let height = profile.hght[index];
            let temp = value_at(pressure);
            let layer =
                9.8 * ((-t1 / (t1 + 273.15)) + (-temp / (temp + 273.15))) / 2.0 * (height - h1);
            if temp > 0.0 && !warm {
                warm = true;
                top = pressure;
            }
            if temp < 0.0 && warm && !cold {
                cold = true;
                bottom = pressure;
            }
            if warm {
                if layer > 0.0 {
                    positive += layer;
                } else {
                    negative += layer;
                }
            }
            h1 = height;
            t1 = temp;
        }
    }
    if warm && cold {
        (positive, negative, top, bottom)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

pub fn compute(profile: &Profile, missing: f64) -> PrecipDetails {
    let (dgz_pbot, dgz_ptop) = indices::dgz(profile);
    let dgz_meanomeg = mean_omega(profile, dgz_pbot, dgz_ptop, missing) * 10.0;
    // Use the same Rust winter fields published to Python.  This keeps OPRH
    // internally consistent with `dgz_pw` and `dgz_meanrh` in the viewer.
    let dgz_pw = indices::precip_water(profile, Some(dgz_pbot), Some(dgz_ptop)).unwrap_or(f64::NAN);
    let dgz_meanrh =
        indices::mean_relh(profile, Some(dgz_pbot), Some(dgz_ptop)).unwrap_or(f64::NAN);
    let oprh = dgz_meanomeg * dgz_pw * (dgz_meanrh / 100.0);
    let (plevel, phase, tmp, st) = init_phase(profile, missing);
    let (tpos, tneg, ttop, tbot) = layer_energy(profile, plevel, false);
    let (wpos, wneg, wtop, wbot) = layer_energy(profile, plevel, true);
    let init_height = interp::hght(plevel, &profile.pres, &profile.hght)
        .map(|height| profile.to_agl(height))
        .unwrap_or(f64::NAN);
    let precip_type = watch_type::best_guess_precip(
        phase,
        tmp,
        init_height,
        tpos,
        tneg,
        profile.tmpc[profile.sfc],
    )
    .to_string();
    PrecipDetails {
        dgz_meanomeg,
        oprh,
        plevel,
        phase: phase.code(),
        tmp,
        st: st.to_string(),
        tpos,
        tneg,
        ttop,
        tbot,
        wpos,
        wneg,
        wtop,
        wbot,
        precip_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sharppyrs::sharprs::profile::StationInfo;

    fn profile(temperatures: &[f64], omega: &[f64]) -> Profile {
        let pressure = [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0];
        let height = [0.0, 1000.0, 2000.0, 3000.0, 4200.0, 5600.0, 7200.0];
        let dewpoint: Vec<f64> = temperatures.iter().map(|temp| temp - 0.2).collect();
        Profile::new(
            &pressure,
            &height,
            temperatures,
            &dewpoint,
            &[],
            &[],
            omega,
            StationInfo::default(),
        )
        .unwrap()
    }

    #[test]
    fn phase_thresholds_match_sharppy_exactly() {
        assert_eq!(
            classify_phase(0.0),
            (PrecipPhase::FreezingOrMix, "Freezing Rain")
        );
        assert_eq!(
            classify_phase(-5.0),
            (PrecipPhase::FreezingOrMix, "ZR/S Mix")
        );
        assert_eq!(classify_phase(-9.0), (PrecipPhase::Snow, "Snow"));
    }

    #[test]
    fn phase_source_uses_upward_omega_when_five_values_are_available() {
        let temperatures = [-2.0, -2.0, 2.0, -10.0, -15.0, -20.0, -30.0];
        let upward = profile(&temperatures, &[0.0, 0.0, 0.0, 0.0, 0.05, 1.0, 1.0]);
        let (plevel, phase, _, text) = init_phase(&upward, -9999.0);
        assert_eq!(plevel, 725.0);
        assert_eq!(phase, PrecipPhase::FreezingOrMix);
        assert_eq!(text, "ZR/S Mix");

        let no_usable_omega = profile(&temperatures, &[1.0; 7]);
        let (plevel, phase, _, text) = init_phase(&no_usable_omega, -9999.0);
        assert_eq!(plevel, 625.0);
        assert_eq!(phase, PrecipPhase::Snow);
        assert_eq!(text, "Snow");
    }

    #[test]
    fn pressure_grid_has_numpy_stop_exclusive_semantics() {
        assert_eq!(pressure_grid(700.0, 700.0), vec![700.0]);
        assert_eq!(
            pressure_grid(702.5, 700.2),
            vec![702.5, 701.5, 700.5, 699.5]
        );
    }

    #[test]
    fn all_missing_omega_retains_ten_times_missing_sentinel() {
        let prof = profile(
            &[-2.0, -2.0, 2.0, -10.0, -15.0, -20.0, -30.0],
            &[f64::NAN; 7],
        );
        assert_eq!(compute(&prof, -9999.0).dgz_meanomeg, -99990.0);
    }

    #[test]
    fn warm_then_cold_layer_reports_energy_bounds_but_warm_only_reports_zeros() {
        let cold_surface = profile(&[-2.0, -2.0, 2.0, -10.0, -15.0, -20.0, -30.0], &[0.0; 7]);
        let (positive, negative, top, bottom) = layer_energy(&cold_surface, 700.0, false);
        assert_eq!(positive, 0.0);
        assert!(negative < 0.0);
        assert_eq!((top, bottom), (800.0, 900.0));

        let warm_surface = profile(&[4.0, 3.0, 2.0, -10.0, -15.0, -20.0, -30.0], &[0.0; 7]);
        assert_eq!(
            layer_energy(&warm_surface, 700.0, false),
            (0.0, 0.0, 0.0, 0.0)
        );
    }
}
