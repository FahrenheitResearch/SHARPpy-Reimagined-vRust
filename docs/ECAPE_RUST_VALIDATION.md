# Rust ECAPE validation

SHARPpy Reimagined vRust treats `ecape-parcel-py` as the numerical reference
for Peters-style analytic most-unstable ECAPE. The application calculation is
implemented by `ecape-rs`; it is not expected to be bit-for-bit identical to
the Python result because the two implementations use independent numerical
primitives and floating-point paths.

The release acceptance threshold is:

```text
absolute difference <= max(10 J/kg, 5% of the reference ECAPE)
```

This is a direct engine comparison. It reads `sharpmod_native`'s raw
`ecape-rs` result and therefore cannot pass by silently invoking the normal
Python fallback.

The viewer's **Profiles → Compare Rust vs Python…** dialog checks calculations
that have a legacy SHARPpy reference. Analytic ECAPE is instead release-gated
by the independent `ecape-parcel-py` matrix here; switching the visible profile
in that dialog is not a substitute for this oracle test.

## v0.3.2 verification matrix

The checked matrix contains 66 profiles:

- 14 controlled synthetic soundings spanning different instability,
  inhibition, moisture-depth, elevation and shear combinations;
- one observed OAX sounding;
- the same real HRRR point sounding through independent SPC and NPZ decode
  paths; and
- all 49 forecast profiles in the bundled HRRR BUFKIT example.

With `ecape-parcel` 1.2.2, MetPy 1.7.1 and NumPy 2.4.6, the reference returned
a comparable ECAPE value for 55 profiles. All 55 passed:

| Metric | Result |
| --- | ---: |
| Passing comparisons | 55 / 55 |
| Mean absolute difference | 15.12 J/kg |
| RMSE | 16.92 J/kg |
| Largest absolute difference | 42.49 J/kg |
| Mean relative difference | 0.94% |
| Largest relative difference | 2.81% |

These are independent ECAPE-oracle comparisons; they are separate from the
83,631 SHARPpy-compatibility comparisons reported in
[Rust calculation parity](NATIVE_PARITY.md).

The remaining 11 cases are BUFKIT forecast indices 1 and 22 through 31. MetPy
reported no LFC or EL for the reported-level parcel profile, and
`ecape-parcel-py` therefore raised while attempting to use the absent LFC/EL.
They are oracle-unresolved cases, not Rust agreement successes or failures;
the automated test still requires any Rust value returned for them to be
finite and non-negative.

On Windows, the same matrix also compares the separately packaged
`rw_ecape_analytic.exe` helper. When fed the exact normalized calculation
columns used by the in-process engine (including bracketed OAX wind-gap
interpolation), it produced the same scalar ECAPE values and passed all 55
reference comparisons.

## Weak-instability regression coverage

Validation exposed an older `ecape-rs` pin that returned zero for two weakly
unstable profiles and a bridge check that rejected a third valid value. In the
third case, raw analytic ECAPE was slightly greater than the solver's internal
CAPE. That can happen in the published analytic formulation and is also
accepted by `ecape-parcel-py`; it is not malformed output. The display-level
contract remains bounded separately by SHARPpy most-unstable CAPE.

The native extension now pins `ecape-rs` commit
`414cac67ce1ce4bff64c7c74449ed4ccddb3ebc0`. The three weak-instability cases
are explicit non-zero regressions in the automated matrix.

## Reproduce

After building and installing the native extension, run:

```powershell
python -m pytest -q sharpmod/tests/test_ecape_rust_parity.py
```

The test fails with the profile name, Rust value, reference value, absolute
difference and allowed tolerance. Environments without the native extension
skip the in-process test rather than accidentally validating the Python
fallback.

## Interpretation

This result establishes numerical agreement with the maintained software
reference over the committed matrix. It does not establish forecast skill,
and it does not claim exact equality for arbitrary atmospheric profiles.
Maintaining both real-profile tests and the independently generated synthetic
set is intended to catch engine, dependency and input-conversion regressions
before release.
