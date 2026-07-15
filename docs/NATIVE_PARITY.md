# Rust calculation parity

SHARPpy Reimagined vRust validates its native sounding calculations against
the vendored Python SHARPpy behavior before release. The validation exercises
the public profile objects consumed by the viewer, so it covers both the Rust
engines and the adapter that translates native results into SHARPpy's object
model.

The acceptance statement is numerical agreement within published,
unit-specific tolerances. It is not a claim of bit-for-bit equality or proof
over every physically possible atmosphere.

> **Current numerical status:** the 57-profile corpus passed 83,631 of 83,631
> comparisons. The complete 100-case stress run isolated three failures to one
> zero-depth effective-layer case; after correction, that case passed 1,455 of
> 1,455 comparisons and the affected canonical profile passed 1,473 of 1,473.
> The final packaged-extension hash remains pending the release build.

## What is covered

The main audit inventories every key in the 84-field Rust `derived` schema.
There are no untracked native fields:

- 82 fields use the SHARPpy or pre-Rust SHARPpy Reimagined Python formula as a
  tolerance-gated oracle;
- analytic ECAPE has a dedicated `ecape-parcel-py` oracle because it is not a
  legacy SHARPpy calculation; and
- MMP is recorded but intentionally non-gating because the legacy function
  reads uninitialized `np.empty` cells and is nondeterministic.

The audit also checks all five public parcels, parcel and downdraft traces,
environmental arrays, thermodynamic and kinematic layers, fire and winter
indices, severe-weather composites, watch and precipitation categories,
temperature advection, and the normalized storm-slinky path. It requires the
profile backend to identify itself as `sharppyrs/sharprs`; a silent Python
fallback is a failure.

The native engine owns all normal runtime meteorological calculations,
including the detailed fire/PBL and precipitation-source/layer fields. The
only Python work retained on a successful native profile is SARS analog-
database lookup and station PWV-climatology lookup. Those are data lookups,
not numerical-formula fallbacks; failure or explicit disabling of the native
extension still selects the complete legacy Python profile fallback.

## Validation matrices

The committed corpus contains 57 profiles:

- one observed OAX SPC sounding;
- the same real HRRR point sounding through independent SPC and NPZ decode
  paths;
- all 49 forecast hours in the bundled HRRR BUFKIT file; and
- five deterministic regimes covering stable/dry, moist/unstable, elevated,
  zero-wind, and Southern Hemisphere behavior.

A second fixed-seed stress matrix contains 100 perturbed profiles:

| Family | Cases |
| --- | ---: |
| Dense HRRR perturbations | 50 |
| BUFKIT / missing-upper-moisture perturbations | 25 |
| Sparse observed OAX perturbations | 15 |
| Stable, saturated, hot/moist, dry-mixed and strong-shear edges | 10 |

The stress matrix has 50 Northern and 50 Southern Hemisphere profiles, 35
cases with missing upper moisture, 14 with added internal wind gaps, 13 with
missing omega, randomized elevation changes, and independently varied wind
speed, turning, and deep-layer shear. Its generator seed is `0x5A17C0DE` and
the complete input matrix SHA-256 is:

```text
E6BF88FF073EFA5DF6327203CE087B2A6E5B652F8D3E6AAFDC7A11DA9606BE68
```

Any change to the generated numerical inputs, metadata, source ordering, or
seed changes that fingerprint and fails the focused regression test.

The complete stress run exposed only three failed comparisons, all in
`fuzz-061-bufkit_upper_dry`: a valid zero-depth effective inflow layer. The
post-fix targeted rerun passed all 1,455 comparisons for that case. The
corresponding canonical zero-depth profile, `bufkit-kbvo-2026062608`, also
passed all 1,473 comparisons. This focused rerun preserves the failure's exact
inputs while avoiding an unnecessary repeat of every already-passing stress
case.

## Tolerance policy

Every numeric comparison uses the same rule:

```text
absolute error <= max(field absolute tolerance,
                      field relative tolerance * abs(Python reference))
```

Representative contracts are:

| Quantity | Absolute floor | Relative limit |
| --- | ---: | ---: |
| CAPE/CIN and other energies | 10 J/kg | 5% |
| Heights | 50 m | 5% |
| Pressure levels | 2 hPa | 1% |
| Temperature | 0.25 C | 2% |
| Wind components/speeds | 0.5 kt | 5% |
| Storm-relative helicity | 5 m2/s2 | 5% |
| Basic composites | 0.05 | 5% |

Environmental input arrays use much tighter identity tolerances. Wind
directions are compared circularly, polar direction/speed pairs are converted
to Cartesian components, parcel and downdraft traces are interpolated onto a
common log-pressure grid, and slinky paths are resampled by normalized path
distance. Missing-value state and categorical differences are hard failures;
they are not hidden behind numeric tolerances.

The tolerances are defined once in `sharpmod/tools/native_parity.py`. The fuzz
driver calls that auditor directly and deliberately owns no second tolerance
table, so a difficult randomized profile cannot pass by widening a stress-test
threshold.

## Explicit behavior seams

The report distinguishes corrections and undefined legacy behavior from
ordinary numerical agreement:

- **MMP:** legacy SHARPpy can read uninitialized memory. The deterministic Rust
  result is retained as informational evidence but cannot be judged against an
  undefined oracle. For watch categories only, the audit temporarily supplies
  the deterministic Rust MMP value to SHARPpy's own watch classifier; any row
  affected by that normalization is counted explicitly.
- **Full-profile extended formulas:** DCP, VGP, normalized CAPE, large-hail and
  related pre-Rust SHARPpy Reimagined formulas are called on the full physical
  legacy `ConvectiveProfile`. The old lightweight companion hard-coded a
  surface index of zero and lost sparse wind context; its differences are
  separately inventoried rather than used as the scientific reference.
- **Observed OAX normalization:** the OAX file starts with a placeholder row and
  contains internal missing wind levels. The native bridge removes only leading
  non-physical rows and interpolates only bracketed internal winds in u/v on
  log pressure. It never extrapolates exterior gaps.
- **Hemisphere-selected aggregates:** upstream SHARPpy hard-wires public STP,
  SCP and watch aggregates to the right mover. vRust selects the left mover
  south of the equator. The audit compares each aggregate with the appropriate
  side-specific legacy ingredient while still gating both right- and left-mover
  values independently.
- **Upper stratosphere:** saturation-derived environmental arrays below 100 hPa
  are outside SHARPpy's sounding-analysis domain and can be undefined in the
  legacy formulas. Their exclusion is counted explicitly; pressure-level
  coverage and all in-domain levels remain gated.

These are named, counted seams. The audit does not treat arbitrary Python
missing values as informational, and it does not relax tolerances to obtain a
pass.

## ECAPE

ECAPE is checked directly against `ecape-parcel-py`, not against a simplified
legacy field or a Python fallback hidden behind the normal adapter. The
separate 66-profile matrix currently has 55 comparable oracle results and all
55 pass the `max(10 J/kg, 5%)` criterion. Eleven BUFKIT profiles are explicitly
oracle-unresolved because `ecape-parcel-py` cannot obtain an LFC/EL; they are
not counted as passes. See [Rust ECAPE validation](ECAPE_RUST_VALIDATION.md) for
the measured error statistics and weak-instability regressions.

## Interactive spot-check

In a Rust-backed sounding window, choose **Profiles → Compare Rust vs
Python…**. The viewer snapshots the highlighted sounding and runs a legacy
Python `ConvectiveProfile` in a background thread only on demand. The table
shows Rust and Python values, absolute error, the allowed error from this
audit's tolerance schema, and the result. Differences appear first.

**Show Legacy Python** mounts the reference as a temporary, separate profile;
**Show Rust (fast)** returns to the original cached Rust profile. Neither
button mutates or replaces the source sounding. Comparisons are cached for the
unchanged profile and invalidated when an in-place map refresh replaces it.

MMP is shown as informational because the legacy value is undefined. Watch
rows labeled `MMP-normalized` use SHARPpy's legacy watch logic with the
deterministic Rust MMP substituted only for that classification, matching the
release auditor's normalization above.

This dialog is for inspecting one sounding. It deliberately does not replace
the complete corpus, fixed-seed stress matrix, array/trace checks, or the
independent ECAPE oracle described in this document.

## Reproduce

Build and install the native extension first. Then run the committed corpus:

```powershell
python -m sharpmod.tools.native_parity --json native-parity.json
```

Run the exact 100-case stress matrix:

```powershell
python -m sharpmod.tools.native_parity_fuzz --json native-parity-fuzz.json
```

Run the focused regression gates:

```powershell
python -m pytest -q `
  sharpmod/tests/test_native_parity_corpus.py `
  sharpmod/tests/test_native_parity_fuzz.py `
  sharpmod/tests/test_ecape_rust_parity.py
```

Both command-line audits exit nonzero for a backend/schema error, missing-state
mismatch, category mismatch, or out-of-tolerance numeric result. JSON reports
include the backend revisions; the stress report additionally records the
loaded native-extension SHA-256 and every generated case's perturbation
metadata.

## v0.3.2 numerical result

| Audit | Cases | Result | Audit build / final verification |
| --- | ---: | --- | --- |
| Committed real/synthetic corpus | 57 | 83,631 / 83,631 passed | Pre-final audit build; pinned engines unchanged |
| Fixed-seed randomized stress, discovery run | 100 | One case contained 3 failed comparisons | Pre-final discovery build; pinned engines unchanged |
| Corrected zero-depth fuzz regression | 1 | 1,455 / 1,455 passed | `76146F5F28859B01F774179F7B128D41AB4D77CAB07F5AD1169367E3EAA85877` |
| Affected canonical zero-depth regression | 1 | 1,473 / 1,473 passed | `76146F5F28859B01F774179F7B128D41AB4D77CAB07F5AD1169367E3EAA85877` |
| Final precipitation-only committed sweep | 51 | 714 / 714 passed | `76146F5F28859B01F774179F7B128D41AB4D77CAB07F5AD1169367E3EAA85877` |

The stress discovery result and the post-fix focused results are reported
separately so the table does not imply that either complete matrix was rerun
after a localized correction. The final release extension passed both affected
edge regressions, the precipitation-only sweep, the focused UI/native gates,
and the complete 580-test Python suite. Its SHA-256 is shown above.
