# GeoPyMPM Stage 10A — uGIMP domain / PIC–FLIP / particle-resolution sensitivity

**Baseline:** `GeoPyMPM_Stage10A_ForceDriven_Validated_28Oct2026.zip`  
**Date:** 28 October 2026  
**Scope:** Numerical sensitivity of the existing explicit 2D MPM/uGIMP solver. This stage **does not change** numerical solver kernels or material routines.

## 1. Purpose

The prior Stage 10A force-driven Gaussian travelling-wave benchmark established:

1. Analytical stress reference from infinitesimal elastic d'Alembert theory in fixed material coordinates `X`.
2. Actual stress -> nodal internal force -> momentum -> particle velocity/position -> strain/stress feedback.
3. Particle crossing of fixed background cells under `U=1 m/s` translation.
4. Small conservation errors but significant numerical differences between boosted (`U=1`) and rest (`U=0`) frames.

This study checks whether the error is sensitive to domain averaging width, particle-to-grid resolution, and PIC/FLIP blending. No expected superiority of uGIMP was assumed.

## 2. Formulation and parameter definitions

The interpolated particle domain for **uGIMP** is rectangular, axis-aligned and unchanged by deformation:

`domain_size_x = domain_factor * (cell_dx / ppc_x)`

`domain_size_y = cell_dy / ppc_y` (unchanged throughout this sensitivity study).

The PIC–FLIP blending follows the solver's original convention:

`v_new = (1 - pic_fraction) * v_FLIP + pic_fraction * v_PIC`

- `pic_fraction=0`: pure FLIP.
- `pic_fraction=0.1`: 90% FLIP + 10% PIC.
- `pic_fraction=1`: pure PIC.

**Caution:** `domain_factor=1` nominally tiles the reference material region before the small initial elastic deformation. `domain_factor=0.5` leaves gaps, while `domain_factor=2` creates overlap between nominal interpolation domains. These are *interpolation-support sensitivity* cases, not interchangeable physical particle discretizations. Particle mass and volume remain unchanged as the GIMP averaging domain changes. In particular, do not choose factor=2 solely because it reduces a frame-dependence metric.

## 3. Benchmark configuration

| Parameter | Setting |
|---|---|
| Elastic model | Linear elastic, plane strain |
| Young's modulus | 1200 Pa |
| Poisson's ratio | 0.25 |
| Density | 1200 kg/m³ |
| Analytical wave | Small-strain, right-going Gaussian pulse on an infinite homogeneous bar |
| Gaussian width / displacement amplitude | 0.42 m / 0.002 m |
| Cell size | `dx=0.25 m`, `dy=0.25 m` |
| Particle resolution | `ppc_x = 2, 4, 8`, `ppc_y = 2` |
| Particle domain x factor | `0.5, 1, 2` for uGIMP |
| PIC fraction | `0, 0.1, 1` |
| Frame translation | `U=0` and `U=1 m/s` |
| Time step / total steps | 0.002 s / 450 steps (0.9 s) |
| Boundary condition | Only zero transverse nodal velocity prescribed, axial dynamics free |
| Total cases | 9 MPM + 27 uGIMP = 36 configurations, each in 2 frames = 72 dynamic simulations |

The **same initial physical particle state** is used within each reference-frame pair; changing particle count changes the sampling resolution. MPM is recomputed once per PPC/PIC because its interpolant does not depend on GIMP domain width.

## 4. Diagnostics

Let `sigma_num(X,t)` be numerical particle stress and `sigma_ref(X,t)` the analytical stress at the same particle material labels.

- **Time-RMS relative stress error:** time-root-mean-square of `RMS(sigma_num-sigma_ref) / RMS(sigma_ref(X,0))`, sampled every 5 steps.
- **Galilean-frame difference at the final time:** `RMS(sigma_boost(X,T)-sigma_rest(X,T)) / RMS(sigma_ref(X,T))`; `X` labels remain identical between runs.
- **Crossing counts** are based on the change in `floor(x_particle/dx)` at successive steps.
- **Crossing stress-error increment RMS** is collected only where the analytical stress is at least 5% of its theoretical peak; crossing and non-crossing subsets are *not* matched controls and should not be used to claim a causal improvement.
- **Mass and horizontal momentum conservation** are checked for each run.

A smaller Galilean-frame difference does **not** imply better reproduction of the analytical wave. Numerical smoothing or phase/amplitude distortion can make boosted and rest-frame results agree while both are incorrect.

## 5. Results — reference PPC=4

### 5.1 Domain/PIC sensitivity

The reported values below are from the saved full-run CSV (`validation/results/stage10a_sensitivity/10A_SENS_all_cases.csv`).

| Interpolation | PIC fraction | Domain factor | Time-RMS stress error | Final Galilean-frame difference |
|---|---:|---:|---:|---:|
| MPM | 0 | n/a | 24.72% | 39.20% |
| uGIMP | 0 | 0.5 | 24.85% | 39.31% |
| uGIMP | 0 | **1.0 (nominal)** | **25.28%** | **38.64%** |
| uGIMP | 0 | 2.0 | 27.05% | 19.35% |
| MPM | 0.1 | n/a | 67.93% | 17.46% |
| uGIMP | 0.1 | **1.0** | **69.00%** | **16.83%** |
| uGIMP | 0.1 | 2.0 | 71.76% | 7.72% |
| MPM | 1.0 | n/a | 111.02% | 3.66% |
| uGIMP | 1.0 | **1.0** | **111.41%** | **3.48%** |
| uGIMP | 1.0 | 2.0 | 112.28% | 1.56% |

**Interpretation:** Increasing GIMP averaging width improves the measured boosted-vs-rest agreement in this benchmark, but simultaneously worsens agreement with independent analytical stress. Increasing PIC fraction similarly makes boosted/rest outcomes closer yet produces much greater analytical stress error. Therefore PIC damping or overextended domains must not be declared physically optimal by the invariance metric alone.

### 5.2 Nominal domain, pure FLIP, varying particle resolution

| PPC in x | MPM stress error | uGIMP stress error | MPM Galilean difference | uGIMP Galilean difference |
|---:|---:|---:|---:|---:|
| 2 | 24.72% | 27.02% | 35.14% | 35.68% |
| 4 | 24.72% | 25.28% | 39.20% | 38.64% |
| 8 | 24.73% | 24.86% | 40.24% | 39.29% |

Raising PPC at the *same cell size* does not appreciably reduce MPM's analytical stress error in this test. uGIMP approaches MPM as its nominal domain becomes smaller relative to a fixed grid cell. The previously completed grid-refinement experiment (reducing `dx` from 0.5 to 0.125 m) produced a much stronger reduction in analytical stress error, which points to grid resolution/integration as the more relevant next investigation for this Gaussian-wave setup.

### 5.3 Verification status

- 36 parameter configurations and 72 frame-paired simulations completed.
- All 36 configurations showed actual particle cell crossing with `U=1 m/s`.
- All runs passed mass and horizontal momentum conservation thresholds (< `1e-10` and `<1e-8`, respectively), and stress metrics were finite.
- All Stage 10 focused tests: **42 passed**.
- Full GeoPyMPM test suite, including newly added tests: **425 passed**.

## 6. Conclusions and recommendations

1. **Preserve** nominal nonoverlapping GIMP support (`domain_factor=1`) as the Stage 10A baseline. Use modified factors only for sensitivity studies with explicitly acknowledged gaps/overlap.
2. For this *force-driven elastic wave*, **pure FLIP** (`pic_fraction=0`) gives a substantially smaller analytical stress error than PIC fractions of 0.1 or 1.0. Do not generalize this to all dynamic geomechanics cases, where FLIP noise and PIC damping can trade off differently.
3. Increasing particle count at fixed grid spacing provides limited improvement for this benchmark; refining grid spacing had a much larger effect in the earlier Stage 10A experiment.
4. **uGIMP mitigates discontinuous force transfer**, as shown in the separate Stage 10A force-continuity validation. This experiment does not establish universally lower stress errors or exact Galilean invariance.
5. Do **not** infer a solver bug solely from a nonzero boosted/rest difference; the discrete grid and transfer algorithm can introduce frame dependence. A focused algorithm/dispersion analysis with grid refinement and a finer near-FLIP PIC scan would be valuable before claiming robust predictive improvements.
6. Stage 10B (deforming domains / cpGIMP or CPDI) can be investigated next as a *new algorithmic capability*, but it must not be described as fixing all Stage 10A stress errors automatically.

## 7. Reproduction

From the repository root with `.venv` active and package installed in editable mode:

```bash
# Full sensitivity matrix (450 time steps; 72 dynamic simulations):
python -m validation.stage10.sensitivity_gimp

# Short diagnostic / smoke test (not the reported physical results):
python -m validation.stage10.sensitivity_gimp --quick

# Investigate a single case without changing source:
python -m validation.stage10.force_driven_crossing \
  --ppc-x 4 --domain-factor 2 --pic-fraction 0.1 --no-plot

# Stage 10 regression:
python -m pytest tests/test_stage10_gimp.py \
  tests/test_stage10a_validation.py \
  tests/test_stage10a_force_driven.py \
  tests/test_stage10a_sensitivity.py -q

# Repository-wide regression:
python -m pytest -q
```

Expected output folder: `validation/results/stage10a_sensitivity/`.

Files produced:

- `10A_SENS_all_cases.csv` — complete 36-case machine-readable results.
- `10A_SENS_summary.json` — setup, invariants, reference scenarios, plots.
- `10A_SENS_stress_vs_domain.png` — stress accuracy vs GIMP domain factor.
- `10A_SENS_galilean_vs_pic.png` — boosted-vs-rest sensitivity vs PIC blend.
- `10A_SENS_stress_vs_ppc.png` — stress error vs particle resolution.

### Modified/new files

- **Modified** `validation/stage10/force_driven_crossing.py` — optional `domain_factor` and `pic_fraction`, defaults preserve the prior benchmark.
- **New** `validation/stage10/sensitivity_gimp.py` — full factorial runner, diagnostics, CSV/JSON/plots.
- **New** `tests/test_stage10a_sensitivity.py` — numerical and input validation regression tests.
- **New** this technical report and the generated output directory.

No changes were made to `src/geopympm/` numerical solver modules.
