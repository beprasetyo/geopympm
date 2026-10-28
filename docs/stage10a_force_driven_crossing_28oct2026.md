# GeoPyMPM Stage 10A-FD — Force-driven Cell-crossing Stress Benchmark

**Date:** 28 October 2026  
**Baseline:** Stage 10A validated uGIMP release (28 October 2026).  
**Implementation:** `validation/stage10/force_driven_crossing.py`  
**Status:** Implemented and targeted tests passed. Numerical **benchmark**, not a blanket claim of uGIMP superiority or final large-deformation validation.

## 1. Why another benchmark?

Previous Stage 10A experiments demonstrated that uGIMP can regularize nodal internal forces when interpolation domains cross grid lines. The prescribed-translation and prescribed-velocity cases did *not* establish that uGIMP reduces stress error in a **dynamically coupled** computation. The prior standing-wave calculation coupled force to motion but did not require substantial cell crossing.

This experiment combines (a) a real stress-gradient-induced force and acceleration, (b) particle crossing of Eulerian grid lines, and (c) an **analytic stress field not computed by MPM or GIMP**. No axial nodal velocity is prescribed. Both approaches use the identical geometry, particle masses, constitutive properties, initial stress/velocity and time integration.

## 2. Independent analytical reference

Consider a small-strain, homogeneous elastic bar constrained to plane-strain motion in the transverse direction. Let material coordinate \(X\) label a continuum point. Define a right-going Gaussian displacement pulse

\[
 g(z)=\exp\left(-\frac{z^2}{2w^2}\right),\qquad
 u(X,t)=U t + A g(X-X_c-ct).
\]

Here \(U\) is an optional **rigid translational drift**, \(A\) is the pulse amplitude and \(w\) its width. The linear elastic modulus for one-dimensional longitudinal strain under plane strain is

\[
 M=\lambda+2G=\frac{E(1-\nu)}{(1+\nu)(1-2\nu)},\quad
 c=\sqrt{M/\rho_0}.
\]

The **analytic material velocity and stress** are

\[
 v_x(X,t)=U-c A g'(X-X_c-ct),\qquad
 \sigma_{xx}^{ref}(X,t)=M A g'(X-X_c-ct),
\]

with \(g'(z)=-z g(z)/w^2\). The expected current coordinate is

\[
 x^{ref}(X,t)=X+U t+A g(X-X_c-ct).
\]

In the linear plane-strain setting, \(\sigma_{yy}=\sigma_{zz}=\lambda A g'\). This solution satisfies the infinite-domain **linear** one-dimensional elastic wave equation and is computed with closed-form expressions (`analytical`), not with an MPM solver. It is an **asymptotic reference** for the updated-configuration, finite-strain variables in GeoPyMPM; its finite-amplitude discrepancy is of higher order in \(A/w\), which is kept small. The wave is initialized far enough from the finite specimen's ends so reflected waves are negligible during the chosen time interval.

## 3. Implementation and boundaries

SI units; 2D plane strain with a 0.5 m tall bar (unit out-of-plane thickness):

| Parameter | Value |
|---|---:|
| Bar material-coordinate x interval | 1–11 m |
| Background grid x interval | 0–14 m |
| Grid spacing (baseline) | 0.25 × 0.25 m |
| Material | E = 1,200 Pa, ν = 0.25, ρ = 1,200 kg/m³ |
| Plane-strain axial modulus M | 1,440 Pa |
| Analytic longitudinal wave speed | 1.095445 m/s |
| Initial Gaussian center Xc | 5 m |
| Pulse width w and amplitude A | 0.42 m and 0.002 m |
| Velocity drift U | 0 and 1 m/s |
| Particles per cell | 4 in x × 2 in y (640 total) |
| GIMP full domain width | 0.0625 × 0.125 m, unchanged in time |
| Time increment and steps | 0.002 s × 450 = 0.9 s |
| PIC fraction | 0 (FLIP) |
| Stress update / deformation update / advection | Stage 10 default: incremental / Euler / Euler |
| Axial velocity boundary conditions | **None** |
| Transverse velocity boundary conditions | vy = 0 at all grid nodes |

The particles are initialized at the analytically displaced coordinates; initial \(\sigma_{xx}\), transverse stresses, strain, deformation gradient, mass and current volume are consistent to first order with the analytic field. For both interpolation modes, the existing `ExplicitMPMSolver.step()` computes P2G, forces, nodal accelerations, particle deformation and stress updates, then G2P. There is **no prescribed x-velocity**, gravity or externally imposed x-force.

The run with \(U=0\) acts as an unboosted control. Both modes also run at \(U=1\) m/s, giving genuine particle cell crossing. The benchmark evaluates stress against the **fixed material label** \(X\), rather than spatially comparing particles that have moved to different coordinates.

## 4. Reproduction

From the repository root, with editable install active:

```bash
python -m validation.stage10.force_driven_crossing --refinement
python -m pytest tests/test_stage10a_force_driven.py -q
python -m pytest tests/test_stage10_gimp.py tests/test_stage10a_validation.py tests/test_stage10a_force_driven.py -q
```

The default command (`--refinement`) writes:

- `validation/results/stage10a_force_driven/10A_FD_summary.json`: full configuration, checks and metrics.
- `10A_FD_timeseries.csv`: time-dependent L2 stress/velocity/position errors and conservation indicators.
- `10A_FD_stress_snapshots.csv`: stress per selected fixed material label at three times.
- `10A_FD_grid_refinement.csv`: refinement comparison of 0.5, 0.25 and 0.125 m cell sizes.
- `10A_FD_stress_error.png`, `10A_FD_stress_profile.png`, `10A_FD_grid_refinement.png`: figures.

Matplotlib is optional for plots; JSON and CSV require only solver dependencies. A shorter smoke test can use `--steps 45 --no-plot`, but does **not** reproduce the published numbers.

## 5. Results — actual source execution

For a fixed 0.25 × 0.25 m grid with 640 particles:

| Case | Number of particle x-cell crossing events | Time-RMS relative L2 stress error | Final relative L2 stress error |
|---|---:|---:|---:|
| MPM, drift 0 m/s | 0 | 0.336823 | 0.468293 |
| uGIMP, drift 0 m/s | 0 | 0.334866 | 0.468099 |
| **MPM, drift 1 m/s** | **2,240** | **0.247238** | **0.412980** |
| **uGIMP, drift 1 m/s** | **2,240** | **0.252814** | **0.423649** |

The final absolute stress RMS errors for drift 1 m/s were **0.546348 Pa (MPM)** and **0.560463 Pa (uGIMP)**. Normalization for relative error uses the initial analytic RMS particle stress, identical for MPM and uGIMP at a given resolution.

### Grid refinement (drift 1 m/s)

| Δx (m) | MPM time-RMS rel L2 | uGIMP time-RMS rel L2 |
|---:|---:|---:|
| 0.500 | 0.627636 | 0.635821 |
| 0.250 | 0.247238 | 0.252814 |
| 0.125 | 0.069439 | 0.071379 |

Grid refinement reduces continuum stress error substantially for both formulations, confirming that the analytical comparator is relevant and numerical dispersion decreases as the mesh is refined. This is a **grid-convergence trend**, not an observed uGIMP advantage at these parameters.

### Force-feedback, conservation and relative drift

All seven runtime benchmark checks in `10A_FD_summary.json` are true: force feedback is active, there is no prescribed axial velocity, particles cross cells in the drifting case, mass is conserved, x-momentum is conserved, stress values are finite, and the refined grid reduces error relative to coarse grid. The worst normalized x-momentum deviations are below 1.1 × 10⁻¹⁵ for drift=1, and mass deviation is zero to displayed floating-point precision.

The final **boosted-vs-unboosted** stress difference (relative to analytic initial RMS stress) is 0.391967 for MPM and 0.386385 for uGIMP. This is a separate, imperfect measure of sensitivity to bulk drift and numerical wave dispersion; do **not** confuse it with the analytic stress error or assert Galilean invariance from this result.

For stress-active particles at cell crossing, the RMS temporal *stress-error increment* is approximately **0.00479 Pa (MPM)** and **0.00629 Pa (uGIMP)**; the comparison is observational and is influenced by stress location and local wave phase. It is not an independent causal demonstration of stress noise reduction.

## 6. Interpretation, scope and next steps

1. **Verified numerically:** P2G/grid force/G2P and stress integration are coupled, and the same moving particles genuinely cross background-grid cells; there is an independent (linear continuum) stress reference.
2. **Verified numerically:** mass and horizontal momentum are conserved (within numerical tolerance), and both methods converge under x-grid refinement.
3. **Not established:** GIMP superiority in dynamic stress accuracy. On this case uGIMP has slightly **higher** RMS stress error, despite the demonstrated smoothing of internal forces in earlier Stage 10A controlled tests.
4. **Important limitations:** small-amplitude *linear* reference vs current updated-configuration solver; free-surface truncation is negligible here but not exactly infinite; one Gaussian width, one PIC/FLIP blend, one stress/kinematic update choice, y dimension held fixed, no plasticity, contact or slope runout. 2D y-component is constrained, not a multidimensional wave benchmark.
5. **Before Stage 10B:** further vary grid spacing, particle-domain width and PIC/FLIP blending, and consider independent published material-point benchmarks for stress noise and energy. Avoid treating this one numerical result as a general endorsement or rejection of GIMP.

**No solver implementation was modified for this benchmark.** The patch adds an independent validation program, tests, generated results and this report.
