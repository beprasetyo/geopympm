# GeoPyMPM Stage 10A — uGIMP Numerical Validation

**Date:** 28 October 2026  
**Baseline:** Stage 10A uGIMP (repository archive `GeoPyMPM_Stage10A_uGIMP_28Oct2026.zip`)  
**Scope:** standard point MPM vs unchanged-domain GIMP (uGIMP) in 2D, structured grid, unit out-of-plane thickness, SI units, tension-positive Cauchy stress.

This validation adds **no solver or constitutive changes**. Its goal is to quantify how the new interpolation affects cell-crossing force assembly, to verify unchanged stress during rigid translation, and to avoid conflating interpolation improvements with coupled dynamic constitutive validation.

## Reproduce

From the repository root with the `geopympm` environment active:

```bash
python -m validation.stage10.validate_gimp
python -m pytest tests/test_stage10_gimp.py tests/test_stage10a_validation.py -q
```

Matplotlib is needed **only to generate PNG figures**; CSV/JSON and numerical tests work without it. Other CLI options: `--output`, `--ppc`, `--steps`, `--no-plots`. Results are written under `validation/results/stage10a/`. To keep the published numbers, use the default arguments.

## Equations

Point MPM evaluates the grid hat at the material-point center:

\[
N_{ip}=N_i(\mathbf x_p),\qquad \nabla N_{ip}=\nabla N_i(\mathbf x_p).
\]

The uGIMP implementation averages the shape function over the particle's fixed rectangular interpolation domain \(\Omega_p\):

\[
S_{ip}=\frac{1}{|\Omega_p|}\int_{\Omega_p}N_i(\mathbf x)\,d\Omega,
\qquad
\nabla S_{ip}=\frac{1}{|\Omega_p|}\int_{\Omega_p}\nabla N_i(\mathbf x)\,d\Omega.
\]

The nodal internal force is assembled in both modes by

\[
\mathbf f_i^{\mathrm{int}}=-\sum_p V_p\,\boldsymbol{\sigma}_p\,\nabla S_{ip},
\]

using \(S=N\) in point MPM. **The physical representative volume \(V_p\) and the interpolation domain \(|\Omega_p|\) are separate concepts.** When comparing to an exact continuum integral over a block, we deliberately set the rectangular GIMP domains equal to the physical quadrature tiles and ensure the tiles are contiguous without overlap. This is a *special benchmark construction*, not an assumption about every GeoPyMPM simulation.

## 10A-1 — Single-particle force discontinuity

A particle of constant \(\sigma_{xx}=100\,\mathrm{Pa}\), physical volume \(V=0.2\,\mathrm{m^3}\), and GIMP domain \(0.5\times 0.5\,\mathrm m\) is prescribed at 201 x-positions in \([1.75,2.25]\,\mathrm m\) with \(y=1.5\,\mathrm m\). Measure x-force on grid node (2,1) at \(x=2\,\mathrm m\). The analytical comparator is the exact integrated bilinear-grid hat gradient over that domain, multiplied by the particle's **physical** volume.

| Quantity | Standard MPM | uGIMP |
|---|---:|---:|
| Maximum adjacent position sample change in nodal force | 20 N | 0.1 N |
| RMS error vs exact domain integral | 5.759 N | ~2.16 × 10⁻¹⁵ N |

**Interpretation:** GIMP removes the finite force jump as the particle center crosses the grid line. Its 0.1 N adjacent-sample change is due to the finite sampling interval, **not a discontinuity**. The point-MPM result reflects the center-point quadrature approximation. This probe is a prescribed-position force-assembly diagnostic, not a stress or dynamic simulation.

File: `10A_single_particle.csv`, `10A_single_particle_force.png`.

## 10A-2 — Translating, uniformly prestressed block

The block occupies initially \(x\in[1.22,4.22]\) m, \(y\in[0,1]\) m; uniform \(\sigma_{xx}=120\) Pa, density 1200 kg/m³. The grid spacing is 1 × 1 m, with a sample node (5,0). Two particle rows and four particles per cell in x (24 particles total) tile the domain. Full-domain dimensions are \(1/4\) × \(1/2\) m for uGIMP.

Every grid velocity DOF is prescribed to (0.9,0) m/s to isolate **interpolation/force quadrature**; nodal internal forces do **not** drive the motion. The explicit solver advances 150 steps with \(\Delta t=0.01\) s. The reference nodal force is the analytical volume integral of the constant continuum stress against the grid hat gradient, using the moving physical block boundary (the same analytical comparator as Stage 6E).

| Quantity | Standard MPM | uGIMP |
|---|---:|---:|
| RMS internal-force error | **4.2008 N** | **2.99 × 10⁻¹² N** |
| Maximum force-increment noise vs reference | 14.46 N | 3.82 × 10⁻¹² N |
| Max particle stress drift | ~2.73 × 10⁻¹² Pa | ~3.41 × 10⁻¹² Pa |
| Max position error | ~5.06 × 10⁻¹⁴ m | ~5.06 × 10⁻¹⁴ m |

**Interpretation:** With rectangular domains that **exactly tile** the prestressed continuum, uGIMP force quadrature reproduces this piecewise-linear continuum force integral to roundoff precision; this is the expected mathematical result, **not universal evidence of machine-precision GIMP accuracy**. The constant-stress diagnostic is an invariant check: stress fluctuations should not arise in either method under rigid translation.

Files: `10A_moving_block.csv`, `10A_block_force.png`, `10A_block_force_error.png`.

## 10A-3 — Prescribed nonuniform-velocity stress evolution across a grid line

A particle starts at \(x=1.70\) m, with a 0.5 × 0.5 m interpolation domain, on a grid of 0.5 m spacing. All nodal velocity DOFs are prescribed to

\[
v_x(x_i)=v_0+a\sin(kx_i),\quad v_0=0.6\,\mathrm{m/s},\quad
 a=0.08\,\mathrm{m/s},\quad k=\pi/2\,\mathrm{m^{-1}},\quad v_y=0.
\]

Both implementations use E = 120 kPa, ν = 0.25, ρ = 1200 kg/m³, PIC, 160 steps of 0.005 s. They therefore cross x=2 m, and generate **nonzero evolving strain and stress** from grid velocity gradients. An independently integrated (RK4) **smooth continuum** trajectory \(\dot x=v_0+a\sin(kx)\) and its incremental isotropic elastic strain solution provide a reference:

\[
\sigma_{xx}^{\mathrm{ref}}(t)=M\ln\frac{v(x(t))}{v(x_0)},
\qquad M=\lambda+2G.
\]

The reference is the smooth sinusoidal continuum field, while the discrete algorithms interpolate **nodal samples** of that field. Thus differences include spatial discretization and particle advection errors; they are not pure cell-crossing noise.

| Quantity | Standard MPM | uGIMP |
|---|---:|---:|
| RMS stress difference vs smooth continuum | **565.7 Pa** | **598.0 Pa** |
| RMS stress-increment difference vs continuum | **7.03 Pa** | **7.09 Pa** |

**Interpretation:** In this prescribed-flow example, uGIMP is **slightly less accurate** by the chosen error measures. GIMP smoothing does not guarantee smaller stress error for every velocity profile and grid resolution. **This test is not force-driven:** prescribed velocities prevent elastic forces from affecting the trajectory.

Files: `10A_strain_driven_stress.csv`, `10A_stress_crossing.png`, `10A_stress_error.png`.

## 10A-4 — Force-driven plane-strain elastic standing wave

A smaller variant of Stage 6D with a 4 × 0.5 m bar, 24 × 2 cells, 192 particles, E=120 kPa, ν=0.25, ρ=1200 kg/m³, mode \(u(X,0)=A\sin(\pi X/L)\) with A=0.002 m and consistent initial stress. The end axial velocities and all transverse velocities are constrained, **interior axial velocities remain force-driven**. Two runs use identical parameters and initial conditions, except interpolation. \(\Delta t=0.001\) s and 300 steps. Compare \(\sigma_{xx}\) at original material coordinates to the small-strain analytical standing-wave field.

| Quantity | Standard MPM | uGIMP |
|---|---:|---:|
| Maximum relative L2 error, σxx | **0.06046** | **0.06046** |
| Mass-relative error | 0 | 0 |

These equal results are **expected**: each GIMP interpolation domain here is sufficiently small to remain within a single cell, where averaging a linear grid hat yields the point-MPM shape/gradient for this configuration. Accordingly, this is a **regression/scheme-compatibility verification**, not a demonstration of uGIMP superiority under stress waves or a cell-crossing test. The 0.08 stress-error acceptance level is the Stage 6D order-of-magnitude criterion; this setup is a shorter, coarser variant rather than the Stage 6D standard test.

Files: `10A_force_driven_wave.csv`, `10A_wave_stress_error.png`.

## 10A-5 — Particle-resolution sensitivity

Repeat Test 10A-2 with 2, 4 and 8 particles/cell in the x direction, holding grid, continuum shape, stress, velocity, and time step fixed. Two particle rows are always used in y.

| Particles/cell (x) | MPM RMS force error (N) | uGIMP RMS force error (N) |
|---:|---:|---:|
| 2 | 8.642 | 2.38 × 10⁻¹² |
| 4 | 4.201 | 2.99 × 10⁻¹² |
| 8 | 2.150 | 2.75 × 10⁻¹² |

MPM error decreases with particle refinement. uGIMP is exact to floating-point precision for **this specially tiled rectangular domain / constant stress / integrated hat configuration**; this does not establish general convergence rates.

File: `10A_refinement.csv`, `10A_refinement.png`.

## Acceptance and limitations

- `12/12` automated numerical benchmark checks passed with the default configuration; Stage 10 unit/integration tests are supplied in `tests/test_stage10a_validation.py`. The test checks do **not** assert uGIMP superiority in the stress-deformation case.
- Point-MPM vs GIMP interpolation changes are not accompanied by changes to the old Stage 1–9 algorithms or the constitutive model.
- uGIMP domain size remains **unchanged** even while the deformation gradient evolves. This limits large-deformation interpretability.
- None of these experiments validates Mohr–Coulomb plasticity, runout, contact, irregular FEM-to-MPM mapping, CPDI, or constitutive response under severe domain distortion.
- For a complete stress-noise validation under *force-driven* cell crossing, a separate well-defined analytical/manufactured dynamic solution (or a carefully converged independent reference) is needed. Prescribed translation alone cannot produce constitutive stress oscillations and must not be reported as proof of their elimination.

**Stage 10A conclusion:** uGIMP interpolation and integration with the solver are *verified against controlled cell-crossing force and compatible elastic tests*. Stage 10A **does not establish** that GIMP is more accurate than standard MPM for every dynamic stress problem. Deforming-domain GIMP/CPDI should be evaluated separately as Stage 10B after wider benchmarks.
