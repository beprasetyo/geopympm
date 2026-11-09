# GeoPyMPM Stage 10E-B2 — General 2D force-driven shear and rotation

**Date:** 8 November 2026  
**Base:** Stage 10E-B, full source dated 8 November 2026.  
**Status:** Initial mixed-wave verification implemented; nonlinear general-2D validation remains open.

## 1. Why a new benchmark?

Stage 10E-B constrained the transverse nodal velocity (`v_y=0`), so its exact
Jaumann shear-wave reduction was effectively one dimensional despite being
embedded in a 2D grid. Stage 10E-B2 removes **all nodal velocity constraints**.
Both `v_x` and `v_y` vary with x and y; internal forces develop in **both**
coordinate directions. It is not an externally loaded cantilever or traction
simulation. It is an initial-value elastodynamic benchmark.

The solver, Numba kernels, grid, particle data structures, and constitutive
code are **unchanged**. The new code lives entirely in `validation/`, `tests/`,
`examples/` and `docs/`.

## 2. Independent analytical linear-elastic reference

For linearized, homogeneous, isotropic plane-strain elastodynamics,

\[
 \rho_0\ddot{\mathbf u} = \nabla\cdot\boldsymbol\sigma,\qquad
 \boldsymbol\sigma=\lambda\operatorname{tr}(\boldsymbol\epsilon)\mathbf I
            +2\mu\boldsymbol\epsilon,\qquad
 \boldsymbol\epsilon=\frac12(\nabla\mathbf u+\nabla\mathbf u^T).
\]

The characteristic P- and S-wave velocities are

\[
 c_P=\sqrt{(\lambda+2\mu)/\rho_0},\qquad
 c_S=\sqrt{\mu/\rho_0}.
\]

Use two oblique **different** wave vectors

\[
 \mathbf q_P=(\pi/2,\pi/2),\quad
 \mathbf q_S=(\pi/2,\pi),\quad
 \mathbf e_P=\mathbf q_P/|\mathbf q_P|,\quad
 \mathbf e_S=(-q_{Sy},q_{Sx})/|\mathbf q_S|.
\]

The independent reference is the **superposition**

\[
 \mathbf u(\mathbf X,t)=\sum_{r=P,S}
 \frac{A_r}{\omega_r}\sin(\omega_r t)
 \cos(\mathbf q_r\cdot\mathbf X)\,\mathbf e_r,
 \qquad \omega_r=c_r|\mathbf q_r|,
\]

\[
 \mathbf v(\mathbf X,t)=\sum_r A_r\cos(\omega_r t)
 \cos(\mathbf q_r\cdot\mathbf X)\,\mathbf e_r,
\]

\[
 \nabla\mathbf u=-\sum_r\frac{A_r}{\omega_r}\sin(\omega_r t)
 \sin(\mathbf q_r\cdot\mathbf X)\,\mathbf e_r\otimes\mathbf q_r.
\]

The reference stress and F are evaluated directly from these expressions;
no GeoPyMPM solver or interpolation routine is used to compute them. At
`t=0` displacement and stress are zero but both components of initial
velocity are nonzero and spatially varying. Subsequent particle acceleration
comes from **grid internal forces**. Neither x nor y velocity is prescribed.

**Essential limitation:** These are exact solutions of the **linearized**
continuum equations, whereas the MPM simulation integrates finite-deformation
kinematics with a **Jaumann hypoelastic rate law**. At finite amplitude,
reference error includes continuum-model linearization, not just numerical
interpolation error. A small-amplitude reference and separate strong-amplitude
stress test are therefore reported; their accuracies must not be conflated.

## 3. Discretization and evaluation

| Parameter | Value |
|---|---:|
| E, Poisson ratio | 1200 Pa, 0.25 |
| Density | 1200 kg/m³ |
| Particle block | [2,10] × [2,10] m (unit out-of-plane thickness) |
| Background grid | [0,12] × [0,12] m |
| Reference core window | [4.5,7.5] × [4.5,7.5] m |
| Default grid | 0.25 m, 2 PPC per axis |
| Fixed time step | 0.002 s |
| Total time | 0.30 s (150 steps) |
| Base velocity amplitudes | A_P=0.14, A_S=0.18 m/s, scaled by 0.25 |
| Transfer / stress / F / advection | FLIP, corotational, exponential, affine-exponential |
| Boundary conditions | NO prescribed nodal x or y velocities |

The particle block has free outer faces. The interior comparison window is
separated from those faces so that at the chosen short final time it lies
outside the physical domain of influence of free-face disturbances; the
largest P-wave speed is about 1.095 m/s. Edge errors elsewhere are not
suppressed by the solver and are not included in the reference norm.

Use `stress_relative_l2`, `shear_stress_relative_l2`, RMS F, RMS polar
angle, and velocity error. Track mass, both momentum components, nonzero
internal forces, particle-cell crossings and `det(F)>0`.

## 4. Measured numerical results

### Base: small-amplitude mixed P/S waves (h=0.25 m)

| Interpolation | Final relative stress L2 error |
|---|---:|
| Standard MPM | 17.852% |
| uGIMP | 17.187% |
| cpGIMP | 17.188% |
| CPDI1 | 17.217% |

These are not sub-percent errors. Domain methods have modestly smaller
errors than standard MPM in this case, but CPDI1 is **not** the most accurate.
The error is dominated by spatial-resolution effects at these wavelengths.

### CPDI1 spatial refinement (same dt = 0.002 s)

| Grid spacing | Stress relative L2 error |
|---|---:|
| 1.00 m | 73.692% |
| 0.50 m | 37.903% |
| 0.25 m | 17.217% |

The error decreases monotonically on this interval. A smaller grid was also
checked separately: h=0.125 m gave approximately 7.98% error, but a complete
asymptotic convergence study, time-step refinement and finite-strain reference
remain necessary.

### Separate finite-amplitude / strongly nonlinear stress test

At h=0.50 m and amplitude multiplier **3.0**, CPDI1 reached:

| Quantity | Value |
|---|---:|
| Maximum local polar angle | 9.743° |
| Max \|F_xy\| | 0.389 |
| Max \|F_yx\| | 0.154 |
| Minimum det F | 0.736 |
| Particle cell crossing events | 46 |
| Maximum mismatch between formerly shared CPDI edges | 0.020 m |

Both momentum components remain unconstrained, and both internal force
components are nonzero. **Do not use the linear-wave relative L2 metric to
claim an exact accuracy for this strong-amplitude case**. CPDI1 cannot
perfectly preserve neighbor-domain compatibility under non-affine deformation.
The edge mismatch compares endpoints of two previously shared straight edges;
it does **not** measure overlap area or detect contact.

### Nodal-acceleration diagnostic (all methods, base h=0.25 m)

The solver remains unmodified. The reported maximum is the 2D magnitude
`||a_i||`, and the *mass-supported* maximum excludes nodes below `10^-8`
times the median positive nodal mass **for diagnostic reporting only**.

| Method | Maximum raw nodal acceleration (m/s²) | At nodal mass (kg) | Max mass-supported acceleration (m/s²) |
|---|---:|---:|---:|
| MPM | 0.167 | 37.62 | 0.167 |
| uGIMP | 1.093 × 10⁴ | 1.27 × 10⁻¹² | 71.59 |
| cpGIMP | 1.212 × 10⁵ | 3.93 × 10⁻¹⁵ | 197.32 |
| CPDI1 | 0.434 | 6.88 × 10⁻⁴ | 0.434 |

This reveals new near-massless-node acceleration spikes for uGIMP/cpGIMP,
consistent with previous Stage 10E diagnostics. **We have not measured the
P2G/G2P impulse contribution of those specific nodes in this new experiment.**
Large raw nodal acceleration is not itself evidence of an instability of the
particle solution, but its harmlessness cannot be asserted without that
additional check. No nodal force/mass cutoff was added to the solver.

## 5. Validation scope and unresolved questions

- Verified: independent 2D linear-wave solution; 2D forces and movement;
  nonzero shear / rotation; conservation checks; CPDI reference-error
  reduction on grid refinement; positive volume; affine CPDI shared-edge
  regression; non-affine domain mismatch measured.
- Not verified: force-driven large general 2D rotations with an **independent
  finite-strain reference**, traction boundary loading, bending, nonuniform
  density, plasticity, and contact. The strong-amplitude test is an exploratory
  stability and geometric-consistency test, not full constitutive validation.
- The large standard MPM and CPDI errors on coarse grids are deliberately
  reported; passing unit tests does not establish engineering accuracy.
- The `near-massless` nodal diagnostics from Stage 10E-A.1 remain available.
  No mass cutoff is introduced by this Stage 10E-B2 patch.

## 6. Reproducing

```bash
source .venv/bin/activate
python examples/16_force_driven_general_2d.py
python -m validation.stage10eb2.general_2d_waves --refinement --amplitude-sweep
python -m pytest tests/test_stage10eb2_general_2d.py -q
python -m pytest -q
```

Results: `validation/results/stage10eb2/10EB2_summary.json`, `10EB2_cases.csv`,
`10EB2_history.csv`, `10EB2_summary.png`, and `10EB2_shear_field.png`.

Stage 10E-B2 establishes a controlled two-component baseline for Stage 10F;
large-deformation validation of unrestricted 2D traction/bending problems
remains an open milestone.

## 7. Regression status in the development environment

The 524 repository tests passed when split into four disjoint sets:

- Core particle/grid/explicit/kinematic regression: **296 passed**.
- Stage 10 GIMP/cpGIMP/CPDI and Stage 10E-B2 focused regression: **105 passed**.
- Stage 6A–E plus Stage 10A and Stage 10E-A regression: **96 passed**.
- Stage 6F energy/convergence regression: **27 passed**.

A single all-in-one `python -m pytest -q` run reached the execution timeout;
no claim is made that this particular uninterrupted run completed here.
It can be repeated locally to independently confirm the complete suite.
