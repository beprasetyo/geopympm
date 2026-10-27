# GeoPyMPM — Stage 6A–6C verification

**Scope:** Stage-5 USL-like, structured 2D Quad4, linear elastic, PIC/FLIP.
**Implementation:** `validation/stage6/benchmarks.py` and `run_all.py`.
**No changes to existing solver, materials, interpolation, or numerical kernels.**

## How to run

From repository root with editable package installed:

```bash
source .venv/bin/activate
python -m pytest tests/ -v
python -m validation.stage6.run_all --output validation/results/stage6
```

Produces `summary.json`, `6A_history.csv`, `6B_history.csv`, and
`6C_history.csv`. CSV records provide scalar errors versus time.
Optional plots (requires `matplotlib`):

```bash
python -m pip install matplotlib
python -m validation.stage6.run_all --plots
```

Results are written to `validation/results/stage6/` by default. Add
`validation/results/` to `.gitignore` if generated reports should not be
committed. See `summary.json` for pass/fail, max errors, and thresholds.

## 6A — Multi-particle rigid translation

- 24 particles with common initial velocity `(1.1, -0.35)` m/s.
- Gravity and initial stress are zero; elastic material, no BC.
- Uniform particle velocity is expected to remain uniform as particles
  change cells. Rigid body motion should produce zero velocity gradient,
  stress and strain, and `F=I`.
- Reference: `x(t)=x0+v0*t`; `v(t)=v0`.
- Checks: max position/velocity error, stress, deformation-gradient drift,
  total mass/momentum error, plus at least one cell crossing.
- Runs 50 steps at `dt=0.01 s`. PIC, FLIP and blending are tested in pytest.

**Interpretation:** This is a zero-stress kinematic and transfer check, not a
force-equilibrium test under applied loading.

## 6B — Uniform gravitational acceleration (free fall)

- Nine particles with common initial velocity `(0.2, 0.25)` m/s.
- Gravity `(0, -9.81)` m/s²; no constrained nodes or initial stress.
- Continuous exact reference:

  `v(t)=v0+g*t`, `x(t)=x0+v0*t+0.5*g*t²`.

- Important: Stage 5 advects with the **updated** nodal velocity. The exact
  reference for this *discrete integrator*, after `k` steps of `dt`, is:

  `x_k=x0+v0*k*dt + (g/2)*dt²*k*(k+1)`.

- Therefore continuous and discrete positions differ by `g*T*dt/2` at time
  `T=k*dt`. This is the known **first-order** global position error, not a
  failure of the numerical method to implement its stated algorithm.
- Study `dt=[0.02, 0.01, 0.005]` at fixed `T=0.2 s`. The observed
  convergence order is calculated as

  `log(e_coarse/e_fine) / log(dt_coarse/dt_fine)`.

**Interpretation:** Tests uniform body force and time integration; not wave
propagation or energy conservation under arbitrary loading.

## 6C — Homogeneous affine velocity / elastic kinematic patch

- Grid `(4,3)` cells, `dx=dy=1 m`, 2×2 particles per cell (48 total),
  `rho=1800 kg/m³`, `E=20 MPa`, `nu=0.25`, plane strain.
- **Every nodal velocity DOF is prescribed** to the analytic affine field:

  `v(x) = A x + b`, with

  ```text
  A = [[ 0.20,  0.15],
       [ 0.05, -0.30]] s^-1
  b = [0.04, -0.03] m/s
  ```

- Quad4 bilinear interpolation should exactly reproduce affine velocity
  and its constant gradient `L=A` at particle locations. The scheme's
  reference update is `F_{k+1} = (I+dt*A) F_k` and
  `V_k=det(F_k)*V0`, with `dt=0.001 s` and default 5 steps.
- Accumulated engineering strain reference per time step:

  `[A_xx, A_yy, 0, A_xy+A_yx, 0, 0] * dt`.

- Plane-strain incremental elastic stress per time step:

  ```text
  sigma_xx += dt*((lambda+2G)*A_xx + lambda*A_yy)
  sigma_yy += dt*(lambda*A_xx + (lambda+2G)*A_yy)
  sigma_zz += dt*lambda*(A_xx + A_yy)
  sigma_xy += dt*G*(A_xy + A_yx)
  ```

  with `lambda=E*nu/((1+nu)*(1-2nu))`, `G=E/(2*(1+nu))`.

- The independent discrete reference for particle advection is
  `x_{k+1}=(I+dt*A) x_k + dt*b`.
- Checks: max error in particle stress (including nonzero `sigma_zz` and
  `sigma_xy`), accumulated strain, deformation gradient, volume and position.

**Critical limitation:** All velocity degrees of freedom are imposed,
so computed stresses do not drive the prescribed deformation. This is
*not* an elastic force-equilibrium patch test, traction-loading benchmark,
experimental validation, or proof of dynamic solver stability. A real
mechanical patch test with traction/force balance and boundary reactions
is still required when the BC system is extended.

## Acceptance criteria

| Case | Quantity | Maximum error |
|---|---|---:|
| 6A | Position, velocity | `1e-10` in respective SI units |
| 6A | Stress | `1e-6 Pa` |
| 6A | Deformation gradient | `1e-10` |
| 6A | Total mass | `1e-9 kg` |
| 6A | Total momentum | `1e-8 kg m/s` |
| 6B | Velocity and discrete Euler position | `1e-10` |
| 6B | Stress | `1e-6 Pa` |
| 6B | Strain | `1e-10` |
| 6B | Temporal order against continuous x(t) | `|observed_order-1| < 1e-6` |
| 6C | Stress | `1e-6 Pa` |
| 6C | Strain, F, volume, position | `1e-11` in respective units |

These tolerances are chosen for these deterministic, analytically simple
benchmarks and use `float64`; they are **not universal MPM acceptance limits**.
Larger, multiphysics or nonlinear simulations need separate criteria.

## Current verification outcome (2026-10-25)

Ran with reconstructed Stage 1–5 source ZIPs and NumPy/Numba/Pytest in a
separate Python environment. The Stage-6 example reported:

- 6A: `position_error=5.11e-15 m`; `stress_abs_max=3.88e-12 Pa`;
  `crossing_particles=24`.
- 6B: `velocity_error=4.44e-16 m/s`;
  `Euler position error=4.44e-15 m`;
  continuous-position error at `dt=0.01s, T=0.2s` = `0.00981 m`;
  observed orders ≈ `1, 1`.
- 6C: `stress_error=2.55e-11 Pa`, `F_error=2.22e-16`;
  expected final `sigma_zz=-4000 Pa`, `sigma_xy=8000 Pa`.

Re-run locally on the repository to verify consistency with your own code.

## Next steps

1. Develop a **force-equilibrium elastic patch test** after adding consistent
   traction/BC and reaction calculations.
2. Stage 6D: wave propagation and phase/amplitude errors.
3. Stage 6E: cell-crossing with nonzero stress, to quantify MPM grid-crossing
   oscillations and later compare conventional MPM vs GIMP.
4. Stage 6F: energy diagnostics, discretization studies and reproducible plots.
5. Before large-rotation runout, make stress integration objective and validate
   rigid rotation of a prestressed body.
