# Stage 9 — Affine-exponential particle advection (2D Quad4)

## Purpose

Stage 8 corrected the **deformation-gradient** update by adding
`F_(n+1) = exp(dt L_p) F_n`. However, Stage 3C still used **Euler particle
positions** `x_(n+1) = x_n + dt v_PIC`, producing O(dt) rotational trajectory
drift. Stage 9 adds an optional integrator for **positions**, leaving PIC/FLIP
**velocity** update unchanged.

## Equations and assumptions

After Stage 3B, the nodal velocities are updated. At the beginning of the
particle-motion substep, we already have from Stage 3D:

- `v_p = sum_i N_i(x_p^n) v_i^(n+1)` (updated-grid PIC velocity);
- `L_p = sum_i v_i^(n+1) tensor grad N_i(x_p^n)` (2×2 spatial gradient).

Freeze the locally affine velocity field for one time step:

`v(x) ≈ v_p + L_p (x-x_p^n)`.

Its exact (local) trajectory increment is:

`x_p^(n+1) = x_p^n + dt * phi1(dt L_p) @ v_p`,

where `phi1(B) = integral_0^1 exp(sB)ds = I + B/2! + B²/3! + ...`.

This formulation does not require `inverse(L)`: it supports **singular** L,
including translation, shear, and spin. The kernel computes phi1 using a
Taylor series after scaling the 2×2 matrix and recovers the original scale
using `phi1(2B)=(I+exp(B))phi1(B)/2`. The 2×2 exponential is reused from
Stage 8. Numba compiles the per-particle kernel.

The following relations remain **unchanged**:

`v_PIC = sum_i N_i v_i^(n+1)`;

`v_FLIP = v_p^n + dt sum_i N_i a_i`;

`v_p^(n+1) = alpha*v_PIC + (1-alpha)*v_FLIP`.

**Important:** Particle advection uses PIC *nodal velocity* even if `alpha=0`
(FLIP). It does **not** integrate the FLIP velocity, preserving the Stage 3C
separation between position advection and particle velocity update.

## API

```python
solver = ExplicitMPMSolver(
    particles, grid, material, dt=0.001, pic_fraction=0.0,
    stress_update="corotational",
    kinematics_update="exponential",
    advection_update="affine_exponential",
)
solver.run(100)
```

Backward-compatible default:

`advection_update="euler"` preserves all Stage 1–8 tests and references.

For a manual Stage 3C call after Stage 3D:

```python
kin = update_particle_kinematics(
    particles, grid, interpolation, dt,
    kinematics_update="exponential",
)
grid_to_particle(
    particles, grid, interpolation, dt,
    pic_fraction=0.0,
    advection_update="affine_exponential",
    velocity_gradient=kin.velocity_gradient,
)
```

Stage 3D must still occur **before Stage 3C**, because interpolation depends
on the **old** particle position. Stage 4 stress update also precedes Stage 3C.

## Validation

From root repository:

```bash
python -m pytest tests/test_stage9_affine_advection.py -v
python -m validation.stage9.run_advection --plots
python examples/10_affine_exponential_advection.py
```

**9A — prescribed rigid spin**: 16 particles, `omega=3 rad/s`, `dt=0.001 s`,
100 steps. Compare Euler and affine-exponential positions against the **exact**
`R(omega t) X` reference. With `kinematics_update="exponential"` and
`stress_update="corotational"`, volume and stress remain consistent. The
Stage 9 integration has max position error near roundoff (Euler drift is
approximately `2.02e-4 m` for this setup).

**9B — general stationary affine velocity** `v=A x+b` for 80 time steps.
Reference positions and F are generated *independently* from a Taylor-series
3×3 augmented homogeneous transform. The full MPM solver reproduces the
affine trajectory, `F`, and particle volume for both PIC and FLIP.

The validation uses imposed nodal velocities. It does not test a free dynamic
contact problem, or prove global higher-order accuracy for non-affine fields.

## Numeric and physical limitations

- The local affine approximation is valid over a *frozen* field during the
  step. A bilinear Quad4 velocity field is not globally affine, and crossing
  cell boundaries can invalidate the local representation mid-step. This is
  **not** exact path integration for arbitrary MPM fields.
- The Stage 9 method does **not** modify the stress integration (Stage 7) or
  deformation-gradient integration (Stage 8). For large rotation, enable
  those complementary modes explicitly.
- Velocity FLIP/PIC, grid momentum, CFL guard, and grid-boundary exit checks
  retain their previous meanings. Error handling still performs all-or-nothing
  G2P commit and whole-step solver rollback.
- Euler remains the default for regression compatibility; the new integrator
  is opt-in until its dynamic behavior and cost are assessed more widely.
- The method is a **local-affine exponential integrator**; using it does not
  automatically make the entire MPM algorithm higher-order in time or provide
  objective accumulated small strain under finite rotation.
- Contact, plasticity, GIMP, and 3D remain out of scope. The technique is
  independent of the application (landslide runout or pile penetration).
