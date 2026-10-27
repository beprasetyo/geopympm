# Stage 7 — Objective Stress Update (2D Plane Strain)

## Motivation

The Stage-4 incremental elastic relation `sigma += C:Delta_epsilon` is not objective under rotations. For example, a material carrying non-hydrostatic initial stress rotated rigidly by 90° should have its **Cauchy stress components rotated** even with zero strain rate.

Stage 7 adds a **corotational midpoint integration of the Jaumann rate**. It does not overwrite the legacy Stage-4 path. The default of `ExplicitMPMSolver` remains `stress_update="incremental"` **for compatibility with Stage 6A–6F**. Explicitly select `stress_update="corotational"` for the new formulation.

## Formulation

For spatial velocity gradient `L = grad(v)`, define `D=(L+L^T)/2`, `W=(L-L^T)/2`, and plane spin `omega=(L_yx-L_xy)/2`, so `W=[[0,-omega],[omega,0]]`.

The Cauchy stress Jaumann rate is `sigma^triangle = dot(sigma)-W*sigma+sigma*W = C:D`. Stage 7 approximates the integrated stress by

```
R(a) = [[cos(a), -sin(a)], [sin(a), cos(a)]]
Delta_sigma_el = lambda*trace(Delta_epsilon)*I + 2*G*Delta_epsilon
sigma_new_inplane = R(omega*dt) @ sigma_old @ R.T +
                    R(omega*dt/2) @ Delta_sigma_el @ R(omega*dt/2).T
sigma_zz_new = sigma_zz_old + lambda*(Delta_eps_xx + Delta_eps_yy)
```

It uses **engineering shear** in `strain_increment[:,3]` and updates the 6-component tension-positive Voigt stress `[xx,yy,zz,xy,yz,xz]`. Out-of-plane stresses yz and xz are zero in 2D plane strain. The rotation is exact for **constant pure in-plane spin and D=0**, and the rotating elastic increment uses a midpoint approximation when D != 0. The method is a **hypoelastic rate integrator**, **not finite-strain hyperelasticity**, and no claim is made of energy conservation for arbitrary large-strain loading paths.

## API

```python
from geopympm.materials import LinearElastic, update_objective_linear_elastic_stress

update_objective_linear_elastic_stress(
    particles, kinematics.strain_increment,
    kinematics.velocity_gradient, materials, dt,
)
```

Integrated solver:

```python
solver = ExplicitMPMSolver(particles, grid, LinearElastic(E, nu),
                           dt=dt, pic_fraction=0.0,
                           stress_update="corotational")
```

The material routine validates shapes, finite values, plane strain, and consistency between `strain_increment` and `dt*sym(L)` **before** computing a candidate array. The candidate is atomically committed to `particles.stress` only when all particles have finite stress. The Stage-5 rollback protects full particle/grid state on failure during an integrated step.

## Verification

1. **Exact rigid-stress rotation**, including a 90° rotation, invariant in-plane principal stresses, trace, and sigma_zz.
2. **Mixed strain and spin**, checked against independent integration of the constant-rate Jaumann equation, with decreasing error as dt decreases.
3. **Full Stage-5 cycle**, all nodal velocities prescribed as `v=(-omega*y, omega*x)`. This isolates stress objectivity. The legacy incremental method fails this objectivity check, as intended.
4. **Legacy regressions**: Stage 6A–6F still run with the unchanged default `stress_update="incremental"`.

Run:

```bash
python -m validation.stage7.run_rotation --plots
python -m pytest tests/test_stage7_objective_stress.py -v
```

## Critical limitations / next stage

- **Stage-3D kinematics is unchanged**: `F_new=(I+dt*L)F_old`; pure rigid rotation artificially increases `det(F)` by approximately `(1+omega**2 * dt**2)**n` after `n` steps. PIC particle advection also uses an explicit time step and therefore deviates from the exact circular trajectory. **Stage 7 fixes stress orientation, not finite-rotation kinematics**.
- This test prescribes *all* nodal velocities, so it does **not** validate unconstrained rigid-body rotation or force-driven objectivity.
- The accumulated `particles.strain` remains small-increment diagnostic and is not an objective large-strain tensor.
- Objective hypoelasticity does not imply finite-strain energy consistency; later upgrades should include proper large-deformation kinematics and possibly a finite-strain hyperelastic baseline.
- Contact, GIMP, Mohr–Coulomb, 3D, pile penetration, and impact physics are out of scope.
- Stage 7 maintains Python orchestration, NumPy particle state, and Numba-only per-particle kernels in line with GeoPyMPM Arsitektur.
