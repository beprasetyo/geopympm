# GeoPyMPM Stage 13B — Mohr–Coulomb smooth-face stress integration

**Baseline:** Stage 13A. **Deliverable:** incremental patch, no full repository.

## Scope and safety

- Implemented: isotropic 3D Mohr–Coulomb yield criterion evaluated on a **2D plane-strain** stress state (including nonzero sigma_zz); small-strain *incremental* elastic predictor; perfect plasticity; associated/non-associated flow; consistent per-particle plastic strain and equivalent plastic strain histories.
- Implemented: elastic CFL estimate, existing USF/USL/MUSL and MPM/uGIMP/cpGIMP/CPDI1 integration, transactional stress/history update and time-step rollback.
- Implemented: checkpoint **schema 5** for Mohr–Coulomb including exact material parameters and plastic history, with backward compatibility for schemas 1–4.
- **Not implemented:** Mohr–Coulomb edge/vertex/apex return mapping, tension cutoff, hardening/softening, corotational MC, finite-strain MC, mixed material families, pore pressure/effective stress. Reaching a corner/edge/apex raises an error **without committing** that step. Stage 13C is required for full 3D principal stress space coverage.
- **Not validated for landslide runout.** Stage 13D must add independent material-point and boundary-value validation.

## Conventions and equations

Tension-positive stress (Pa), Voigt `[xx, yy, zz, xy, yz, xz]`; `strain_increment` has engineering gamma_xy. For plane strain, delta epsilon_zz = 0, but sigma_zz and plastic epsilon_zz may be nonzero.

With ordered eigenvalues `sigma_max >= sigma_mid >= sigma_min`:

\[
 f = (\sigma_{max}-\sigma_{min}) + (\sigma_{max}+\sigma_{min})\sin\phi - 2c\cos\phi \le 0.
\]

Non-associated smooth-face potential:

\[
 g = (\sigma_{max}-\sigma_{min})+(\sigma_{max}+\sigma_{min})\sin\psi.
\]

Trial stress: `sigma_trial = sigma_old + C : delta_epsilon`, isotropic plane strain. If `f_trial <= numerical_tol`, the step is elastic. Otherwise for one **smooth face**:

\[
 \Delta\gamma = f_{trial}/(n_f^T C n_g), \qquad
 \boldsymbol\sigma_{new} = \boldsymbol\sigma_{trial} - \Delta\gamma\, C n_g.
\]

Here `n_f = [sin(phi)-1, 0, sin(phi)+1]`, and `n_g` replaces phi with psi, in **ascending** principal order. The principal eigenvector basis of the trial stress is held fixed for this return. A positive correction must retain **strict ordering** of all three corrected eigenvalues; otherwise the step raises (`edge/apex return is not supported until Stage 13C`).

Plastic strain increment is `delta_gamma * dg/dsigma` transformed to the global axes, stored as **engineering Voigt strain** in `MaterialState['plastic_strain']`; `equivalent_plastic_strain` is the integrated `sqrt(2/3) ||dev(delta_epsilon_plastic)||`. Perfect plasticity: c and phi do not evolve.

The model is **rate-independent** and acts on the spatial small-strain increment from MPM kinematics. It has no objective stress rotation. When significant rotations occur, it is not physically objective. Do not interpret the Stage 13B support of CPDI/large-deformation kinematics as finite-strain Mohr–Coulomb validation.

## API

```python
from geopympm.materials import MohrCoulomb

mat = MohrCoulomb(
    young_modulus=1.0e7, poisson_ratio=0.30,
    cohesion=1.0e4, friction_angle_deg=30.0,
    dilation_angle_deg=0.0,
)
solver = ExplicitMPMSolver(
    particles, grid, mat, dt=1e-5,
    scheme="usl", stress_update="incremental",
)
# solver.material_state automatically has
#   ['plastic_strain']            shape (n, 6)
#   ['equivalent_plastic_strain'] shape (n,)
```

Multiple parameter sets in a homogeneous MC mapping are supported by integer material ID. There is no mixed linear/MC/CNH family dispatch yet. An explicit custom `MaterialState` must at minimum declare the two required fields with exact shapes.

## Checkpoint

A Mohr–Coulomb checkpoint is written as schema **5**, distinct from schema 4 (custom histories on elastic/CNH). Manifest contains the MC parameters (E, nu, c, phi, psi), model identifier, solver scheme, and declared history field metadata. NPZ contains all numerical histories. `load_checkpoint()` reconstructs and verifies the full configuration and state; older schemas remain valid. Unsupported future model kinds/versions fail closed.

## Verification and remaining risks

- Tests: invalid parameters, elastic predictor vs Hooke, stress exactly on active yield face, associated vs non-associated dilation, compression/tension sign, atomic rollback on apex failure, 3 schemes × 4 interpolations, checkpoint round trip and continuation, backward compatibility.
- Sample `examples/27_mohr_coulomb_material_point.py` writes `results/stage13b_mc_shear.csv` for a small-increment shear loading path.
- The active-face test rejects some real stress paths that *should* yield at edges/corners, by design. This is not a general-purpose MC integrator until Stage 13C completes multisurface return mapping. High angular rotation must likewise await an objective/finite-strain formulation.

## Install and test

Apply ZIP at root of an existing **Stage 13A** repository. Run:

```bash
python -m pytest -q tests/test_stage13b_mohr_coulomb.py
python examples/27_mohr_coulomb_material_point.py
```

The tests and example should be rerun in the user's Ubuntu environment before proceeding to Stage 13C.
