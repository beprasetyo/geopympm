# GeoPyMPM Stage 11 — Compressible Neo-Hookean (CNH)

**Date:** 10 November 2026  
**Patch baseline:** Stage 10G-D4 + Sulsky colliding disks example patch (Example 23)  
**Patch only:** new/modified files; no full repository.  
**Status:** CNH material and solver/checkpoint integration implemented; constitutive verification and selected regression tests pass. Quantitative validation against Uintah executable output is **not complete**.

## Stage boundaries

| Substage | Deliverable | Status |
|---|---|---|
| 11A | Validated `CompressibleNeoHookean` material, E/nu conversion, material-family dispatch | Implemented |
| 11B | Numba plane-strain UCNH Cauchy stress + reference strain-energy density | Implemented, unit-tested |
| 11C | CNH integrated into `ExplicitMPMSolver`; checkpoint/restart/recovery compatible | Implemented, integration-tested |
| 11D | Finite-strain stress/energy verification (rotation, shear, compression, energy derivative) | Tests pass |
| 11E | Sulsky colliding disks CNH example, momentum/energy diagnostics, dt sensitivity | Runnable / preliminary validation; exact Uintah solver comparison pending |

**Stage 12 USF/MUSL** and **Stage 13 elastoplasticity** are deliberately not included.

## Constitutive reference

Authoritative source in the uploaded Uintah code:
`src/CCA/Components/MPM/Materials/ConstitutiveModel/UCNH.cc`, in the elastic branch of `computeStressTensor` (the relevant source calculates `bElBarTrial`, `pStress = I*p + tauDev/J` and strain energy `U+W`). Original setup: `src/StandAlone/inputs/UintahRelease/MPM/disks_sulsky.ups`.

The implemented **elastic, isothermal** branch uses:

\[
J=\det F > 0,\quad B = FF^T,\quad \bar B=J^{-2/3} B,
\]
\[
\sigma = \frac{K}{2}\left(J-\frac{1}{J}\right) I
       + \frac{G}{J}\left(\bar B-\frac{\operatorname{tr}(\bar B)}{3}I\right),
\]
\[
\Psi(F)=\frac{K}{2}\left[\frac{J^2-1}{2}-\ln J\right]
          +\frac{G}{2}(\operatorname{tr}(\bar B)-3).
\]

Plane strain uses `F3=diag(F2,1)` with zero out-of-plane shear. Stress Voigt ordering is `[xx,yy,zz,xy,yz,xz]` with tension-positive sign. `Psi` is **energy per reference volume**. Physical stored energy is `sum(volume0[p]*Psi[p])`.

The source UCNH also includes plasticity, thermal expansion, artificial viscosity, and other options; none of these are implemented in Stage 11. GeoPyMPM continues to use a **single nodal velocity field**, not a multimaterial contact solver.

## API example

```python
from geopympm.materials import CompressibleNeoHookean, neo_hookean_strain_energy
from geopympm.solvers import ExplicitMPMSolver

mat = CompressibleNeoHookean.from_young_poisson(1000.0, 0.30)
# Equivalent K = 833.333... Pa, G = 384.615... Pa.
# Alternatively: mat = CompressibleNeoHookean(833.333333, 384.6153846)
solver = ExplicitMPMSolver(
    particles, grid, mat, dt=0.0005,
    interpolation="mpm", stress_update="hyperelastic",
    kinematics_update="exponential", pic_fraction=0.0,
)
solver.step()
energy_joules = neo_hookean_strain_energy(particles, mat)
```

For material IDs, use `{0: mat0, 1: mat1}`. Stage 11 intentionally supports a homogeneous **constitutive family** across IDs; mixed `LinearElastic` and `CompressibleNeoHookean` in one solver are rejected explicitly until future material-group dispatch exists.

### Validation and restrictions

- CNH requires `stress_update="hyperelastic"`. LinearElastic retains its existing default incremental and optional corotational stress update. `kinematics_update="exponential"` is recommended for finite-strain kinematics but is a *kinematic scheme*, not a new MPM integration scheme.
- The solver's CFL estimate for CNH uses reference small-strain `K+4G/3` over current density. **CFL is heuristic and can be insufficient** for energetic stability at large deformation; use a time-step convergence study.
- The reference `F=I` means a stress-free initial reference state. Prescribed initial stress from FEM without a consistent elastic deformation and reference configuration is **not** supported by this simple CNH model; otherwise the first stress update overwrites initial stress.
- CNH material law is truly finite-strain hyperelastic; the solver **still uses its Stage 10 USL-like time integration**. MUSL and USF are not Stage 11 features.
- No contact, damage, plasticity, thermal EOS, artificial viscosity, material-to-material separation algorithm or mesh independence claim is made.

## Checkpoint compatibility

- Existing **LinearElastic** schema-1 checkpoints are written/restored in their original format and retain byte-level numeric state.
- CNH checkpoints use **schema version 2** and save validated `{model:"CompressibleNeoHookean", parameters:{bulk_modulus, shear_modulus}}` metadata. The numeric arrays are unchanged; no new plastic internal-state fields are introduced.
- Schema versions mismatched with material model names are rejected. Existing SHA-256 archive/VTU protection, restore, recovery and retention stay active.
- The *older* Stage 10G-B2 loader cannot read new CNH checkpoints; update both writer and loader with this patch.

## Sulsky colliding disks validation example

`examples/24_sulsky_colliding_disks_cnh.py` uses two disks with centers (0.25,0.25) and (0.75,0.75) m, R=0.20 m, domain 1.0x1.0 m, 20x20 cells, thickness 0.05 m, density 1000 kg/m^3, equal and opposite (0.1,0.1) m/s velocities, and E=1000 Pa, nu=0.30. With two particles per cell direction, the sampling gives 208 particles per disk. GeoPyMPM reduces the through-thickness particles to a 2D plane-strain representation.

**Run from repository root** after applying patch:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_stage11_cnh.py

# fast test
python examples/24_sulsky_colliding_disks_cnh.py --steps 50 --no-vtu \
  --output-dir results/sulsky_cnh_smoke

# full 3 s, recommended stable baseline: dt=0.0005 s, 6000 steps
python examples/24_sulsky_colliding_disks_cnh.py --plot \
  --output-dir results/sulsky_cnh_full
```

Open `results/sulsky_cnh_full/visualization/particles.pvd` in ParaView, or `particles.visit` in VisIt. The step number is **not** a time in seconds; for the default dt, `step=2150` corresponds to `t=1.075 s`. Existing OutputManager outputs at intervals of 50 steps = 0.025 s.

### Observed time-step sensitivity (Stage 11 implementation, original local test runtime)

Simulations were run with the same seeded particles and initial conditions using `mpm`, `pic_fraction=0`, `kinematics_update="exponential"`, no physical damping and standard Stage-10 solver sequencing.

| Fixed dt (s) | Steps | Initial total energy (J) | Final total energy (J) | Relative energy change at 3 s | Max momentum drift (kg m/s) |
|---:|---:|---:|---:|---:|---:|
| 0.0010 | 3000 | 0.130000 | 0.168744 | +29.80% | 1.16e-3 |
| 0.0005 | 6000 | 0.130000 | 0.127938 | −1.59% | 2.14e-15 |

**Do not report the 0.001 s result as validated energy conservation.** The large late-time energy gain around `t≈2.2 s` signals time-integration/particle-grid sensitivity. Reducing dt improves the result dramatically, but **−1.59%** is still not exact energy conservation. This is a first numerical sensitivity check, not mesh/algorithmic convergence or proof of full benchmark accuracy.

Reproducing Uintah numerically requires aligning its own deformation-gradient integrator, adaptive stepping, mass/particle quadrature, interpolation, artificial viscosity switches, boundary conditions, single/multi material semantics and 3D-vs-2D representation. No Uintah output time history was executed and compared in this patch.

## Tests executed in this development environment

- `tests/test_stage11_cnh.py`: 33 passed (including 3D independent stress reference, strain-energy derivative, frame objectivity, material validation, checkpoint + VTU recovery, solver modes, example smoke).
- `tests/test_stage10g* tests/test_stage11_cnh.py tests/test_sulsky_colliding_disks.py`: 238 passed before two tests were added; repeat locally after patch installation.
- Selected prior constitutive/solver/interpolation tests: 211 passed.
- A full-repository run exceeded the environment time limit after approximately 55% of the tests had run; therefore **full-suite pass is not claimed**.

## Files changed

```text
src/geopympm/kernels/neo_hookean.py           NEW
src/geopympm/materials/neo_hookean.py         NEW
src/geopympm/materials/registry.py            NEW
src/geopympm/materials/__init__.py            MODIFIED
src/geopympm/solvers/explicit.py              MODIFIED
src/geopympm/io/checkpoint.py                 MODIFIED
src/geopympm/io/restart.py                    MODIFIED
examples/24_sulsky_colliding_disks_cnh.py    NEW
tests/test_stage11_cnh.py                     NEW
docs/stage11_compressible_neo_hookean_10nov2026.md NEW
```

This stage intentionally **does not alter** the existing Example 23, Stage 10 numerical kernels, the old LinearElastic stress integrator, or the interpolation library.
