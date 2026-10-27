# GeoPyMPM Stage 6F — Energy and Separated Convergence Verification

## Aim and scope

Stage 6F **adds only validation code**; it does not change `ExplicitMPMSolver`,
interpolation, Numba kernels, or any previous benchmark. It reuses Stage 6D's
plane-strain standing longitudinal elastic wave, where end velocities are
constrained to zero, all transverse nodal velocities are zero, and interior
longitudinal velocities are **computed from internal forces**. Therefore the
benchmark actually exercises particle stress → nodal forces → velocity → strain.

These are **code-verification** tests for the small-strain linear-elastic
standard point-MPM baseline, **not** validation for landslide runout.

## 6F-1: Closed-system mechanical-energy assessment

With no gravity, applied traction, plastic dissipation, prescribed nonzero
velocity, or external work, the ideal *continuum* mechanical energy is constant.
We diagnose the **approximate** small-strain energy

\[
K=\frac12\sum_p m_p\mathbf v_p\cdot\mathbf v_p,\qquad
U\approx\frac12\sum_p V_{p,0}\,\boldsymbol\sigma_p:\boldsymbol\varepsilon_p,
\qquad E=K+U.
\]

Here `particles.strain` is accumulated engineering strain from Stage 3D,
`particles.stress` follows Stage 4's linear elastic law, and Voigt order is
`[xx,yy,zz,xy,yz,xz]`. Thus `stress dot strain` correctly includes engineering
shear gamma, **without an extra factor of 2**, and includes sigma_zz. Reference
volume V0, rather than current V, is used in this small-strain approximation.

This energy expression is valid only for these small-deformation, nonplastic,
rotation-negligible simulations. It is **not** a discrete energy conservation
theorem for USL-like PIC/FLIP and is not a finite-strain energy density. We
measure relative residual `(K+U-E0)/E0` for FLIP, alpha=.05 PIC blend, and PIC.
The FLIP residual is subject to a 2% acceptance criterion; the dissipative PIC
and blend are interpreted **as comparisons**, not failures. No artificial
physical damping is added.

## 6F-2: Separated spatial refinement (dt fixed)

- Domain L=4m, H=.5m, E=120 kPa, nu=.25, rho=1200kg/m3, amplitude=.002m;
  ny=2, 4 particles/cell.
- **Fixed** time step `dt=.00025 s` and final time `.3 s` for every run.
- Change **only** `nx = 16, 32, 64`, therefore h = L/nx.
- Report the **maximum over time** normalized stress L2 error against the
  analytical continuum first standing-wave mode, plus displacement and
  velocity errors.
- Estimate observed order from consecutive stress errors vs h.

This reports a spatial refinement trend at fixed temporal resolution; it does
**not** mathematically eliminate residual time integration or particle
integration errors, and the observed rate is not universal.

## 6F-3: Separated temporal refinement (grid fixed)

- **Fixed** nx=32, ny=2 and 4 particles/cell, t_end=.3s.
- Change **only** dt=.002, .001, .0005, .00025s. All satisfy Stage 5's
  heuristic CFL guard for these inputs.
- A reference calculation at **the same grid** but dt=.000125s provides a
  *fine numerical reference*, not the exact continuum solution.
- Compare final-time particle displacement and sigma_xx fields in L2 against
  that reference, retaining particle identity and material coordinates.
- Independently, evaluate first-order **self-convergence** from stress field
  differences between successive dt values. The ratio of successive differences
  should be approximately 2, giving observed temporal order approx. 1.
- Also report maximum analytic continuum stress error; unlike same-grid
  reference error it cannot shrink to zero solely by decreasing dt because
  the grid discretization remains finite.

## 6F-4: Heuristic CFL guard

Use c_p = sqrt((lambda+2G)/rho), h_min=min(dx,dy), and a stationary one-particle
elastic problem. Verify that a time step below `0.25 h_min/c_p` is accepted and
a step above this bound raises an error **before mutating particle state**.
This tests Stage 5's *input guard*, not stability of an arbitrary simulation.

## Run

From GeoPyMPM root, after installing Stage 6D and Stage 6E:

```bash
unzip ~/Downloads/GeoPyMPM_Stage6F_Energy_Convergence.zip
source .venv/bin/activate
python -m pytest tests/ -v
python -m validation.stage6.run_energy --plots
```

Files are written under `validation/results/stage6f/`:

- `6F_summary.json` — measurements, acceptance flags, caveats, observed rates.
- `6F_FLIP_energy_history.csv`, `6F_BLEND_0p05_energy_history.csv`,
  `6F_PIC_energy_history.csv` — kinetic / elastic / total histories.
- `6F_spatial_refinement.csv`, `6F_temporal_refinement.csv` — raw error tables.
- `6F_energy_ratio.png`, `6F_flip_energy_components.png`,
  `6F_spatial_convergence.png`, `6F_temporal_convergence.png` (if `--plots`).

## Expected result on reference baseline

| Experiment | Expected result |
|---|---|
| Energy: FLIP | E_final/E0 ≈ 1.00162; max relative energy residual ≈ 0.00219 |
| Energy: blend 5% PIC | E_final/E0 ≈ 0.88315 |
| Energy: PIC | E_final/E0 ≈ 0.10899 |
| Space nx=16,32,64, dt fixed | Max stress error ≈ 9.108%, 4.535%, 2.267%; apparent orders ≈ 1.01, 1.00 |
| Time dt=.002,.001,.0005,.00025, nx fixed | Stress error vs fine reference ≈ .5083%, .2374%, .1022%, .03424%; self-convergence orders ≈ 1.00 |
| CFL input guard | dt_limit ≈ .00285272 s; one step below accepted, one above rejected |

## Remaining limitations after Stage 6F

- There is still no fully objective large-rotation constitutive stress update.
- No plasticity, contact, basal friction, or load-aware general external-work
  energy accounting has yet been implemented.
- Standard Quad4 point MPM still shows cell-crossing errors (Stage 6E).
- Stage 5's explicit CFL formula remains a useful **heuristic**, not a proof
  of numerical stability.
- Energy conservation for this special closed system does not guarantee correct
  runout distance or energy dissipation for soil slides.
