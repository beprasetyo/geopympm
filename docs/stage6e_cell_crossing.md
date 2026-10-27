# GeoPyMPM — Stage 6E: Cell-Crossing Error in Point MPM

**Status:** controlled numerical verification/diagnostics; no modification to Stage 1–5 core code. Uses 2D Quad4 standard point MPM; **not GIMP**.

## Purpose and scope

The particle-to-grid nodal internal force in Stage 3B is:

\[
\mathbf f_i^{\mathrm{int}}=-\sum_p V_p\,\boldsymbol\sigma_p\,\nabla N_i(\mathbf x_p).
\]

In point MPM with piecewise linear shape functions, spatial gradients `gradients` can jump when a particle crosses a cell boundary. Therefore a single particle's *contribution to a particular node* can change discontinuously. Whether this produces significant errors for a physical many-particle simulation depends on particle distribution, material response, time step, constraints, and the benchmark. Stage 6E quantifies the phenomenon instead of assuming it always produces a particular runout error.

These are two **controlled prescribed-velocity force-assembly tests**, not free-response dynamics. Stage 6D already checked force-driven elastic oscillations at small strain. Stage 6E isolates a different numerical effect. The test does not validate landslide runout, GIMP, contact, or a stress update under large rotation.

## Test E1: single prestressed particle across an internal grid line

- 2D grid `dx=dy=1 m`, node `(ix,iy)=(2,0)` at `(x,y)=(2,0)`.
- One particle at `(2−epsilon, 0.5)` and `(2+epsilon, 0.5)` with `V=0.2 m³`, `sigma_xx=+120 Pa`, other stress components zero.
- Stage 2 interpolation and Stage 3A/3B force assembly are executed at each position.

At the lower node, `N_y(0.5)=0.5`:

\[
\lim_{x_p\to2^-}f_{(2,0),x}=-\frac{V\sigma_{xx}}{2\,h_x}=-12\,\mathrm N,
\qquad
\lim_{x_p\to2^+}f_{(2,0),x}=+12\,\mathrm N.
\]

The predicted signed jump is `+24 N`. The benchmark PASS means the code reproduces this expected *point-MPM discontinuity*, **not** that the discontinuity is physically desirable or has been eliminated.

## Test E2: uniformly translating prestressed finite block

- Uniform material block: `L=3 m`, `H=1 m`, unit out-of-plane thickness, `rho=1200 kg/m³`.
- Initial left/right block boundaries: `1.22 m` and `4.22 m`.
- 2D structured grid: `8×1` cells with `dx=dy=1 m`.
- Fixed uniform tensile stress `sigma_xx=+120 Pa`; `E=120 kPa`, `nu=0.25`.
- All nodal velocities are prescribed as `(0.9,0) m/s`, so the block translates rigidly. This deliberately **disables feedback from nodal internal forces to velocity**.
- Two rows of particles in `y`; particles per *x-cell* are varied among `2,4,8`. These are **not** 2/4/8 particles total per 2D cell: total particles per full 2D cell are 4/8/16.
- Sample the `x`-component of internal force at lower grid node `(5,0)`. The force from Stage 3B corresponds to particle positions **before** Stage 3C advection in that step.

For the *continuum* rectangular block with the same spatially constant stress, the nodal force is obtained independently by integrating the Quad4 nodal gradient across the moving block:

\[
 f_{(i,0),x}^{\mathrm{ref}}=-\int_{\Omega_\mathrm{block}}\sigma_{xx}\frac{\partial N_{(i,0)}}{\partial x}\,\mathrm dV
 =\frac{\sigma_{xx}H}{2h_x}\bigl(\ell_R-\ell_L\bigr),
\]

where `ell_L` is the length of the overlap of the block with `[x_i−h_x,x_i]` and `ell_R` is the overlap with `[x_i,x_i+h_x]`. The transverse integral of the bottom-node hat function is `H/2`. This reference is continuous as a block edge translates through a grid cell, while point-particle quadrature generates a staircase approximation.

### Error definitions

- `max_force_error_n = max |f_MPM − f_ref|` over the sampled times.
- `normalized_max_force_error = max_force_error_n / (sigma_xx*H/2)`.
- `normalized_rms_force_error = RMS(f_MPM − f_ref)/(sigma_xx*H/2)`.
- `max_force_change_noise_n = max |Δf_MPM − Δf_ref|`; a sampled finite-difference deviation, *not* a mathematical limiting jump.
- `max_stress_drift_pa` verifies the fully prescribed uniform motion does not create significant spurious stress.
- `crossing_particles` counts distinct particles that cross `x=5 m` during the simulation; it is not the total number of cell-crossing events anywhere on the grid.

### Reference values (current baseline)

| x particles/cell | Total particles | max normalized force error | RMS normalized error | max force change noise |
|---:|---:|---:|---:|---:|
| 2 | 12 | ~0.249 | ~0.144 | ~29.46 N |
| 4 | 24 | 0.125 | ~0.0700 | ~14.46 N |
| 8 | 48 | ~0.062 | ~0.0358 | ~6.96 N |

The error diminishes when the quadrature points are refined **while the background grid and time step are fixed**. This is particle refinement, *not* spatial/grid convergence. The coarse-to-fine error behavior partly reflects the specifically chosen uniform rectangular block and sampling points; it is not a universal convergence order for MPM.

### PASS semantics

- E1: the numerical left/right forces agree with derived analytical limits.
- E2: nonzero-stress force sampling approximates the integrated continuum force within a resolution-dependent tolerance; uniform advection preserves particle stress and movement; some particles cross x=5.
- Refinement: the maximum sampled force error decreases when 2→4→8 x-particles/cell.

**PASS means a known behavior has been correctly reproduced and quantified. It does not mean cell-crossing noise is absent.**

## Install and run

Extract the ZIP in the existing GeoPyMPM repository root. No core source files will be overwritten.

```bash
cd ~/BagusDoc/myfem/geopympm
unzip ~/Downloads/GeoPyMPM_Stage6E_Cell_Crossing_Verification.zip
source .venv/bin/activate
python -m pytest tests/ -v
python -m validation.stage6.run_crossing --plots --refinement
```

Default outputs: `validation/results/stage6e/` with summary JSON, CSV series and optional PNG plots. `matplotlib` is required only for `--plots`.

## Next steps

Use the same analytical continuum integral and spatial/temporal sampling strategy when GIMP interpolation is introduced. A true **free-response** stress-bearing cell-crossing benchmark, dynamic energy diagnostic, and objective stress update are still needed before claiming quantitative large-deformation runout predictive validity. Do not add artificial damping to hide cell-crossing noise.
