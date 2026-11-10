# GeoPyMPM Stage 12 — USF / USL-like / MUSL explicit integration

**Baseline:** Stage 11 CNH patch on top of 10G-D4. **Installation:** unzip the incremental Stage 12 patch into the existing repository root (preserves all other files).

## Algorithm specification

The preexisting default is deliberately preserved and named `scheme="usl"` (more precisely **USL-like**, not yet independently proven equivalent to all published USL variants). All three schemes support LinearElastic and CompressibleNeoHookean (CNH), with MPM, uGIMP, cpGIMP, and CPDI1 transfer. PIC/FLIP blend and particle advection are independent options. `interpolation` weights and gradients are evaluated at old particle position/domain, never silently at moved positions.

- **USL-like (default, unchanged):** evaluate interpolation at x_n; P2G mass and momentum; grid internal/external force and momentum update using stress_n; kinematics/F from updated grid velocity; update constitutive stress; G2P velocity and position; update cpGIMP or CPDI1 domain.
- **USF:** interpolation at x_n; P2G; apply prescribed velocity BC to *pre-force projected nodal velocity*; F/strain from projected grid velocity; update constitutive stress to stress_(n+1); assemble forces and advance grid momentum using *new* stress; G2P velocity and position; update particle domains. For affine-exponential advection, use post-force grid velocity gradient, not USF pre-force gradient.
- **MUSL:** interpolation at x_n; P2G; grid force/momentum update using stress_n; G2P particle **velocity only** at x_n; second P2G of **updated particle momentum** using original x_n weights, recompute nodal velocity and reapply nodal BC; kinematics/F/constitutive stress using remapped nodal velocity; advect particle **position only** using that remapped field; update particle domains. Grid mass unchanged, acceleration retains physical force-update value. This is the specific MUSL variant chosen for Stage 12.

The second momentum remapping is implemented in `src/geopympm/solvers/momentum_remap.py` using the same Numba P2G kernel as the first remap. It does **not** advance forces/time a second time. Interpolation weights and gradients on the first and second remap are identical.

## Public API

```python
solver = ExplicitMPMSolver(particles, grid, materials,
    dt=0.0005, scheme="musl", interpolation="mpm",
    stress_update="hyperelastic", kinematics_update="exponential")
```

Values: `scheme="usl"` (default), `"usf"`, or `"musl"`. Existing Stage 11 examples continue to default to `"usl"` and are unmodified.

## Checkpoint compatibility

- Schema **1**: legacy LinearElastic + implicit USL-like (config has no `scheme`).
- Schema **2**: legacy CNH + implicit USL-like (config has no `scheme`).
- Schema **3**: either material family, *explicit* scheme `"usf"` or `"musl"` in config. Unknown/missing/mismatched schema/field rejected. Old checkpoint schema 1/2 remains supported and regenerates `scheme="usl"`. This convention preserves legacy checkpoint payload shapes and configuration equality.
- The checkpoint array inventory is unchanged; the solver state contains a new `scheme` string. Restart, recovery, SHA-256 integrity, and OutputManager retain previous semantics.
- A scheme cannot be changed by loading a checkpoint and silently resuming another scheme; to compare schemes, start independent simulations from identical initial particle and grid state.

## Verification

`python -m pytest -q tests/test_stage12_integration_schemes.py`

Tests cover: invalid options; all 3 × 4 interpolation methods × 2 material families; conservation of mass and total momentum in free, unforced systems; USF same-step internal-force ordering; MUSL exact second momentum projection; schema-3 checkpoint/restart reproducibility; schema 1/2 compatibility; tamper protection; affine-exponential advection; stage regression tests.

**Sulsky compare** (with CNH, source `examples/24_sulsky_colliding_disks_cnh.py`):

```bash
python examples/25_compare_time_integration_schemes.py --steps 200 --no-plot \
    --output-dir results/stage12_sulsky_smoke
python examples/25_compare_time_integration_schemes.py --steps 6000 \
    --output-dir results/stage12_sulsky_full
```

Outputs: per-scheme CSV with `time_s`, kinetic/elastic/total energy, momentum, disk centers and J; comparison.json with final energy change, momentum drift, rebound and wall time; comparative PNGs if requested. No VTU is produced by this comparison script: run the Stage 11E single-case output workflow when needed for particle animations.

## Constraints / open validation

- Existing 2D single-velocity-field, explicit grid with no dedicated multimaterial contact; Uintah 3D/artificial viscosity/adaptive-step differences remain.
- USF uses projected pre-force nodal velocities for kinematics; MUSL uses post-force updated particle momentum remapped to grid. These are explicit algorithm choices, **not** a claim of unique standard USF/MUSL convention across literature.
- No claim that a scheme necessarily improves energy conservation or is stable at arbitrary dt. Assess time-step, particle density, interpolation and boundary sensitivity.
- Slope/runout/plastic materials and substantial cell crossing are future validation targets.
- Using the same grid and particle velocity field does not imply that the independent bodies have physically correct frictional contact.

## Measured 3.0 s comparison (indicative, local execution)

Configuration: CNH Sulsky surrogate; standard MPM interpolation; 20×20 cells; 416 particles; `dt=0.0005 s`, 6,000 steps; `pic_fraction=0.0` (FLIP); identical geometry and material parameters. These figures come from **completed individual-scheme simulations**, not an externally validated Uintah benchmark:

| Scheme | Final relative total energy change | Maximum drift of total momentum |
|---|---:|---:|
| Legacy USL-like | −1.5865% | ~1.6e−15 kg m/s |
| USF | +0.0435% | ~1.1e−15 kg m/s |
| MUSL | −0.0424% | ~2.9e−15 kg m/s |

All three exhibited rebound of both disks. The diagnostic energy is `kinetic + reference-volume-integrated CNH strain energy`. Drift estimates depend on diagnostic sampling, integration order and platform. A smaller final drift does **not** prove superior accuracy or stability; mesh/step convergence and published-reference comparisons are needed.

Regression checks completed in the patch creation environment: **261 passed** (Stage 12, Stage 11, selected Stage 7–10G and Sulsky tests). A full `pytest tests/` run reached approximately 51% before execution timeout; the full suite is **not** claimed to pass.
