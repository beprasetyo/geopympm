# GeoPyMPM Stage 3A — Particle-to-Grid (P2G)

## Scope

Implements **conventional 2D MPM** projection of **mass** and **momentum** from particles to four supporting nodes on each structured Quad4 cell, using interpolation computed in Stage 2.

This does **not** implement an explicit solver, grid forces, boundary-condition enforcement, G2P, contact, or stress updates.

## Equations

For particles `p`, nodes `i`, shape weight `N_ip`:

\[
m_i = \sum_p N_{ip} m_p, \qquad
\mathbf p_i = \sum_p N_{ip} m_p \mathbf v_p.
\]

Projected, **unconstrained** nodal velocity (at occupied nodes):

\[
\mathbf v_i^{\mathrm{proj}} = \frac{\mathbf p_i}{m_i}\quad(m_i>0).
\]

Velocity is zero for exactly massless nodes. These nodal quantities are *pre-force update*; velocity boundary conditions are stored but **not enforced** by P2G.

## Files

- `src/geopympm/solvers/p2g.py`: validation, interpolation freshness check, grid reset and kernel calls.
- `src/geopympm/kernels/p2g.py`: Numba kernels for mass/momentum accumulation and nodal velocity recovery.
- `tests/test_p2g.py`: isolated and integration tests.
- `examples/02_particle_to_grid.py`: demonstration and conservation sums.

## Usage

```python
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import particle_to_grid

grid = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
particles = ParticleSet(position=[[.5, .5]], mass=[4.], volume=[.002], velocity=[[2., -1.]])
interp = evaluate_quad4(particles, grid)
particle_to_grid(particles, grid, interp)
```

Output: each of four cell nodes has `mass=1`, `momentum=[2,-1]`, and `velocity=[2,-1]`.

## Important rules

1. **Re-evaluate Quad4 interpolation after particles move.** The wrapper detects stale particle positions using the stored cell and local coordinates; do not alter interpolation arrays manually.
2. **Reset on every call:** `particle_to_grid` calls `grid.reset_nodal_state()` first, clearing nodal mass, momentum, velocity, acceleration, and forces. It intentionally does **not** reset geometry or velocity boundary-condition data.
3. **No naive parallel assembly:** many particles can update the same node, creating write races; the initial Numba loop is serial. Parallelization needs a safe accumulation scheme.
4. **Single velocity field:** different material IDs currently share one nodal field; multimaterial contact will need more information.
5. **Conservation:** for valid Quad4 partition of unity, sum of nodal mass equals total particle mass and sum of nodal momentum equals total particle momentum (within floating-point roundoff).
6. **Grid boundaries:** as in Stage 2, out-of-domain particle positions must be handled by the caller (domain extension, exit/removal strategy, etc. will be addressed in the solver).
7. **No stress use here:** particle stress and volume are not used for mass/momentum projection; they will enter the internal-force and constitutive stages.

## Tests

From the installed editable package:

```bash
python -m pytest tests/test_particles.py tests/test_grid.py tests/test_interpolation_quad4.py tests/test_p2g.py -v
python examples/02_particle_to_grid.py
```

## Next

Stage 3B should assemble internal/external nodal forces, apply constraints, update grid momentum/velocity explicitly, and then add G2P and kinematic updates. The exact MPM scheme (USF/USL/MUSL and PIC/FLIP) remains to be chosen and tested.
