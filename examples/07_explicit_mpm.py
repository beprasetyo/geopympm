"""Stage 5: integrate multiple 2D elastic MPM steps (validation demonstration).

Run from repository root: python examples/07_explicit_mpm.py
This is NOT a landslide runout or large-rotation constitutive validation.
"""

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def main():
    particles = ParticleSet(
        position=[[0.50, 0.50]],
        mass=[450.0],
        volume=[0.25],
    )
    grid = BackgroundGrid(origin=(0., 0.), cell_size=(1., 1.), n_cells=(1, 1))
    # Constrain vertical velocity of the two bottom grid nodes.
    grid.velocity_bc_mask[[0, 1], 1] = True

    soil = LinearElastic(young_modulus=20_000.0, poisson_ratio=0.25)
    solver = ExplicitMPMSolver(
        particles, grid, soil, dt=0.001, cfl=.25,
        gravity=(0., -9.81), pic_fraction=1.0,
    )
    print(f"Indicative dt limit = {solver.stable_dt_estimate():.6g} s")
    for _ in range(10):
        diag = solver.step()
        if diag.step in (1, 5, 10):
            print(
                f"step={diag.step:2d} t={diag.time:.4f} s "
                f"y={particles.position[0, 1]:.6f} m "
                f"vy={particles.velocity[0, 1]:.6g} m/s "
                f"K={diag.kinetic_energy:.6g} J"
            )
    print("stress [xx, yy, zz, xy, yz, xz] (Pa):", particles.stress[0])
    print("Jacobian det(F):", diag.min_jacobian)
    print("This is an integration smoke test, not a validated geotechnical prediction.")


if __name__ == "__main__":
    main()
