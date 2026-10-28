"""Stage 10: side-by-side standard MPM / uGIMP cell-boundary comparison.

Run from repository root after editable install:
    python examples/11_gimp_interpolation.py
"""
from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_gimp, evaluate_quad4
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def main():
    grid = BackgroundGrid(origin=(0, 0), cell_size=(1, 1), n_cells=(5, 4))
    particle = ParticleSet(
        position=[[1.95, 2.1]], mass=[1.0], volume=[0.2],
        velocity=[[0.5, 0.0]], domain_size=[[0.5, 0.5]],
    )
    for name, result in (("MPM", evaluate_quad4(particle, grid)),
                         ("uGIMP", evaluate_gimp(particle, grid))):
        print(name, ': positive support nodes =', (result.weights[0] > 0).sum())
        print('    sum weights =', result.weights.sum())
        print('    sum gradients =', result.gradients.sum(axis=1)[0])
    solver = ExplicitMPMSolver(
        particle, grid, LinearElastic(100.0, 0.25),
        dt=0.01, interpolation='gimp', pic_fraction=0.0,
        stress_update='corotational', kinematics_update='exponential',
        advection_update='affine_exponential',
    )
    diagnostics = solver.run(20)
    print('Final particle position:', particle.position[0])
    print('Final particle velocity:', particle.velocity[0])
    print('Final mass:', diagnostics[-1].total_mass)
    print('Final domain_size (uGIMP, unchanged):', particle.domain_size[0])


if __name__ == '__main__':
    main()
