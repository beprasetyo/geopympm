"""Stage 13C: 3 material-point increments demonstrating MC face, edge, apex.

Run at repository root:
    python examples/28_mohr_coulomb_multisurface_material_point.py

No mesh/solver is needed here: this isolates the constitutive stress update.
The dilation angle ψ=0 does not permit correction of excess hydrostatic tension
and will instead cause a clear ValueError with no state changes.
"""
from pathlib import Path
import csv
import numpy as np
from geopympm.core import ParticleSet
from geopympm.materials import (
    MohrCoulomb, MaterialState, MC_HISTORY,
    update_mohr_coulomb_stress, mohr_coulomb_yield,
)


def main():
    increments = np.array([
        [-.14993903, .12121359, 0, -.32762793, 0, 0],   # smooth face, ψ=0
        [.21555855, -.19711230, 0, -.25622659, 0, 0],   # edge, ψ=0
        [.01072789, .13656687, 0, .06050925, 0, 0],    # apex, ψ=15°
    ], dtype=np.float64)
    labels = ('face', 'edge', 'apex')
    mat = {0: MohrCoulomb(1000, .3, 10, 30, 0),
           1: MohrCoulomb(1000, .3, 10, 30, 0),
           2: MohrCoulomb(1000, .3, 10, 30, 15)}
    p = ParticleSet([[.2,.5],[.5,.5],[.8,.5]], [1.,1.,1.], [.01,.01,.01],
                    material_id=[0,1,2])
    history = MaterialState(3, MC_HISTORY)
    update_mohr_coulomb_stress(p, increments, mat, history)
    out = Path('results/stage13c_mc_multisurface.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('branch', 'sigma_xx_Pa', 'sigma_yy_Pa', 'sigma_zz_Pa',
                    'sigma_xy_Pa', 'yield_residual_Pa', 'equivalent_plastic_strain'))
        for i, label in enumerate(labels):
            residual = mohr_coulomb_yield(p.stress[i], mat[i])
            w.writerow([label, *p.stress[i,:4], residual,
                        history['equivalent_plastic_strain'][i]])
            print(f'{label:>5}: f={residual:.3e} Pa, '
                  f'eqp={history["equivalent_plastic_strain"][i]:.6f}')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
