"""Produce reproducible Stage 10 point MPM vs uGIMP cell-crossing force data.

    python -m validation.stage10.run_crossing [--output results/stage10]

This is a prescribed-particle-position diagnostic: it is NOT a dynamic solve.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_quad4, evaluate_gimp
from geopympm.solvers.p2g import particle_to_grid
from geopympm.solvers.grid_update import update_grid_momentum


def run(output: str = 'validation/results/stage10'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    grid = BackgroundGrid((0,0), (1,1), (4,4))
    node_id = grid.node_id(1,1)
    rows = []
    for x in np.linspace(0.75, 1.25, 101):
        p = ParticleSet([[x, 1.5]], mass=[1], volume=[0.2],
                        stress=[[100,0,0,0,0,0]], domain_size=[[0.5,0.5]])
        values = []
        for func in (evaluate_quad4, evaluate_gimp):
            ip = func(p, grid)
            particle_to_grid(p, grid, ip)
            update_grid_momentum(p, grid, ip, 0.001)
            values.append(float(grid.force_internal[node_id,0]))
        rows.append((float(x), *values))
    output_csv = out/'10_cell_crossing_forces.csv'
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(('x_particle_m','f_x_mpm_N', 'f_x_gimp_N'))
        w.writerows(rows)
    data = np.asarray(rows)
    delta_mpm = np.max(np.abs(np.diff(data[:,1])))
    delta_gimp = np.max(np.abs(np.diff(data[:,2])))
    print(f'MPM maximum adjacent force change: {delta_mpm:.6g} N')
    print(f'uGIMP maximum adjacent force change: {delta_gimp:.6g} N')
    print(f'Result: {output_csv}')
    return delta_mpm, delta_gimp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='validation/results/stage10')
    args = parser.parse_args()
    run(args.output)
