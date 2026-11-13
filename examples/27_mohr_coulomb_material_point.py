"""Stage 13B: smooth-face Mohr–Coulomb direct material-point shear test.

Example only: no grid/MPM dynamics. Uses the same constitutive function
that ExplicitMPMSolver calls through the Stage 13A adapter.

Run: python examples/27_mohr_coulomb_material_point.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from geopympm.core import ParticleSet
from geopympm.materials import (
    MohrCoulomb, MaterialState, MC_HISTORY, mohr_coulomb_yield,
    update_mohr_coulomb_stress,
)


def run(output="results/stage13b_mc_shear.csv", *, steps=100, dgamma=0.001):
    if steps < 1 or dgamma <= 0:
        raise ValueError("steps and dgamma must be positive")
    p = ParticleSet([[0.5, 0.5]], [1.0], [0.01])
    mat = MohrCoulomb(young_modulus=1000., poisson_ratio=0.3, cohesion=10.0,
                    friction_angle_deg=30.0, dilation_angle_deg=0.0)
    state = MaterialState(p.n_particles, MC_HISTORY)
    rows = []
    for step in range(steps + 1):
        rows.append({
            "step": step, "gamma_xy": step * dgamma,
            "tau_xy_pa": float(p.stress[0, 3]),
            "yield_f_pa": mohr_coulomb_yield(p.stress[0], mat),
            "eq_plastic_strain": float(state["equivalent_plastic_strain"][0]),
        })
        if step < steps:
            update_mohr_coulomb_stress(
                p, np.array([[0., 0., 0., dgamma, 0., 0.]]), mat, state
            )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Material:", mat)
    print("Final shear stress [Pa]:", rows[-1]["tau_xy_pa"])
    print("Final equivalent plastic strain:", rows[-1]["eq_plastic_strain"])
    print("CSV:", destination)
    return rows


if __name__ == "__main__":
    run()
