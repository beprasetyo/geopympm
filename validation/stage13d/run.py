"""Reproducible Stage 13D analytical/MPM validation runner.

From repository root:
    python -m validation.stage13d.run --output-dir results/stage13d

Does not alter the solver. The affine velocity Dirichlet field is enforced
at EVERY grid node, making this an interpolation/constitutive patch test,
NOT a freely evolving dynamic boundary-value or landslide problem.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import (
    MohrCoulomb, MC_HISTORY, MaterialState, mohr_coulomb_yield,
    update_mohr_coulomb_stress,
)
from geopympm.solvers import ExplicitMPMSolver
from .references import (plane_strain_confined_yield_strain,
                         tresca_pure_shear, triaxial_compression_strength)

E = 1000.0
NU = 0.3
COHESION = 10.0
DT = 1.0e-4
RATE = 20.0  # engineering shear gamma_dot [s^-1]
SEEDS = np.array([[.32, .32], [.46, .38], [.54, .52], [.68, .62]])
INTERPOLATIONS = ("mpm", "gimp", "cpgimp", "cpdi")
SCHEMES = ("usl", "usf", "musl")


def shear_material_point(nsteps: int, *, gamma_final: float = .08) -> list[dict]:
    """Pure shear from virgin state, stress/history compared to closed form."""
    if type(nsteps) is not int or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")
    material = MohrCoulomb(E, NU, COHESION, 0, 0)
    particles = ParticleSet([[.5,.5]], [1.], [.01])
    state = MaterialState(1, MC_HISTORY)
    records = []
    for step in range(1, nsteps + 1):
        increment = np.zeros((1, 6))
        increment[0, 3] = gamma_final / nsteps
        update_mohr_coulomb_stress(particles, increment, material, state)
        exact = tresca_pure_shear(step * gamma_final / nsteps, E, NU, COHESION)
        records.append({"step": step, "gamma": exact["gamma"],
                        "tau_numerical_pa": float(particles.stress[0, 3]),
                        "tau_exact_pa": exact["tau_pa"],
                        "eqp_numerical": float(state["equivalent_plastic_strain"][0]),
                        "eqp_exact": exact["equivalent_plastic_strain"],
                        "yield_residual_pa": float(mohr_coulomb_yield(particles.stress[0], material))})
    return records


def confined_compression_material_point(nsteps: int = 60, final_e: float = .05) -> dict:
    """Plane-strain confined compression: compare independently predicted onset."""
    if nsteps < 2 or final_e <= 0:
        raise ValueError("need positive strain and at least two steps")
    material = MohrCoulomb(E, .2, COHESION, 10., 0.)
    onset = plane_strain_confined_yield_strain(E, .2, COHESION, 10.)
    particles = ParticleSet([[.5, .5]], [1.], [.01])
    state = MaterialState(1, MC_HISTORY)
    first_plastic = None
    samples = []
    for step in range(1, nsteps + 1):
        inc = np.array([[-final_e / nsteps, 0, 0, 0, 0, 0]], dtype=float)
        update_mohr_coulomb_stress(particles, inc, material, state)
        e = final_e * step / nsteps
        eqp = float(state["equivalent_plastic_strain"][0])
        if eqp > 1.e-12 and first_plastic is None:
            first_plastic = e
        samples.append({"step": step, "compression_strain": e,
                        "sigma_xx_pa": float(particles.stress[0, 0]),
                        "sigma_yy_pa": float(particles.stress[0, 1]),
                        "sigma_zz_pa": float(particles.stress[0, 2]),
                        "eqp": eqp,
                        "yield_residual_pa": float(mohr_coulomb_yield(particles.stress[0], material))})
    return {"onset_analytical": onset, "first_plastic_sample": first_plastic,
            "increment_size": final_e / nsteps, "samples": samples}


def triaxial_yield_envelope(confinements=(0., 5., 10., 20., 40.),
                             cohesion=COHESION, friction_deg=30.) -> list[dict]:
    """Independent triaxial yield criterion, checked against 3D principal stresses.

    These states represent triaxial STRESSES, not a 2D triaxial experiment.
    """
    material = MohrCoulomb(E, NU, cohesion, friction_deg, 0)
    records = []
    for s3 in confinements:
        q = triaxial_compression_strength(s3, cohesion, friction_deg)
        stress = np.array([-s3-q, -s3, -s3, 0., 0., 0.])
        residual = mohr_coulomb_yield(stress, material)
        records.append({"confinement_compression_pa": float(s3),
                        "deviator_strength_pa": q,
                        "yield_residual_pa": residual})
    return records


def affine_shear_boundary_value(*, method: str = "mpm", scheme: str = "usl",
                                nsteps: int = 40) -> dict:
    """Fully prescribed affine shear (uniform-stress MPM patch test).

    Dirichlet velocities prescribed at all nodes remove dynamic stress-wave
    physics; this intentionally isolates interpolation/kinematics/material.
    """
    if method not in INTERPOLATIONS or scheme not in SCHEMES or nsteps < 1:
        raise ValueError("unknown method/scheme or invalid step count")
    p = ParticleSet(SEEDS.copy(), np.ones(len(SEEDS)), np.full(len(SEEDS), .01))
    if method != "mpm":
        p.domain_size[:] = .025
        p.domain_size0[:] = .025
    grid = BackgroundGrid([0., 0.], [.1,.1], (10,10))
    grid.velocity_bc_mask[:] = True
    grid.velocity_bc_value[:,0] = RATE * (grid.node_coordinates[:,1] - .5)
    grid.velocity_bc_value[:,1] = 0.
    material = MohrCoulomb(E, NU, COHESION, 0., 0.)
    solver = ExplicitMPMSolver(p, grid, material, dt=DT,
                               interpolation=method, scheme=scheme)
    for _ in range(nsteps):
        solver.step()
    expected = tresca_pure_shear(RATE * DT * nsteps, E, NU, COHESION)
    tau_errors = abs(p.stress[:, 3] - expected["tau_pa"])
    eqp_errors = abs(solver.material_state["equivalent_plastic_strain"]
                     - expected["equivalent_plastic_strain"])
    yield_max = max(mohr_coulomb_yield(p.stress[i], material)
                    for i in range(len(SEEDS)))
    return {"method": method, "scheme": scheme, "steps": nsteps,
            "total_engineering_shear": RATE * DT * nsteps,
            "tau_expected_pa": expected["tau_pa"],
            "max_tau_error_pa": float(tau_errors.max()),
            "max_eqp_error": float(eqp_errors.max()),
            "max_yield_function_pa": float(yield_max),
            "max_sigma_normal_abs_pa": float(np.max(np.abs(p.stress[:, :3]))),
            "total_mass_kg": float(np.sum(p.mass)),
            "min_jacobian": float(np.min(np.linalg.det(p.deformation_gradient)))}



def affine_confined_compression_boundary_value(*, method: str = "mpm",
                                               scheme: str = "usl",
                                               nsteps: int = 25) -> dict:
    """Plane-strain confined-compression kinematic patch test.

    All grid velocities are prescribed: v_x=-20*(x-0.5), v_y=0. The
    independently derived first-yield strain is compared to the observed
    onset. Full post-yield stress/history is additionally checked against
    the SAME strain path integrated separately at one material point.
    This is not a free-surface laboratory or field boundary-value analysis.
    """
    if method not in INTERPOLATIONS or scheme not in SCHEMES or nsteps < 18:
        raise ValueError("invalid method/scheme or insufficient steps to reach yield")
    positions = np.array([[.35,.35],[.45,.45],[.55,.55],[.65,.65]])
    p = ParticleSet(positions, np.ones(4), np.full(4, .01))
    if method != "mpm":
        p.domain_size[:] = .025
        p.domain_size0[:] = .025
    g = BackgroundGrid([0.,0.], [.1,.1], (10,10))
    g.velocity_bc_mask[:] = True
    g.velocity_bc_value[:,0] = -RATE * (g.node_coordinates[:,0] - .5)
    g.velocity_bc_value[:,1] = 0.
    material = MohrCoulomb(E, .2, COHESION, 10., 0.)
    solver = ExplicitMPMSolver(p, g, material, dt=DT,
                               interpolation=method, scheme=scheme)
    first_yield = None
    for step in range(1, nsteps + 1):
        solver.step()
        if (first_yield is None and
                np.any(solver.material_state['equivalent_plastic_strain'] > 1e-12)):
            first_yield = RATE * DT * step
    # Material-point check uses the same imposed engineering strain path.
    reference = ParticleSet([[.5, .5]], [1.], [.01])
    history = MaterialState(1, MC_HISTORY)
    for _ in range(nsteps):
        delta = np.array([[-RATE * DT, 0., 0., 0., 0., 0.]])
        update_mohr_coulomb_stress(reference, delta, material, history)
    stress_error = float(np.max(abs(p.stress[:,:4] - reference.stress[0,:4])))
    eqp_error = float(np.max(abs(solver.material_state['equivalent_plastic_strain']
                                 - history['equivalent_plastic_strain'][0])))
    analytic_onset = plane_strain_confined_yield_strain(E, .2, COHESION, 10.)
    return {"method": method, "scheme": scheme, "steps": nsteps,
            "analytical_yield_onset": analytic_onset,
            "first_plastic_sample": first_yield,
            "max_stress_error_pa": stress_error,
            "max_eqp_error": eqp_error,
            "max_yield_residual_pa": float(max(mohr_coulomb_yield(row, material)
                                                 for row in p.stress))}

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty result set")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def _plot_results(output_dir: Path, shear: list[dict], compression: dict,
                  envelope: list[dict]) -> None:
    """Optional plots; matplotlib intentionally NOT a runtime dependency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires optional matplotlib; install with pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r['gamma'] for r in shear], [r['tau_numerical_pa'] for r in shear],
            label='GeoPyMPM')
    ax.plot([r['gamma'] for r in shear], [r['tau_exact_pa'] for r in shear],
            '--', label='Analytic Tresca limit')
    ax.set(xlabel='Engineering shear strain, gamma', ylabel='Shear stress (Pa)',
           title='Stage 13D: Mohr-Coulomb pure shear, phi=psi=0')
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/'pure_shear_validation.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r['compression_strain'] for r in compression['samples']],
            [-r['sigma_xx_pa'] for r in compression['samples']],
            label='Compressive stress magnitude')
    ax.axvline(compression['onset_analytical'], linestyle='--',
               label='Analytic first yield')
    ax.set(xlabel='Compressive axial strain (positive)',
           ylabel='Compression stress magnitude (Pa)',
           title='Stage 13D: constrained plane-strain compression')
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/'confined_compression_validation.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r['confinement_compression_pa'] for r in envelope],
            [r['deviator_strength_pa'] for r in envelope], 'o-')
    ax.set(xlabel='Confining compression magnitude (Pa)',
           ylabel='Triaxial failure deviator q (Pa)',
           title='Stage 13D: independent Mohr-Coulomb yield envelope')
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_dir/'triaxial_yield_envelope.png', dpi=160)
    plt.close(fig)

def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=Path("results/stage13d"))
    cli.add_argument("--plot", action="store_true", help="create optional PNG plots (matplotlib)")
    cli.add_argument("--steps", type=int, default=40,
                     help="Number of MPM steps; at least 20 to reach pure-shear yield")
    cli.add_argument("--schemes", nargs="+", choices=SCHEMES,
                     default=list(SCHEMES))
    cli.add_argument("--interpolations", nargs="+", choices=INTERPOLATIONS,
                     default=list(INTERPOLATIONS))
    args = cli.parse_args(argv)
    if args.steps < 20:
        cli.error("--steps must be >=20 to exercise the plastic shear plateau")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shear = shear_material_point(80)
    compression = confined_compression_material_point()
    envelope = triaxial_yield_envelope()
    affine = [affine_shear_boundary_value(method=m, scheme=s, nsteps=args.steps)
              for m in args.interpolations for s in args.schemes]
    compressed = [affine_confined_compression_boundary_value(method=m, scheme=s)
                  for m in args.interpolations for s in args.schemes]
    _write_csv(args.output_dir / "pure_shear.csv", shear)
    _write_csv(args.output_dir / "confined_compression.csv", compression["samples"])
    _write_csv(args.output_dir / "triaxial_yield_envelope.csv", envelope)
    _write_csv(args.output_dir / "affine_shear_mpm.csv", affine)
    _write_csv(args.output_dir / "affine_confined_compression_mpm.csv", compressed)
    max_tau_error = max(row["max_tau_error_pa"] for row in affine)
    max_eqp_error = max(row["max_eqp_error"] for row in affine)
    max_normal = max(row["max_sigma_normal_abs_pa"] for row in affine)
    max_compression_stress_error = max(row["max_stress_error_pa"] for row in compressed)
    max_compression_eqp_error = max(row["max_eqp_error"] for row in compressed)
    compression_onset_grid_error = max(abs(row["first_plastic_sample"] - row["analytical_yield_onset"]) for row in compressed)
    onset_error = abs(compression["first_plastic_sample"] - compression["onset_analytical"])
    limits = {"max_tau_error_pa": 1e-7, "max_eqp_error": 1e-8,
              "max_normal_pa": 1e-7, "max_compression_stress_error_pa": 1e-7,
              "max_compression_eqp_error": 1e-8,
              "compression_onset_max_error": compression["increment_size"] + 1e-12,
              "compression_grid_onset_max_error": RATE * DT + 1e-12}
    passed = (max_tau_error < limits["max_tau_error_pa"]
              and max_eqp_error < limits["max_eqp_error"]
              and max_normal < limits["max_normal_pa"]
              and onset_error <= limits["compression_onset_max_error"]
              and max_compression_stress_error < limits["max_compression_stress_error_pa"]
              and max_compression_eqp_error < limits["max_compression_eqp_error"]
              and compression_onset_grid_error <= limits["compression_grid_onset_max_error"]
              and all(abs(r["yield_residual_pa"]) < 1e-7 for r in envelope))
    report = {"stage": "13D", "passed": bool(passed),
              "scope": "material-point + fully prescribed affine MPM patch test; no slope or GeoPyFEM comparison",
              "analytic_pure_shear": shear[-1],
              "confined_compression": {k:v for k,v in compression.items() if k!="samples"},
              "triaxial_envelope": envelope, "mpm_cases": len(affine),
              "compression_mpm_cases": len(compressed),
              "max_compression_stress_error_pa": max_compression_stress_error,
              "max_compression_eqp_error": max_compression_eqp_error,
              "compression_grid_onset_error": compression_onset_grid_error,
              "max_tau_error_pa": max_tau_error, "max_eqp_error": max_eqp_error,
              "max_normal_abs_pa": max_normal, "compression_onset_error": onset_error,
              "limits": limits}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        _plot_results(args.output_dir, shear, compression, envelope)
    print(f"Stage 13D: {'PASS' if passed else 'FAIL'} | {len(affine)} shear + {len(compressed)} confined MPM cases")
    print(f"max |tau-analytic| = {max_tau_error:.3e} Pa, max eqp error = {max_eqp_error:.3e}")
    print(f"confined-compression first-yield onset difference = {onset_error:.3e}")
    print(f"Output: {args.output_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
