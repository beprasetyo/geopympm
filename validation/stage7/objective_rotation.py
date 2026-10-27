"""Stage 7: imposed rigid-spin objective stress benchmark.

A prescribed v(x)=W x is reproduced exactly by Quad4 nodal interpolation.
All nodal DOFs are prescribed to deliberately isolate STRESS OBJECTIVITY
from force-driven rigid body motion. Stage-3D still integrates F and positions
with forward Euler, which produces a known O(dt) finite-rotation drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.materials import LinearElastic
from geopympm.solvers.explicit import ExplicitMPMSolver


def rotate_stress_reference(old: np.ndarray, angle: float) -> np.ndarray:
    """Analytical 2D plane-strain Cauchy stress after pure rigid rotation."""
    c, s = cos(angle), sin(angle)
    R = np.array([[c, -s], [s, c]])
    sigma = np.array([[old[0], old[3]], [old[3], old[1]]])
    rotated = R @ sigma @ R.T
    result = old.copy()
    result[0], result[1], result[3] = rotated[0, 0], rotated[1, 1], rotated[0, 1]
    return result


@dataclass(slots=True)
class RotationRun:
    name: str
    passed: bool
    metrics: dict[str, float | int]
    thresholds: dict[str, float]
    history: list[dict[str, float | int]]
    notes: str


def rigid_spin(*, stress_update: str = "corotational", dt: float = 0.001,
               n_steps: int = 100, angular_speed: float = 3.0) -> RotationRun:
    """Full explicit-solver test with stress rotation but imposed nodal motion."""
    if stress_update not in ("incremental", "corotational"):
        raise ValueError("Invalid stress_update.")
    if not np.isfinite(dt) or dt <= 0 or isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("Require positive dt and n_steps.")
    if not np.isfinite(angular_speed):
        raise ValueError("angular_speed must be finite.")

    grid = BackgroundGrid(origin=(-1.0, -1.0), cell_size=(0.25, 0.25), n_cells=(8, 8))
    X = np.array([[x, y] for y in (-0.36, -0.12, 0.12, 0.36)
                  for x in (-0.36, -0.12, 0.12, 0.36)], dtype=float)
    n = len(X)
    sigma0 = np.array([100.0, -40.0, 19.0, 12.0, 0.0, 0.0], dtype=float)
    stress = np.repeat(sigma0[None, :], n, axis=0)
    p = ParticleSet(position=X, mass=np.full(n, 6.25), volume=np.full(n, 0.00625), stress=stress)

    # v=(-omega*y,omega*x) prescribed on all grid nodes. This is a kinematic
    # stress-objectivity test, NOT a dynamically free rigid-body rotation.
    grid.velocity_bc_mask[:, :] = True
    grid.velocity_bc_value[:, 0] = -angular_speed * grid.node_coordinates[:, 1]
    grid.velocity_bc_value[:, 1] = angular_speed * grid.node_coordinates[:, 0]
    solver = ExplicitMPMSolver(p, grid, LinearElastic(50_000.0, 0.25), dt=dt,
                               pic_fraction=1.0, stress_update=stress_update)
    eye = np.eye(2)
    W = np.array([[0.0, -angular_speed], [angular_speed, 0.0]])
    M = eye + dt*W
    discrete_reference = X.copy()
    history = []
    for k in range(1, n_steps+1):
        solver.step()
        discrete_reference = discrete_reference @ M.T
        t = solver.time
        expected = rotate_stress_reference(sigma0, angular_speed*t)
        R = np.array([[cos(angular_speed*t), -sin(angular_speed*t)],
                      [sin(angular_speed*t),  cos(angular_speed*t)]])
        physical_reference = X @ R.T
        # Stress metrics including the out-of-plane and shear components.
        history.append(dict(
            step=k, time=float(t),
            stress_objectivity_error_pa=float(np.max(np.abs(p.stress-expected))),
            stress_change_from_initial_pa=float(np.max(np.abs(p.stress-sigma0))),
            inplane_trace_error_pa=float(np.max(np.abs(p.stress[:, 0]+p.stress[:, 1]-(sigma0[0]+sigma0[1])))),
            sigma_zz_error_pa=float(np.max(np.abs(p.stress[:, 2]-sigma0[2]))),
            strain_abs_max=float(np.max(np.abs(p.strain))),
            discrete_position_error_m=float(np.max(np.abs(p.position-discrete_reference))),
            continuum_position_error_m=float(np.max(np.abs(p.position-physical_reference))),
            volume_ratio_max=float(np.max(p.volume/p.volume0)),
            jacobian_max=float(np.max(np.linalg.det(p.deformation_gradient))),
        ))
    thresholds = dict(stress_objectivity_error_pa=1e-8,
                      inplane_trace_error_pa=1e-8, sigma_zz_error_pa=1e-8,
                      strain_abs_max=1e-10, discrete_position_error_m=1e-10)
    metrics = {key: max(float(row[key]) for row in history) for key in thresholds}
    metrics.update(final_angle_rad=float(n_steps*dt*angular_speed),
                   continuum_position_error_end_m=float(history[-1]['continuum_position_error_m']),
                   jacobian_final=float(history[-1]['jacobian_max']),
                   particles=n, steps=n_steps)
    # The non-objective legacy variant is intentionally only a comparison.
    passed = (all(metrics[k] <= v for k,v in thresholds.items())
              if stress_update == 'corotational' else False)
    return RotationRun(
        name=f"Stage 7 prescribed rigid spin ({stress_update})", passed=passed,
        metrics=metrics, thresholds=thresholds, history=history,
        notes=("All nodal velocities prescribed; verifies stress objectivity in a full "
               "solver step. Stage-3D F and G2P positions still use forward Euler, "
               "so J and continuum rotation trajectory drift by O(dt). "
               "A non-objective legacy comparison is expected NOT to pass."),
    )
