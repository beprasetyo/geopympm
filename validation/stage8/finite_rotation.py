"""Stage 8: rigid-rotation comparison for Euler and exponential-map F updates.

The full solver uses prescribed rotational nodal velocities to isolate the
kinematics while retaining objective Stage-7 stress integration. G2P remains
Euler advection; this test does NOT establish exact particle trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from validation.stage7.objective_rotation import rigid_spin


@dataclass(slots=True)
class RotationComparison:
    passed: bool
    metrics: dict[str, float]
    history: list[dict[str, float]]
    notes: str


def rotation_comparison(*, dt: float = 0.001, n_steps: int = 100,
                        angular_speed: float = 3.0) -> RotationComparison:
    euler = rigid_spin(stress_update="corotational", kinematics_update="euler",
                       dt=dt, n_steps=n_steps, angular_speed=angular_speed)
    exponential = rigid_spin(stress_update="corotational", kinematics_update="exponential",
                             dt=dt, n_steps=n_steps, angular_speed=angular_speed)
    euler_history = euler.history
    exponential_history = exponential.history
    history = [dict(step=int(a['step']), time=float(a['time']),
                    jacobian_euler=float(a['jacobian_max']),
                    jacobian_exponential=float(b['jacobian_max']),
                    j_error_euler=float(abs(a['jacobian_max']-1.0)),
                    j_error_exponential=float(abs(b['jacobian_max']-1.0)),
                    stress_error_exponential_pa=float(b['stress_objectivity_error_pa']),
                    continuum_position_error_m=float(b['continuum_position_error_m']))
               for a,b in zip(euler_history, exponential_history)]
    metrics = {
        'euler_j_final': float(euler_history[-1]['jacobian_max']),
        'exponential_j_final': float(exponential_history[-1]['jacobian_max']),
        'euler_j_error_max': max(row['j_error_euler'] for row in history),
        'exponential_j_error_max': max(row['j_error_exponential'] for row in history),
        'exponential_stress_error_pa': float(exponential.metrics['stress_objectivity_error_pa']),
        'euler_position_error_end_m': float(euler.metrics['continuum_position_error_end_m']),
        'exponential_position_error_end_m': float(exponential.metrics['continuum_position_error_end_m']),
    }
    passed = bool(
        euler.passed and exponential.passed
        and metrics['euler_j_error_max'] > 1e-5
        and metrics['exponential_j_error_max'] < 5e-12
        and metrics['exponential_stress_error_pa'] < 1e-8
        and np.isclose(metrics['euler_position_error_end_m'],
                       metrics['exponential_position_error_end_m'], rtol=0, atol=5e-13)
    )
    return RotationComparison(passed, metrics, history,
        'Prescribed rigid spin: F exponential preserves det(F)=1. Particle '
        'positions still use Stage-3C Euler, so continuum trajectory error '
        'does not improve. Diagnostic strain remains D*dt.')
