"""Stage 9: standalone full-solver local-affine advection verification.

Two comparisons: prescribed pure spin (rigid-rotation trajectory) and
stationary affine velocity with nonzero translation and shear. These isolate
position-advection quality, not unconstrained stress/contact dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import cos, sin
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver
from validation.stage7.objective_rotation import rigid_spin


@dataclass(slots=True)
class AdvectionResult:
    name: str
    passed: bool
    metrics: dict[str, float]
    history: list[dict[str, float]]
    notes: str


def rigid_rotation_comparison(dt: float = 0.001, steps: int = 100,
                              omega: float = 3.0) -> AdvectionResult:
    old = rigid_spin(dt=dt, n_steps=steps, angular_speed=omega,
                     stress_update='corotational', kinematics_update='exponential',
                     advection_update='euler')
    new = rigid_spin(dt=dt, n_steps=steps, angular_speed=omega,
                     stress_update='corotational', kinematics_update='exponential',
                     advection_update='affine_exponential')
    history = [dict(step=int(a['step']), time=float(a['time']),
                    euler_position_error_m=float(a['continuum_position_error_m']),
                    affine_exponential_position_error_m=float(b['continuum_position_error_m']),
                    exp_j_error=float(abs(b['jacobian_max']-1.)),
                    objective_stress_error_pa=float(b['stress_objectivity_error_pa']))
               for a,b in zip(old.history,new.history)]
    metrics = dict(euler_position_error_end_m=float(old.metrics['continuum_position_error_end_m']),
                   affine_exponential_position_error_end_m=float(new.metrics['continuum_position_error_end_m']),
                   exp_j_error_max=max(v['exp_j_error'] for v in history),
                   stress_error_max_pa=max(v['objective_stress_error_pa'] for v in history),
                   spin_angle_rad=float(dt*steps*omega))
    passed = bool(old.passed and new.passed and
                  metrics['euler_position_error_end_m']>1e-5 and
                  metrics['affine_exponential_position_error_end_m']<3e-12 and
                  metrics['exp_j_error_max']<1e-11 and
                  metrics['stress_error_max_pa']<1e-8)
    return AdvectionResult('9A prescribed rigid rotation',passed,metrics,history,
          'PIC nodal velocities imposed; updated particle stress uses corotational law. '
          'Rotational advection is checked against exact trajectory, not free dynamics.')


def _expm3_reference(B: np.ndarray) -> np.ndarray:
    """Independent Taylor-series reference for small 3x3 augmented affine flow."""
    power = np.eye(3)
    expB = power.copy()
    for k in range(1,60):
        power = power@B/k
        expB += power
    return expB


def affine_flow(*, dt=0.002, steps=80, pic_fraction=1.) -> AdvectionResult:
    """v(x)=A x+b, prescribed grid velocity; exact via augmented 3x3 exp."""
    A = np.array([[0.18,-0.45],[0.30,-0.10]],dtype=float)
    b = np.array([0.05,-0.03],dtype=float)
    X = np.array([[x,y] for x in (.25,.45,.65) for y in (.25,.40,.60)])
    n=len(X)
    grid=BackgroundGrid(origin=(-1,-1),cell_size=(.25,.25),n_cells=(12,12))
    p=ParticleSet(position=X.copy(),mass=np.full(n,2.),volume=np.full(n,.01),
                  velocity=X@A.T+b)
    grid.velocity_bc_mask[:]=True
    grid.velocity_bc_value[:]=grid.node_coordinates@A.T+b
    solver=ExplicitMPMSolver(p,grid,LinearElastic(10_000.,.25),dt=dt,
                             pic_fraction=pic_fraction, cfl=.5,
                             stress_update='corotational', kinematics_update='exponential',
                             advection_update='affine_exponential')
    B=np.zeros((3,3))
    B[:2,:2]=A
    B[:2,2]=b
    Xref=X.copy()
    ref_M=_expm3_reference(dt*B)
    history=[]
    Fref=np.eye(2)
    for i in range(steps):
        solver.step()
        Xref = Xref@ref_M[:2,:2].T+ref_M[:2,2]
        Fref=ref_M[:2,:2]@Fref
        history.append(dict(step=i+1,time=solver.time,
                            position_error_m=float(np.max(np.abs(p.position-Xref))),
                            F_error=float(np.max(np.abs(p.deformation_gradient-Fref))),
                            volume_error=float(np.max(np.abs(p.volume-p.volume0*np.linalg.det(Fref))))))
    metrics={name: max(row[name] for row in history) for name in
             ('position_error_m','F_error','volume_error')}
    metrics['pic_fraction']=float(pic_fraction)
    passed=bool(metrics['position_error_m']<5e-12 and metrics['F_error']<1e-11 and
                metrics['volume_error']<1e-11)
    return AdvectionResult('9B stationary affine flow',passed,metrics,history,
        'Prescribed affine nodal velocity; independent augmented 3x3 exponential '
        'reference. Velocity PIC/FLIP treatment remains unchanged.')
