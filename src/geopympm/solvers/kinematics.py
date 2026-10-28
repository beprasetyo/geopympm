"""Stage 3D: 2D Quad4 particle deformation and volume update.

Call AFTER the Stage-3B grid momentum update and BEFORE Stage-3C G2P moves
particle positions, using the SAME Stage-2 interpolation computed at x_n.
The old positions are needed to verify interpolation consistency. This stage
updates F, volume and the provisional diagnostic accumulated strain field.
It does not update position, velocity, stress, GIMP domains or material history.

Note: This is a standalone kinematic substep, not a validated full MPM solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.interpolation.mpm import Quad4Interpolation
from geopympm.interpolation.gimp import GIMPInterpolation, validate_current_gimp
from geopympm.kernels.interpolation_quad4 import quad4_kernel
from geopympm.kernels.kinematics import particle_kinematics_quad4_kernel
from geopympm.kernels.kinematics_exponential import particle_kinematics_quad4_exponential_kernel


@dataclass(slots=True)
class KinematicsResult:
    """Calculated per-particle quantities at the CURRENT grid update.

    ``strain_increment`` is the symmetric spatial velocity-gradient increment
    dt*D in engineering Voigt order [xx, yy, zz, xy, yz, xz]. The accumulated
    ``ParticleSet.strain`` is therefore diagnostic for small incremental
    deformations; it is NOT a finite-strain objective constitutive measure.
    """

    velocity_gradient: np.ndarray  # (n, 2, 2), 1/s
    strain_increment: np.ndarray   # (n, 6), dimensionless, engineering shear
    jacobian: np.ndarray           # (n,), det(F_new)


def update_particle_kinematics(
    particles: ParticleSet,
    grid: BackgroundGrid,
    interpolation: Quad4Interpolation | GIMPInterpolation,
    dt: float,
    *,
    kinematics_update: str = "euler",
) -> KinematicsResult:
    """Integrate particle F, volume and diagnostic strain from updated grid v.

    ``kinematics_update="euler"`` retains the Stage-3D baseline.
    ``kinematics_update="exponential"`` uses exp(dt*L)@F, avoiding artificial
    Jacobian drift for rigid rotation at finite dt. Both modes use D*dt as
    diagnostic strain, and neither changes the G2P position integration.

    Euler mode: F_(n+1) = (I + dt L_p) F_n.
    Exponential mode: F_(n+1) = exp(dt L_p) F_n.
    Both use V_(n+1) = V_0 det(F_(n+1)),
    L_p = sum_i v_i^(n+1) tensor grad N_i(x_p^n).

    Uses 2D plane strain (out-of-plane stretch 1): F stored as (n,2,2),
    strain stored as (n,6) with zero incremental zz, yz, xz components.
    In Stage-10 uGIMP mode ``domain_size`` is deliberately unchanged.

    Constraints:
    - Must be called after P2G/grid momentum update and before G2P.
    - The ``dt`` must match grid momentum update dt; current API does not
      carry a time-step generation identifier to verify this automatically.
    - A nonpositive or nonfinite Jacobian raises ValueError WITHOUT changing
      particle state. A positive Jacobian alone does not guarantee accuracy.
    - Euler F integration generates O(dt^2) Jacobian drift under rigid spin.
      The optional exponential map eliminates this particular drift.
    - Particle positions are still advanced with Stage-3C Euler G2P, and
      accumulated strain remains a diagnostic rather than finite strain.
      This is not by itself a complete large-deformation MPM formulation.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet.")
    if not isinstance(grid, BackgroundGrid):
        raise TypeError("grid must be a BackgroundGrid.")
    if not isinstance(interpolation, (Quad4Interpolation, GIMPInterpolation)):
        raise TypeError("interpolation must be Quad4Interpolation or GIMPInterpolation.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Stage 3D requires 2D particles and a 2D background grid.")
    if isinstance(dt, (bool, np.bool_)):
        raise ValueError("dt must be finite and strictly positive.")
    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be finite and strictly positive.") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive.")

    if not isinstance(kinematics_update, str) or kinematics_update not in ("euler", "exponential"):
        raise ValueError("kinematics_update must be 'euler' or 'exponential'.")

    is_gimp = isinstance(interpolation, GIMPInterpolation)
    n_support = 16 if is_gimp else 4
    n = particles.n_particles
    ids = interpolation.node_ids
    grad = interpolation.gradients
    cells = interpolation.cell_indices
    local = interpolation.local_coordinates
    weights = interpolation.weights

    if (ids.shape != (n, n_support) or ids.dtype != np.int64 or
            not ids.flags.c_contiguous or np.any(ids < 0) or
            np.any(ids >= grid.n_nodes)):
        raise ValueError("interpolation.node_ids must be valid contiguous (n,4) int64 IDs.")
    if (grad.shape != (n, n_support, 2) or grad.dtype != np.float64 or
            not grad.flags.c_contiguous or not np.all(np.isfinite(grad))):
        raise ValueError("interpolation.gradients must be finite contiguous (n,4,2) float64.")
    if (weights.shape != (n, n_support) or weights.dtype != np.float64 or
            not np.all(np.isfinite(weights)) or not weights.flags.c_contiguous or
            np.any(weights < 0.0) or not np.allclose(
                weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("interpolation.weights must be valid Quad4 weights.")
    if (cells.shape != (n, 2) or not np.issubdtype(cells.dtype, np.integer) or
            np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= grid.n_cells[0]) or
            np.any(cells[:, 1] < 0) or np.any(cells[:, 1] >= grid.n_cells[1])):
        raise ValueError("interpolation.cell_indices are invalid for this grid.")
    if (local.shape != (n, 2) or not np.all(np.isfinite(local)) or
            np.any(local < 0.0) or np.any(local > 1.0)):
        raise ValueError("interpolation.local_coordinates must lie in [0,1].")

    # Detect changes of particle positions, grid connectivity or cell sizes.
    uv = (particles.position - grid.origin) / grid.cell_size
    tol = 64.0 * np.finfo(np.float64).eps * max(1, *grid.n_cells)
    if (not np.all(np.isfinite(uv)) or
            np.any(np.abs(uv - (cells + local)) > tol)):
        raise ValueError("Interpolation is stale; call Stage 3D before G2P motion or re-evaluate.")
    if not is_gimp:
        expected_ids, expected_weights, expected_grad = quad4_kernel(
            np.ascontiguousarray(cells, dtype=np.int64),
            np.ascontiguousarray(local, dtype=np.float64),
            grid.n_cells[0], float(grid.cell_size[0]), float(grid.cell_size[1])
        )
        if (not np.array_equal(ids, expected_ids) or
                not np.allclose(weights, expected_weights, rtol=1e-13, atol=1e-13) or
                not np.allclose(grad, expected_grad, rtol=1e-13, atol=1e-13)):
            raise ValueError("Interpolation is inconsistent with current grid geometry.")

    else:
        validate_current_gimp(particles, grid, interpolation)

    if (grid.velocity.shape != (grid.n_nodes, 2) or
            not np.all(np.isfinite(grid.velocity))):
        raise ValueError("Updated nodal velocity must be finite with shape (n_nodes,2).")
    if (grid.mass.shape != (grid.n_nodes,) or
            not np.all(np.isfinite(grid.mass)) or np.any(grid.mass < 0.0) or
            not np.isclose(grid.mass.sum(), particles.mass.sum(), rtol=1e-11, atol=0.0)):
        raise ValueError("Grid mass must match particle mass; run P2G first.")
    if np.any((weights > 0.0) & (grid.mass[ids] <= 0.0)):
        raise ValueError("Particle supports include a massless node with positive weight.")
    if (particles.deformation_gradient.shape != (n, 2, 2) or
            not np.all(np.isfinite(particles.deformation_gradient))):
        raise ValueError("Particle F must be finite and shape (n,2,2).")
    if (particles.volume0.shape != (n,) or not np.all(np.isfinite(particles.volume0)) or
            np.any(particles.volume0 <= 0.0)):
        raise ValueError("Particle reference volume must be finite and positive.")
    if (particles.volume.shape != (n,) or not np.all(np.isfinite(particles.volume)) or
            np.any(particles.volume <= 0.0)):
        raise ValueError("Particle current volume must be finite and positive.")
    if (particles.strain.shape != (n, 6) or
            not np.all(np.isfinite(particles.strain))):
        raise ValueError("Particle strain must be finite and shape (n,6).")

    F = particles.deformation_gradient
    det_old = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    if not np.all(np.isfinite(det_old)) or np.any(det_old <= 0.0):
        raise ValueError("Particle deformation gradients must have positive Jacobians.")
    if not np.allclose(particles.volume, det_old * particles.volume0, rtol=1e-10, atol=1e-14):
        raise ValueError("Particle volume is inconsistent with det(F)*volume0.")

    kernel = (particle_kinematics_quad4_exponential_kernel
              if kinematics_update == "exponential" else particle_kinematics_quad4_kernel)
    L, inc, new_F, J, new_volume, new_strain = kernel(
        ids, grad, grid.velocity, F, particles.volume0, particles.strain, dt,
    )
    if (not np.all(np.isfinite(L)) or not np.all(np.isfinite(inc)) or
            not np.all(np.isfinite(new_F)) or not np.all(np.isfinite(J)) or
            not np.all(np.isfinite(new_volume)) or not np.all(np.isfinite(new_strain))):
        raise ValueError("Stage 3D produced a nonfinite kinematic state; no changes committed.")
    if np.any(J <= 0.0) or np.any(new_volume <= 0.0):
        p = int(np.flatnonzero((J <= 0.0) | (new_volume <= 0.0))[0])
        raise ValueError(
            f"Particle {p} has nonpositive deformation Jacobian after update; "
            "reduce dt or inspect the velocity field. No changes committed."
        )

    # Commit only after every particle's candidate state has passed validation.
    particles.deformation_gradient[:] = new_F
    particles.volume[:] = new_volume
    particles.strain[:] = new_strain
    return KinematicsResult(velocity_gradient=L, strain_increment=inc, jacobian=J)
