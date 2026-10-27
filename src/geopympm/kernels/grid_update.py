"""Stage 3B: 2D Quad4 internal-force assembly and explicit nodal update.

Kernels use only scalar values and NumPy arrays. Their particle-to-node
accumulations are intentionally serial because many particles share nodes.

Cauchy stress convention: tension positive; Voigt [xx, yy, zz, xy, yz, xz].
Only in-plane [xx, yy, xy] enter the 2D internal-force assembly.
"""

from numba import njit


@njit(cache=True)
def assemble_internal_force_quad4_kernel(
    node_ids, gradients, particle_volume, particle_stress, node_force_internal
):
    """Accumulate f_int_i = -sum_p V_p * sigma_p @ grad(N_i).

    All output force entries must be initialized to zero by the caller.
    """
    for p in range(particle_volume.shape[0]):
        volume = particle_volume[p]
        sigma_xx = particle_stress[p, 0]
        sigma_yy = particle_stress[p, 1]
        sigma_xy = particle_stress[p, 3]

        for a in range(4):
            i = node_ids[p, a]
            gx = gradients[p, a, 0]
            gy = gradients[p, a, 1]
            node_force_internal[i, 0] -= volume * (sigma_xx * gx + sigma_xy * gy)
            node_force_internal[i, 1] -= volume * (sigma_xy * gx + sigma_yy * gy)


@njit(cache=True)
def explicit_grid_momentum_kernel(
    node_mass,
    node_momentum,
    node_velocity,
    node_acceleration,
    node_force_internal,
    node_force_external,
    velocity_bc_mask,
    velocity_bc_value,
    dt,
):
    """Forward Euler grid momentum update with componentwise velocity BCs.

    For an active node, p_new = p_old + dt * (f_int + f_ext).
    On prescribed-velocity DOFs, p_new = mass * v_prescribed instead.
    Acceleration is (p_new - p_old)/(mass * dt), including the constraint
    effect. Reaction forces are NOT explicitly stored in the grid.

    Massless nodes are retained at zero momentum, velocity, acceleration.
    """
    for i in range(node_mass.shape[0]):
        m = node_mass[i]
        if m > 0.0:
            for d in range(2):
                p_old = node_momentum[i, d]
                p_new = p_old + dt * (
                    node_force_internal[i, d] + node_force_external[i, d]
                )
                if velocity_bc_mask[i, d]:
                    p_new = m * velocity_bc_value[i, d]

                node_momentum[i, d] = p_new
                node_velocity[i, d] = p_new / m
                node_acceleration[i, d] = (p_new - p_old) / (m * dt)
        else:
            for d in range(2):
                node_momentum[i, d] = 0.0
                node_velocity[i, d] = 0.0
                node_acceleration[i, d] = 0.0
