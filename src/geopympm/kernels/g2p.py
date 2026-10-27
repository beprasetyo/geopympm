"""Stage 3C: Numba 2D Quad4 G2P kinematic transfer.

Use grid values AFTER an explicit momentum update:
    PIC  velocity  = sum_i N_ip * v_i^{n+1}
    FLIP velocity  = v_p^n + dt * sum_i N_ip * a_i
    Blend velocity = pic_fraction * PIC + (1-pic_fraction) * FLIP
    Position       = x_p^n + dt * PIC velocity

Here ``a_i`` is the Stage-3B acceleration from the change in nodal momentum,
including effects of prescribed velocity BCs. This is deliberately only a
kinematic transfer; strain, deformation gradient, volume and stress remain
unchanged until subsequent stages.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def g2p_quad4_kernel(
    node_ids,
    weights,
    particle_position,
    particle_velocity,
    node_velocity,
    node_acceleration,
    dt,
    pic_fraction,
):
    """Return independent (new_position, new_velocity) arrays.

    The caller validates shapes, masses, bounds and interpolation freshness.
    Producing temporary arrays allows it to reject any out-of-grid particle
    before changing even one existing particle state.
    """
    n = particle_position.shape[0]
    new_position = np.empty((n, 2), dtype=np.float64)
    new_velocity = np.empty((n, 2), dtype=np.float64)

    for p in range(n):
        for d in range(2):
            v_pic = 0.0
            dv_flip = 0.0
            for a in range(4):
                i = node_ids[p, a]
                w = weights[p, a]
                v_pic += w * node_velocity[i, d]
                dv_flip += w * node_acceleration[i, d]

            v_flip = particle_velocity[p, d] + dt * dv_flip
            new_velocity[p, d] = (
                pic_fraction * v_pic + (1.0 - pic_fraction) * v_flip
            )
            # Grid velocity advects particles even for FLIP particle velocity;
            # this intentional position-update convention must remain explicit.
            new_position[p, d] = particle_position[p, d] + dt * v_pic

    return new_position, new_velocity
