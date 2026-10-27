"""Stage 3A: mass and momentum projection from particles to a 2D grid.

Hot loops receive only NumPy arrays/scalars and are compiled by Numba.
Loop ordering is deliberately serial: multiple particles may write to the
same grid node, so naive ``prange`` would introduce accumulation races.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def p2g_mass_momentum_kernel(node_ids, weights, particle_mass, particle_velocity,
                             node_mass, node_momentum):
    """Accumulate lumped nodal mass and nodal momentum (not force or stress).

    node_ids[p, a] is the global grid-node ID supporting particle p at local
    Quad4 node a. The destination arrays must be initialized by the caller.
    """
    for p in range(particle_mass.shape[0]):
        mp = particle_mass[p]
        vx = particle_velocity[p, 0]
        vy = particle_velocity[p, 1]
        for a in range(4):
            i = node_ids[p, a]
            weighted_mass = weights[p, a] * mp
            node_mass[i] += weighted_mass
            node_momentum[i, 0] += weighted_mass * vx
            node_momentum[i, 1] += weighted_mass * vy


@njit(cache=True)
def recover_nodal_velocity_kernel(node_mass, node_momentum, node_velocity):
    """Compute velocity = momentum/mass on active nodes, zero otherwise.

    This is the projected (pre-force, pre-BC) nodal velocity, not the result
    of an explicit time update. Nodes with exactly zero mass stay at zero.
    """
    for i in range(node_mass.shape[0]):
        if node_mass[i] > 0.0:
            node_velocity[i, 0] = node_momentum[i, 0] / node_mass[i]
            node_velocity[i, 1] = node_momentum[i, 1] / node_mass[i]
        else:
            node_velocity[i, 0] = 0.0
            node_velocity[i, 1] = 0.0
