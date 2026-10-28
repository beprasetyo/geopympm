"""Stage 10C: 2D CPDI1 parallelogram domains convected by full F.

Reference full side lengths ``domain_size0`` define the two initial
half-edge vectors. Current corners = position ± F*a0 ± F*b0.
The existing ``domain_size`` field is the *axis-aligned bounding-box*
full width of the convected parallelogram in CPDI mode only. It is not
used to evaluate CPDI weights; F and domain_size0 own domain geometry.
No corner state is allocated per particle or persisted across steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.gimp import GIMPInterpolation
from geopympm.kernels.interpolation_cpdi import cpdi2d_kernel


@dataclass(slots=True)
class CPDIInterpolation(GIMPInterpolation):
    """GIMP-compatible 16-slot transfer, with a snapshot of current F."""
    deformation_gradient: np.ndarray


def candidate_cpdi_geometry(particles: ParticleSet, grid: BackgroundGrid,
                            *, position=None, deformation_gradient=None):
    """Return corners, two half-edge vectors, bounding full widths.

    Validates positive orientation, finite noncollapsed geometry and all
    four corners within the *fixed* background grid. Does not cap stretch
    at one cell; a corner still contributes only four Quad4 grid nodes.
    """
    if not isinstance(particles, ParticleSet) or not isinstance(grid, BackgroundGrid):
        raise TypeError('CPDI requires ParticleSet and BackgroundGrid.')
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError('Stage 10C CPDI1 requires two dimensions.')
    x = np.asarray(particles.position if position is None else position)
    F = np.asarray(particles.deformation_gradient if deformation_gradient is None
                   else deformation_gradient)
    size = particles.domain_size0
    n = particles.n_particles
    if (x.shape != (n, 2) or F.shape != (n, 2, 2)
            or size.shape != (n, 2) or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(F)) or not np.all(np.isfinite(size))
            or np.any(size <= 0)):
        raise ValueError('CPDI needs finite positions, full F, and positive domain_size0.')
    a = F[:, :, 0] * (size[:, 0, None] * .5)
    b = F[:, :, 1] * (size[:, 1, None] * .5)
    cross = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    scale = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(~np.isfinite(cross)) or np.any(~np.isfinite(scale)) or np.any(
        cross <= 1e-12*scale
    ):
        raise ValueError('CPDI domain has nonpositive/near-singular oriented area det(F).')
    corners = np.stack([x-a-b, x+a-b, x+a+b, x-a+b], axis=1)
    widths = 2*(np.abs(a) + np.abs(b))
    lo = grid.origin
    hi = lo + grid.cell_size * np.asarray(grid.n_cells)
    tol = 64*np.finfo(float).eps * np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1.)
    if np.any(corners < (lo-tol)[None, None, :]) or np.any(
        corners > (hi+tol)[None, None, :]
    ):
        raise ValueError('CPDI particle domain corners extend outside background grid.')
    return (np.ascontiguousarray(corners, dtype=np.float64),
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            np.ascontiguousarray(widths, dtype=np.float64))


def update_cpdi_domain(particles: ParticleSet, grid: BackgroundGrid) -> np.ndarray:
    """Commit CPDI axis-aligned envelope; no mutation if validation fails."""
    _, _, _, widths = candidate_cpdi_geometry(particles, grid)
    particles.domain_size[:] = widths
    return widths


def evaluate_cpdi(particles: ParticleSet, grid: BackgroundGrid) -> CPDIInterpolation:
    """Evaluate CPDI1 corner-weighted shape functions and analytical gradients."""
    corners, a, b, widths = candidate_cpdi_geometry(particles, grid)
    if not np.allclose(widths, particles.domain_size, rtol=1e-12, atol=1e-14):
        raise ValueError('CPDI current domain_size envelope is stale; update_cpdi_domain.')
    uv = (particles.position - grid.origin) / grid.cell_size
    cells = np.minimum(np.floor(np.maximum(uv, 0)).astype(np.int64),
                       np.asarray(grid.n_cells)-1)
    local = uv - cells
    ids, w, grads = cpdi2d_kernel(
        corners, a, b, grid.n_cells[0], grid.n_cells[1],
        float(grid.origin[0]), float(grid.origin[1]),
        float(grid.cell_size[0]), float(grid.cell_size[1]))
    if (not np.all(np.isfinite(w)) or not np.all(np.isfinite(grads))
            or np.any(w < -1e-14) or not np.allclose(w.sum(axis=1), 1., atol=2e-12, rtol=0.)
            or not np.allclose(grads.sum(axis=1), 0., atol=2e-11/min(grid.cell_size), rtol=0.)):
        raise ValueError('CPDI basis failed partition of unity / gradient consistency.')
    return CPDIInterpolation(np.ascontiguousarray(cells), np.ascontiguousarray(local),
                             ids, w, grads, widths.copy(),
                             particles.deformation_gradient.copy())


def validate_current_cpdi(particles: ParticleSet, grid: BackgroundGrid,
                          interpolation: CPDIInterpolation, *,
                          allow_updated_f: bool = False) -> None:
    """Check CPDI snapshot; optionally permit F updated in the current step.

    The integrated explicit solver updates F during the kinematic substep,
    but must use the OLD CPDI interpolation at the G2P stage. In that
    specific case, verify shape functions against the F snapshot instead.
    """
    if not isinstance(interpolation, CPDIInterpolation):
        raise TypeError('Expected a CPDIInterpolation.')
    if not np.array_equal(interpolation.domain_size, particles.domain_size):
        raise ValueError('CPDI interpolation is stale: domain changed.')
    if not allow_updated_f and not np.array_equal(
        interpolation.deformation_gradient, particles.deformation_gradient
    ):
        raise ValueError('CPDI interpolation is stale: F changed.')
    corners, a, b, widths = candidate_cpdi_geometry(
        particles, grid, deformation_gradient=interpolation.deformation_gradient)
    uv = (particles.position-grid.origin)/grid.cell_size
    cells = np.minimum(np.floor(np.maximum(uv, 0)).astype(np.int64),
                       np.asarray(grid.n_cells)-1)
    local = uv - cells
    ids, w, grads = cpdi2d_kernel(
        corners, a, b, grid.n_cells[0], grid.n_cells[1],
        float(grid.origin[0]), float(grid.origin[1]),
        float(grid.cell_size[0]), float(grid.cell_size[1]))
    if (not np.allclose(widths, interpolation.domain_size, atol=1e-14, rtol=1e-12)
            or not np.array_equal(ids, interpolation.node_ids)
            or not np.array_equal(cells, interpolation.cell_indices)
            or not np.allclose(local, interpolation.local_coordinates,
                               atol=1e-13, rtol=0)
            or not np.allclose(w, interpolation.weights, atol=1e-13, rtol=1e-12)
            or not np.allclose(grads, interpolation.gradients, atol=1e-12, rtol=1e-12)):
        raise ValueError('CPDI interpolation is stale or inconsistent with current grid.')
