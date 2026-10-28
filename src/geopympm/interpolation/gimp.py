"""Stage 10: axis-aligned finite-domain GIMP interpolation in two dimensions.

Each particle domain is a finite axis-aligned rectangle, full side lengths
``ParticleSet.domain_size``. Nodal weights and their spatial derivatives are
integrated against the usual piecewise-linear grid hats. They are not Quad4
point evaluations and generally span nodes from several neighboring cells.

Stage 10A keeps domain_size fixed (uGIMP). Stage 10B optionally updates
those same full side lengths from diag(F) at the *end* of a solver step
(cpGIMP). This evaluator reads the current lengths in either mode. Neither
mode is CPDI or a general free-boundary treatment.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from geopympm.core.particles import ParticleSet
from geopympm.core.grid import BackgroundGrid
from geopympm.kernels.interpolation_gimp import gimp2d_kernel


@dataclass(slots=True)
class GIMPInterpolation:
    """Single evaluation at current positions/domains, with 16 padded slots.

    Array shapes: cell_indices/local_coordinates (n,2), node_ids (n,16),
    weights (n,16), gradients (n,16,2). Zero-support slots have zero weight
    and gradient and a clamped, valid node ID.
    """
    cell_indices: np.ndarray
    local_coordinates: np.ndarray
    node_ids: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    domain_size: np.ndarray  # full sizes at evaluation, for stale-domain detection


def validate_gimp_domain(particles: ParticleSet, grid: BackgroundGrid, position=None,
                         domain_size=None) -> None:
    """Ensure current or candidate full domain sizes fit the 4x4 stencil and grid."""
    if not isinstance(particles, ParticleSet) or not isinstance(grid, BackgroundGrid):
        raise TypeError("GIMP requires ParticleSet and BackgroundGrid.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Stage 10 GIMP interpolation is 2D only.")
    x = particles.position if position is None else np.asarray(position)
    if x.shape != (particles.n_particles, 2) or not np.all(np.isfinite(x)):
        raise ValueError("GIMP particle positions must be finite (n,2).")
    lp = particles.domain_size if domain_size is None else np.asarray(domain_size)
    if lp.shape != x.shape or not np.all(np.isfinite(lp)) or np.any(lp <= 0.0):
        raise ValueError("GIMP requires initialized, finite, positive full domain_size (n,2).")
    # Prevent b-a from rounding to zero in the exact average of nodal hats.
    resolvable = (64.0*np.finfo(np.float64).eps *
                  np.maximum(grid.cell_size, np.abs(x-grid.origin)))
    if np.any(lp <= resolvable):
        raise ValueError("GIMP domain_size is too small to resolve relative to grid/position.")
    # The bounded 4x4 candidate stencil is sufficient for full widths <= h.
    if np.any(lp > grid.cell_size[None, :] * (1.0 + 1e-13)):
        raise ValueError("Stage 10 GIMP/cpGIMP requires domain_size <= grid cell_size along each axis (16-slot support limit).")
    lower = grid.origin
    upper = lower + grid.cell_size * np.asarray(grid.n_cells)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1, *grid.n_cells) * grid.cell_size
    if np.any((x - lp/2) < lower - tolerance) or np.any((x + lp/2) > upper + tolerance):
        raise ValueError("GIMP particle domain extends beyond the background grid; enlarge grid.")


def evaluate_gimp(particles: ParticleSet, grid: BackgroundGrid) -> GIMPInterpolation:
    """Compute 2D tensor-product uGIMP weights and exact spatial gradients."""
    validate_gimp_domain(particles, grid)
    uv = (particles.position - grid.origin) / grid.cell_size
    if not np.all(np.isfinite(uv)):
        raise ValueError("GIMP normalized coordinates are not finite.")
    n_cells = np.asarray(grid.n_cells, dtype=np.int64)
    bounded = np.clip(uv, 0.0, n_cells.astype(np.float64))
    cells = np.minimum(np.floor(bounded).astype(np.int64), n_cells - 1)
    local = np.ascontiguousarray(bounded - cells)
    cells = np.ascontiguousarray(cells)
    ids, weights, grad = gimp2d_kernel(
        cells, np.ascontiguousarray(uv, dtype=np.float64),
        np.ascontiguousarray(particles.domain_size),
        grid.n_cells[0], grid.n_cells[1],
        float(grid.cell_size[0]), float(grid.cell_size[1]),
    )
    # Avoid silently masking a numerical conditioning/stencil problem.
    if (not np.all(np.isfinite(weights)) or not np.all(np.isfinite(grad)) or
        np.any(weights < -1e-13) or
        not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2e-12) or
        not np.allclose(grad.sum(axis=1), 0.0, rtol=0.0,
                        atol=2e-11/float(np.min(grid.cell_size)))):
        raise ValueError("GIMP interpolation failed partition-of-unity/gradient consistency checks.")
    return GIMPInterpolation(cells, local, ids, weights, grad, particles.domain_size.copy())


def validate_current_gimp(particles: ParticleSet, grid: BackgroundGrid,
                          interpolation: GIMPInterpolation, *,
                          allow_updated_f: bool = False) -> None:
    """Fail fast on stale particle position/domain or inconsistent grid geometry.

    Re-evaluation is intentionally conservative and incurs computational
    overhead, matching the strict Stage 1-9 correctness-first philosophy.
    """
    from geopympm.interpolation.cpdi import CPDIInterpolation, validate_current_cpdi
    if isinstance(interpolation, CPDIInterpolation):
        validate_current_cpdi(particles, grid, interpolation,
                              allow_updated_f=allow_updated_f)
        return
    if not isinstance(interpolation, GIMPInterpolation):
        raise TypeError("Expected a GIMPInterpolation.")
    if not np.array_equal(interpolation.domain_size, particles.domain_size):
        raise ValueError("GIMP interpolation is stale: particle domain_size changed.")
    other = evaluate_gimp(particles, grid)
    if (not np.array_equal(interpolation.node_ids, other.node_ids) or
        not np.array_equal(interpolation.cell_indices, other.cell_indices) or
        not np.allclose(interpolation.local_coordinates, other.local_coordinates,
                        rtol=0.0, atol=1e-13) or
        not np.allclose(interpolation.weights, other.weights, rtol=1e-12, atol=1e-13) or
        not np.allclose(interpolation.gradients, other.gradients, rtol=1e-12, atol=1e-12)):
        raise ValueError("GIMP interpolation is stale or inconsistent with current particle domains/grid.")
