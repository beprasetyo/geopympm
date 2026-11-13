"""Integrated, fixed-step 2D explicit MPM with Stage 10 uGIMP/cpGIMP options.

Default: legacy updated-grid-velocity / USL-like cycle:

    interpolate at x_n -> P2G -> grid force/momentum update
    -> particle kinematics -> elastic stress -> G2P -> advance time.

The kinematic and stress calculations use updated nodal velocities, but
interpolation gradients are evaluated at the *old* particle positions. Stage 12 additionally provides explicit USF and MUSL options. The default
USL-like branch is unchanged for numerical backward compatibility. By default plane-strain stresses use the Stage-4 non-objective law for
regression compatibility. Stage-7 corotational stresses are available with
``stress_update="corotational"``. Stage-8 optionally uses exponential-map F integration, but
Stage-9 affine-exponential advection can be selected explicitly; neither
option is a validated large-deformation landslide/pile penetration model.

A fixed time step is checked against a *heuristic conservative* elastic
wave / particle-advection CFL estimate on every step. This is not a proof
of numerical stability. Errors restore particle and transient grid state
for that step at the cost of copying arrays (a correctness-first baseline).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.interpolation.mpm import evaluate_quad4
from geopympm.interpolation.gimp import evaluate_gimp, validate_gimp_domain
from geopympm.interpolation.cpgimp import candidate_cpgimp_domain, update_cpgimp_domain
from geopympm.interpolation.cpdi import (evaluate_cpdi, candidate_cpdi_geometry, update_cpdi_domain)
from geopympm.materials.linear_elastic import LinearElastic
from geopympm.materials.neo_hookean import CompressibleNeoHookean
from geopympm.materials.mohr_coulomb import MohrCoulomb
from geopympm.materials.registry import material_family
from geopympm.materials.material_state import MaterialState
from geopympm.materials.interface import resolve_constitutive_adapter
from .p2g import particle_to_grid
from .grid_update import update_grid_momentum
from .kinematics import update_particle_kinematics
from .g2p import grid_to_particle
from .momentum_remap import remap_updated_particle_momentum


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Small scalar snapshot at the *end* of one MPM time step (SI units)."""

    step: int
    time: float
    dt: float
    dt_limit: float
    total_mass: float
    kinetic_energy: float
    momentum: tuple[float, float]
    max_speed: float
    min_jacobian: float
    active_nodes: int


def _finite_scalar(name: str, value: object, *, strictly_positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not math.isfinite(result) or (strictly_positive and result <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and > 0." if strictly_positive else "."))
    return result


class ExplicitMPMSolver:
    """Combine Stage 2/3A/3B/3D/4/3C into an explicit MPM/uGIMP cycle.

    Parameters
    ----------
    particles, grid : ParticleSet, BackgroundGrid
        Existing 2D Stage-1 data containers. The solver mutates both in place.
    materials : LinearElastic or CompressibleNeoHookean, or homogeneous model-ID mapping
        Plane-strain material law(s); mixing the two model classes is not yet supported.
    dt : float
        Fixed positive timestep (s). Rechecked each step against CFL estimate.
    pic_fraction : float, default 1
        1=PIC, 0=FLIP, intermediate values blend particle velocities.
    cfl : float, default 0.25
        Dimensionless fraction of ``h_min / max_p(c_p + |v_p|)``; in (0,1].
        This is a practical screening estimate, not a stability guarantee.
    gravity : tuple[float, float], default (0,0)
        Constant acceleration (m/s^2).
    stress_update : {"incremental", "corotational", "hyperelastic"}, default "incremental"
        Stage-4 legacy or Stage-7 objective in-plane stress integration.
        Default retained to reproduce Stage-6A–6F results. For simulations
        involving rotation explicitly select "corotational" for LinearElastic.
        Select "hyperelastic" with CompressibleNeoHookean (uses updated F).
    kinematics_update : {"euler", "exponential"}, default "euler"
        Stage-3D Euler F integration or Stage-8 exponential-map update.
        The latter avoids pure-spin Jacobian drift; G2P position integration
        remains explicit Euler. Use corotational stress update for rotations.
    advection_update : {"euler", "affine_exponential"}, default "euler"
        Stage-3C Euler particle-position advection or Stage-9 local affine
        exponential trajectory update. The Stage-9 option uses velocity
        gradient from the same current grid update; velocity PIC/FLIP
        interpolation itself remains unchanged.
    material_state : MaterialState or None, default None
        Optional versioned, per-particle history arrays. Stage 13A does not
        update user-declared fields for elastic/CNH laws; future plastic laws
        must declare and explicitly integrate their state variables.
    scheme : {"usl", "usf", "musl"}, default "usl"
        Stage 12 update ordering. The default usl retains the entire legacy
        Stage-11 step path; musl uses an additional updated-momentum remap
        before calculating particle kinematics and stress.
    interpolation : {"mpm", "gimp", "cpgimp", "cpdi"}, default "mpm"
        Quad4 point MPM, unchanged-domain uGIMP, or classical 2D axis-aligned
        contiguous-particle GIMP with l_x=l_x0*F_xx and l_y=l_y0*F_yy.
        Both GIMP modes retain 16 padded node slots and require full domain
        lengths <= cell_size and full rectangles inside the grid. cpGIMP
        rejects unsupported stretches/rotations rather than clamping sizes.
        cpdi uses 2D first-order corner-based parallelograms convected by the
        full deformation gradient, with 16 corner contributions (duplicates
        allowed) and no fixed cell-size cap; corners must remain on the grid.
    nodal_external_force : ndarray or None
        Fixed additional (n_nodes,2) nodal forces in N. This is *not* an
        automatically integrated surface-traction boundary condition.

    Notes
    -----
    - One ``step()`` evaluates a fresh interpolation at particle positions x_n.
    - BC masks / values on ``grid`` persist across all steps.
    - Transient grid data are reassembled by P2G each step.
    - All particle motion must remain inside the existing background grid.
      In either GIMP mode, the entire particle interpolation domain must stay inside.
    - cpGIMP domains remain axis-aligned; CPDI1 handles affine shear/rotation,
      but does not model curved or fragmented material domains.
    - No damping, dedicated contact, plasticity, dynamic grid extension, or FEM transfer.
    - CNH uses plane strain and the elastic/isothermal Uintah UCNH law. Its
      CFL estimate uses reference/small-strain wave moduli, not nonlinear
      tangent bounds; reduce dt and check convergence under large distortion.
    - The default Stage-4 stress law is not objective under large rigid rotations;
    select stress_update="corotational" for Stage-7 objective Cauchy stress.
    - A snapshot of state is made per step for rollback when any substage fails;
      this incurs memory copies and should be profiled before large benchmarks.
    """

    _PARTICLE_MUTABLE = (
        "position", "velocity", "volume", "strain", "stress", "deformation_gradient",
        "domain_size",
    )
    _GRID_MUTABLE = (
        "mass", "momentum", "velocity", "acceleration", "force_internal", "force_external",
    )

    def __init__(
        self,
        particles: ParticleSet,
        grid: BackgroundGrid,
        materials: LinearElastic | CompressibleNeoHookean | MohrCoulomb | Mapping,
        *,
        dt: float,
        pic_fraction: float = 1.0,
        cfl: float = 0.25,
        gravity=(0.0, 0.0),
        nodal_external_force=None,
        stress_update: str = "incremental",
        kinematics_update: str = "euler",
        advection_update: str = "euler",
        interpolation: str = "mpm",
        scheme: str = "usl",
        material_state: MaterialState | None = None,
    ) -> None:
        if not isinstance(particles, ParticleSet) or not isinstance(grid, BackgroundGrid):
            raise TypeError("particles must be ParticleSet and grid must be BackgroundGrid.")
        if particles.ndim != 2 or grid.ndim != 2:
            raise ValueError("Stage 5 supports structured Quad4 MPM in 2D only.")
        self.particles = particles
        self.grid = grid
        self.dt = _finite_scalar("dt", dt, strictly_positive=True)
        self.pic_fraction = _finite_scalar("pic_fraction", pic_fraction)
        if not 0.0 <= self.pic_fraction <= 1.0:
            raise ValueError("pic_fraction must be between 0 and 1.")
        self.cfl = _finite_scalar("cfl", cfl, strictly_positive=True)
        if self.cfl > 1.0:
            raise ValueError("cfl must not exceed 1.0.")
        g = np.asarray(gravity, dtype=np.float64)
        if g.shape != (2,) or not np.all(np.isfinite(g)):
            raise ValueError("gravity must be a finite vector of shape (2,).")
        self.gravity = g.copy()

        # Validate model family and all active IDs before time stepping.
        family = material_family(particles, materials)
        self.materials = materials
        self._constitutive_adapter = resolve_constitutive_adapter(particles, materials)
        if material_state is None:
            material_state = MaterialState(
                particles.n_particles, list(self._constitutive_adapter.history_schema)
            )
        if not isinstance(material_state, MaterialState):
            raise TypeError("material_state must be MaterialState or None")
        material_state.validate(particles.n_particles)
        self._constitutive_adapter.validate_state(material_state)
        self.material_state = material_state
        if not isinstance(stress_update, str) or stress_update not in (
            "incremental", "corotational", "hyperelastic"
        ):
            raise ValueError("stress_update must be incremental, corotational, or hyperelastic")
        if (family == "cnh") != (stress_update == "hyperelastic"):
            raise ValueError("CNH requires stress_update='hyperelastic'; LinearElastic/MC rejects it")
        if stress_update not in self._constitutive_adapter.modes:
            raise ValueError(f"{family} does not support stress_update={stress_update!r}")
        self.stress_update = stress_update
        if not isinstance(kinematics_update, str) or kinematics_update not in ("euler", "exponential"):
            raise ValueError("kinematics_update must be 'euler' or 'exponential'.")
        self.kinematics_update = kinematics_update
        if not isinstance(advection_update, str) or advection_update not in (
            "euler", "affine_exponential"
        ):
            raise ValueError("advection_update must be 'euler' or 'affine_exponential'.")
        self.advection_update = advection_update
        if not isinstance(interpolation, str) or interpolation not in ("mpm", "gimp", "cpgimp", "cpdi"):
            raise ValueError("interpolation must be 'mpm', 'gimp', 'cpgimp', or 'cpdi'.")
        self.interpolation = interpolation
        if not isinstance(scheme, str) or scheme not in ("usl", "usf", "musl"):
            raise ValueError("scheme must be 'usl', 'usf', or 'musl'")
        self.scheme = scheme
        if interpolation in ("gimp", "cpgimp"):
            validate_gimp_domain(particles, grid)
        if interpolation == "cpgimp":
            # The initial state must already use lengths consistent with its
            # cumulative F. Usually F=I and domain_size==domain_size0.
            expected_domain = candidate_cpgimp_domain(particles, grid)
            if not np.allclose(particles.domain_size, expected_domain,
                               rtol=1e-12, atol=0.0):
                raise ValueError("cpGIMP initial domain_size must equal "
                                 "domain_size0*diag(F); initialize a consistent state.")

        if interpolation == "cpdi":
            _, _, _, widths = candidate_cpdi_geometry(particles, grid)
            if not np.allclose(particles.domain_size, widths, rtol=1e-12, atol=1e-14):
                raise ValueError("CPDI initial domain_size must equal the axis-aligned envelope of F*domain_size0.")

        if nodal_external_force is None:
            self.nodal_external_force = None
        else:
            f = np.asarray(nodal_external_force, dtype=np.float64)
            if f.shape != (grid.n_nodes, 2) or not np.all(np.isfinite(f)):
                raise ValueError("nodal_external_force must be finite with shape (n_nodes,2).")
            self.nodal_external_force = np.array(f, order="C", copy=True)

        self.time = 0.0
        self.step_count = 0

    def stable_dt_estimate(self) -> float:
        """Return an indicative elastic wave + particle-advection time-step cap.

        c_p=sqrt((lambda+2G)/rho) in m/s, rho=m/V. For every particle,
        use characteristic cell size h=min(dx,dy) and wave/advection speed
        ``c_p + sqrt(vx**2+vy**2)``. Nonlinear/large-deformation stability
        may require smaller steps, which this estimator cannot certify.
        """
        p = self.particles
        if (p.mass.shape != (p.n_particles,) or p.volume.shape != (p.n_particles,)
                or np.any(p.mass <= 0) or np.any(p.volume <= 0)
                or not np.all(np.isfinite(p.mass)) or not np.all(np.isfinite(p.volume))):
            raise ValueError("Particle masses/volumes must remain finite and positive.")
        if p.velocity.shape != (p.n_particles, 2) or not np.all(np.isfinite(p.velocity)):
            raise ValueError("Particle velocities must remain finite (n_particles,2).")
        lam, shear = self._constitutive_adapter.wave_moduli(p, self.materials)
        rho = p.mass / p.volume
        wave2 = (lam + 2.0 * shear) / rho
        if np.any(wave2 <= 0) or not np.all(np.isfinite(wave2)):
            raise ValueError("Cannot estimate stable dt from material density and moduli.")
        speed = np.sqrt(wave2) + np.linalg.norm(p.velocity, axis=1)
        max_speed = float(np.max(speed))
        h = float(np.min(self.grid.cell_size))
        limit = self.cfl * h / max_speed
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("Computed dt estimate is invalid.")
        return limit

    def _update_material_stress(self, kinematics) -> None:
        """Dispatch an in-place constitutive update via the Stage 13A contract."""
        self._constitutive_adapter.stress_update(
            self.particles, kinematics, self.materials, self.dt,
            self.stress_update, self.material_state,
        )

    def step(self) -> StepDiagnostics:
        """Advance one fixed time step, with rollback if any operation fails."""
        dt_limit = self.stable_dt_estimate()
        if self.dt > dt_limit * (1.0 + 1e-12):
            raise ValueError(
                f"dt={self.dt:.6g} exceeds current Stage-5 CFL estimate "
                f"{dt_limit:.6g}; reduce dt, grid size, or revisit material data."
            )
        self.material_state.validate(self.particles.n_particles)
        self._constitutive_adapter.validate_state(self.material_state)
        # Keep old state available if a late stage fails (e.g. particle exits grid).
        p_old = {name: getattr(self.particles, name).copy() for name in self._PARTICLE_MUTABLE}
        g_old = {name: getattr(self.grid, name).copy() for name in self._GRID_MUTABLE}
        h_old = self.material_state.snapshot()
        try:
            interpolation = (evaluate_cpdi(self.particles, self.grid)
                             if self.interpolation == "cpdi"
                             else evaluate_gimp(self.particles, self.grid)
                             if self.interpolation in ("gimp", "cpgimp")
                             else evaluate_quad4(self.particles, self.grid))
            particle_to_grid(self.particles, self.grid, interpolation)
            if self.scheme == "usf":
                # USF: calculate F/stress using projected pre-force nodal
                # velocity, then assemble forces using the UPDATED stress.
                # Enforce prescribed nodal velocities before taking L.
                bc = self.grid.velocity_bc_mask
                self.grid.velocity[bc] = self.grid.velocity_bc_value[bc]
                kinematics = update_particle_kinematics(
                    self.particles, self.grid, interpolation, self.dt,
                    kinematics_update=self.kinematics_update,
                )
                self._update_material_stress(kinematics)
                update_grid_momentum(
                    self.particles, self.grid, interpolation, self.dt,
                    gravity=self.gravity, nodal_external_force=self.nodal_external_force,
                    allow_updated_deformation_gradient=(self.interpolation == "cpdi"),
                )
                # Affine advection needs the POST-force nodal velocity gradient,
                # not the pre-force gradient used in the USF stress update.
                advection_L = (np.ascontiguousarray(np.einsum(
                    "pai,paj->pij", self.grid.velocity[interpolation.node_ids],
                    interpolation.gradients)) if self.advection_update == "affine_exponential"
                    else None)
                grid_to_particle(
                    self.particles, self.grid, interpolation, self.dt,
                    pic_fraction=self.pic_fraction,
                    advection_update=self.advection_update,
                    velocity_gradient=advection_L,
                    allow_updated_deformation_gradient=(self.interpolation == "cpdi"),
                )
            else:
                # Common P2G -> grid update for USL and MUSL.
                update_grid_momentum(
                    self.particles, self.grid, interpolation, self.dt,
                    gravity=self.gravity, nodal_external_force=self.nodal_external_force,
                )
                if self.scheme == "musl":
                    # Transfer particle velocities at x_n, but do not yet
                    # advance positions (which would invalidate interpolation).
                    grid_to_particle(
                        self.particles, self.grid, interpolation, self.dt,
                        pic_fraction=self.pic_fraction, transfer_mode="velocity",
                    )
                    # Second P2G: updated particle MOMENTUM, unchanged mass,
                    # same x_n interpolation. Prescribed grid BCs are reapplied.
                    remap_updated_particle_momentum(self.particles, self.grid, interpolation)
                kinematics = update_particle_kinematics(
                    self.particles, self.grid, interpolation, self.dt,
                    kinematics_update=self.kinematics_update,
                )
                self._update_material_stress(kinematics)
                grid_to_particle(
                    self.particles, self.grid, interpolation, self.dt,
                    pic_fraction=self.pic_fraction,
                    advection_update=self.advection_update,
                    velocity_gradient=(kinematics.velocity_gradient
                                       if self.advection_update == "affine_exponential" else None),
                    allow_updated_deformation_gradient=(self.interpolation == "cpdi"),
                    transfer_mode=("position" if self.scheme == "musl" else "both"),
                )
            if self.interpolation == "cpgimp":
                # AFTER G2P: every substep above consumes interpolation
                # evaluated at (x_n, domain_n). Domain_(n+1) enters the
                # NEXT step. This validation may raise; solver rollback
                # restores domain_size, F, volume, position and grid state.
                update_cpgimp_domain(self.particles, self.grid)
            if self.interpolation == "cpdi":
                # Post-advection F and position define next-step CPDI corners.
                # Full rollback remains active if corners exit the fixed grid.
                update_cpdi_domain(self.particles, self.grid)
            # Constitutive arrays must remain valid before the step is committed.
            self.material_state.validate(self.particles.n_particles)
            # End-of-step diagnostics (before incrementing clock).
            p = self.particles
            speeds2 = np.sum(p.velocity ** 2, axis=1)
            energy = float(0.5 * np.dot(p.mass, speeds2))
            momenta = np.sum(p.mass[:, None] * p.velocity, axis=0)
            result = StepDiagnostics(
                step=self.step_count + 1,
                time=self.time + self.dt,
                dt=self.dt,
                dt_limit=dt_limit,
                total_mass=float(np.sum(p.mass)),
                kinetic_energy=energy,
                momentum=(float(momenta[0]), float(momenta[1])),
                max_speed=float(np.sqrt(np.max(speeds2))),
                min_jacobian=float(np.min(kinematics.jacobian)),
                active_nodes=int(np.count_nonzero(self.grid.mass > 0.0)),
            )
            if not all(math.isfinite(value) for value in (
                result.time, result.total_mass, result.kinetic_energy,
                *result.momentum, result.max_speed, result.min_jacobian,
            )):
                raise ValueError("Nonfinite Stage-5 diagnostics; state not committed.")
        except Exception:
            # Preserve existing ndarray identities that downstream routines may hold.
            for name, values in p_old.items():
                getattr(self.particles, name)[:] = values
            for name, values in g_old.items():
                getattr(self.grid, name)[:] = values
            self.material_state.restore(h_old)
            raise
        self.time = result.time
        self.step_count = result.step
        return result

    def run(self, n_steps: int) -> list[StepDiagnostics]:
        """Run exactly n_steps, returning one scalar diagnostic per step.

        If a later step raises, completed earlier steps remain committed and
        the failing step is rolled back. This function is not all-or-nothing.
        """
        if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)):
            raise TypeError("n_steps must be a nonnegative integer.")
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative.")
        return [self.step() for _ in range(int(n_steps))]


__all__ = ["ExplicitMPMSolver", "StepDiagnostics"]
