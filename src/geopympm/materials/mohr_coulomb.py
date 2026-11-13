"""Stage 13C: plane-strain small-increment Mohr–Coulomb multisurface return.

Tension-positive Cauchy stress [xx,yy,zz,xy,yz,xz] (Pa), engineering
shear strain; perfectly plastic; non-associated flow is supported.
The 3D principal stresses include sigma_zz even though epsilon_zz = 0.

All six Mohr–Coulomb planes are enforced in principal-stress space.
Active sets of size 1 (face), 2 (edge) and up to 3 (apex) are
examined with the non-associated plastic potential, if admissible.
An inadmissible non-associated apex (notably psi=0 in hydrostatic
excess tension) raises ValueError and commits no state; it is NEVER
silently projected by inventing a different flow rule.

Plane-strain eigenvectors are computed analytically in the 2x2 in-plane
block; no scipy/LAPACK requirement for this Numba kernel. The model
is small-increment perfectly plastic MC, NOT finite-strain objective MC.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from .material_state import StateFieldSpec, MaterialState
from .linear_elastic import LinearElastic

MC_HISTORY = (
    StateFieldSpec("plastic_strain", (6,)),
    StateFieldSpec("equivalent_plastic_strain"),
)


@dataclass(frozen=True, slots=True)
class MohrCoulomb:
    young_modulus: float
    poisson_ratio: float
    cohesion: float
    friction_angle_deg: float
    dilation_angle_deg: float = 0.0

    def __post_init__(self):
        for name in ("young_modulus", "poisson_ratio", "cohesion",
                     "friction_angle_deg", "dilation_angle_deg"):
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite real number")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite real number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.cohesion < 0.0:
            raise ValueError("cohesion must be >= 0")
        if not 0.0 <= self.friction_angle_deg < 90.0:
            raise ValueError("friction_angle_deg must be in [0, 90)")
        if not 0.0 <= self.dilation_angle_deg <= self.friction_angle_deg:
            raise ValueError("dilation_angle_deg must be in [0, friction_angle_deg]")
        LinearElastic(self.young_modulus, self.poisson_ratio)

    @property
    def lame_lambda(self):
        return LinearElastic(self.young_modulus, self.poisson_ratio).lame_lambda

    @property
    def shear_modulus(self):
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))


def _particle_mc_parameters(particles, materials):
    """Return contiguous, validated per-particle Lamé and Mohr–Coulomb values."""
    n = particles.n_particles
    if type(materials) is MohrCoulomb:
        models = [materials] * n
    elif isinstance(materials, Mapping) and materials:
        if any(type(mid) not in (int, np.int32, np.int64) or mid < 0
               or mid > np.iinfo(np.int32).max or type(mat) is not MohrCoulomb
               for mid, mat in materials.items()):
            raise TypeError("MC mapping requires nonnegative int32 IDs and MohrCoulomb values")
        if any(int(mid) not in materials for mid in particles.material_id):
            raise ValueError("Missing MohrCoulomb material ID")
        models = [materials[int(mid)] for mid in particles.material_id]
    else:
        raise TypeError("materials must be MohrCoulomb or homogeneous MC mapping")
    return tuple(np.ascontiguousarray(np.array(vals, dtype=np.float64)) for vals in (
        [mat.lame_lambda for mat in models],
        [mat.shear_modulus for mat in models],
        [mat.cohesion for mat in models],
        [math.sin(math.radians(mat.friction_angle_deg)) for mat in models],
        [math.cos(math.radians(mat.friction_angle_deg)) for mat in models],
        [math.sin(math.radians(mat.dilation_angle_deg)) for mat in models],
    ))


@njit(cache=True)
def _mc_active_solution(qtrial, ftrial, nf, ng, h, indices, nactive,
                        tolerance):
    """Try one active set; return (success, principal stress, plastic increment).

    f = nf @ principal_stress - 2*c*cos_phi. For an active set A,
    (nf_A @ C @ ng_A) * gamma = ftrial_A; gamma >= 0,
    all six f <= 0. Enumerating at most three independent planes is
    sufficient in 3D principal-stress space; redundant apex planes
    do not require independent nonzero multipliers.
    """
    augmented = np.zeros((3, 4), dtype=np.float64)
    for i in range(nactive):
        ai = indices[i]
        for j in range(nactive):
            aj = indices[j]
            w = 0.0
            for k in range(3):
                w += nf[ai, k] * h[aj, k]
            augmented[i, j] = w
        augmented[i, 3] = ftrial[ai]

    # Small pivoted Gaussian elimination; avoids np.linalg.solve/LAPACK.
    max_entry = 0.0
    for i in range(nactive):
        for j in range(nactive):
            max_entry = max(max_entry, abs(augmented[i, j]))
    if max_entry == 0.0 or not np.isfinite(max_entry):
        return False, qtrial.copy(), np.zeros(3, dtype=np.float64)
    for k in range(nactive):
        pivot = k
        for i in range(k + 1, nactive):
            if abs(augmented[i, k]) > abs(augmented[pivot, k]):
                pivot = i
        if abs(augmented[pivot, k]) < 1.0e-12 * max_entry:
            return False, qtrial.copy(), np.zeros(3, dtype=np.float64)
        if pivot != k:
            for j in range(k, nactive):
                tmp = augmented[pivot, j]
                augmented[pivot, j] = augmented[k, j]
                augmented[k, j] = tmp
            tmp = augmented[pivot, 3]
            augmented[pivot, 3] = augmented[k, 3]
            augmented[k, 3] = tmp
        for i in range(k + 1, nactive):
            factor = augmented[i, k] / augmented[k, k]
            for j in range(k + 1, nactive):
                augmented[i, j] -= factor * augmented[k, j]
            augmented[i, k] = 0.0
            augmented[i, 3] -= factor * augmented[k, 3]
    gamma = np.zeros(3, dtype=np.float64)
    for i in range(nactive - 1, -1, -1):
        rhs = augmented[i, 3]
        for j in range(i + 1, nactive):
            rhs -= augmented[i, j] * gamma[j]
        gamma[i] = rhs / augmented[i, i]
    if not np.all(np.isfinite(gamma)):
        return False, qtrial.copy(), np.zeros(3, dtype=np.float64)
    gamma_scale = max(1.0e-15, np.max(np.abs(gamma[:nactive])))
    for i in range(nactive):
        if gamma[i] < -1.0e-10 * gamma_scale:
            return False, qtrial.copy(), np.zeros(3, dtype=np.float64)

    corrected = qtrial.copy()
    epsp_inc = np.zeros(3, dtype=np.float64)
    for i in range(nactive):
        ai = indices[i]
        for k in range(3):
            corrected[k] -= gamma[i] * h[ai, k]
            epsp_inc[k] += gamma[i] * ng[ai, k]

    # Check *all* yield faces (not just the active face). Evaluate the
    # corrected residual via its linearization (exact for each MC plane).
    for a in range(6):
        f = ftrial[a]
        for k in range(3):
            f += nf[a, k] * (corrected[k] - qtrial[k])
        if f > tolerance or not np.isfinite(f):
            return False, qtrial.copy(), np.zeros(3, dtype=np.float64)
    for i in range(nactive):
        a = indices[i]
        f = ftrial[a]
        for k in range(3):
            f += nf[a, k] * (corrected[k] - qtrial[k])
        if abs(f) > tolerance:
            return False, qtrial.copy(), np.zeros(3, dtype=np.float64)
    return True, corrected, epsp_inc


@njit(cache=True)
def _mc_multisurface_kernel(stress, delta, lam, mu, cohesion, sin_phi,
                            cos_phi, sin_psi, plastic_strain, eq_plastic):
    n = stress.shape[0]
    new_stress = stress.copy()
    new_epsp = plastic_strain.copy()
    new_eqp = eq_plastic.copy()
    status = np.zeros(n, dtype=np.int32)
    # branch: 0=elastic, 1=face, 2=edge, 3=apex; -1=inadmissible.
    branches = np.zeros(n, dtype=np.int32)
    for p in range(n):
        trace = delta[p, 0] + delta[p, 1]
        xx = stress[p, 0] + lam[p] * trace + 2.0 * mu[p] * delta[p, 0]
        yy = stress[p, 1] + lam[p] * trace + 2.0 * mu[p] * delta[p, 1]
        zz = stress[p, 2] + lam[p] * trace
        xy = stress[p, 3] + mu[p] * delta[p, 3]

        # Analytic spectral decomposition of the 2x2 (xx,xy;xy,yy)
        # tensor. Principal ordering is arbitrary because all six MC
        # planes are tested; this also handles crossing of sigma_zz.
        theta = 0.5 * math.atan2(2.0 * xy, xx - yy)
        ct, st = math.cos(theta), math.sin(theta)
        center = 0.5 * (xx + yy)
        radial = math.hypot(0.5 * (xx - yy), xy)
        qtrial = np.array((center - radial, center + radial, zz))
        nf = np.zeros((6, 3), dtype=np.float64)
        ng = np.zeros((6, 3), dtype=np.float64)
        h = np.zeros((6, 3), dtype=np.float64)
        ftrial = np.zeros(6, dtype=np.float64)
        face = 0
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                nf[face, i] = 1.0 + sin_phi[p]
                nf[face, j] = sin_phi[p] - 1.0
                ng[face, i] = 1.0 + sin_psi[p]
                ng[face, j] = sin_psi[p] - 1.0
                for k in range(3):
                    ftrial[face] += nf[face, k] * qtrial[k]
                    h[face, k] = 2.0 * mu[p] * ng[face, k] + 2.0 * lam[p] * sin_psi[p]
                ftrial[face] -= 2.0 * cohesion[p] * cos_phi[p]
                face += 1
        scale = max(1.0, abs(qtrial[0]), abs(qtrial[1]), abs(qtrial[2]), cohesion[p])
        tol = 5.0e-10 * scale
        if np.max(ftrial) <= 1.0e-11 * scale:
            new_stress[p, 0] = xx
            new_stress[p, 1] = yy
            new_stress[p, 2] = zz
            new_stress[p, 3] = xy
            continue

        solved = False
        corrected = qtrial.copy()
        plastic_inc = np.zeros(3, dtype=np.float64)
        active = np.zeros(3, dtype=np.int32)
        for count in range(1, 4):
            if solved:
                break
            for a in range(6):
                if solved:
                    break
                active[0] = a
                bstart = a + 1 if count > 1 else 6
                bend = 6 if count > 1 else 7
                for b in range(bstart, bend):
                    if solved:
                        break
                    if count > 1:
                        active[1] = b
                    cstart = b + 1 if count > 2 else 6
                    cend = 6 if count > 2 else 7
                    for c in range(cstart, cend):
                        if count > 2:
                            active[2] = c
                        ok, q, ep = _mc_active_solution(
                            qtrial, ftrial, nf, ng, h, active, count, tol)
                        if ok:
                            corrected = q
                            plastic_inc = ep
                            branches[p] = count
                            solved = True
                            break
        if not solved:
            # No valid nonnegative-multiplier multisurface return. A
            # hydrostatic excess tension with psi=0 is a common example:
            # isochoric plastic flow cannot relax hydrostatic stress.
            status[p] = 1
            branches[p] = -1
            continue

        # Reconstruct in the original fixed axes. +/- in-plane principal
        # directions are respectively (-sin(theta),cos(theta)) and
        # (cos(theta),sin(theta)). Engineering gamma_xy uses factor 2.
        new_stress[p, 0] = corrected[0] * st * st + corrected[1] * ct * ct
        new_stress[p, 1] = corrected[0] * ct * ct + corrected[1] * st * st
        new_stress[p, 2] = corrected[2]
        new_stress[p, 3] = (corrected[1] - corrected[0]) * ct * st
        new_epsp[p, 0] += plastic_inc[0] * st * st + plastic_inc[1] * ct * ct
        new_epsp[p, 1] += plastic_inc[0] * ct * ct + plastic_inc[1] * st * st
        new_epsp[p, 2] += plastic_inc[2]
        new_epsp[p, 3] += 2.0 * (plastic_inc[1] - plastic_inc[0]) * ct * st
        mean = (plastic_inc[0] + plastic_inc[1] + plastic_inc[2]) / 3.0
        d2 = 0.0
        for k in range(3):
            d2 += (plastic_inc[k] - mean)**2
        new_eqp[p] += math.sqrt(2.0 * d2 / 3.0)
    return new_stress, new_epsp, new_eqp, status, branches


def mohr_coulomb_yield(stress, material: MohrCoulomb):
    """Evaluate 3D MC yield f<=0, tension-positive, at one plane-strain stress."""
    value = np.asarray(stress, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("Stress must be a finite 6-component Voigt vector")
    if not isinstance(material, MohrCoulomb):
        raise TypeError("material must be MohrCoulomb")
    if value[4] != 0 or value[5] != 0:
        raise ValueError("Plane-strain shear yz/xz must vanish")
    tensor = np.array([[value[0], value[3], 0.0],
                       [value[3], value[1], 0.0],
                       [0.0, 0.0, value[2]]])
    e = np.linalg.eigvalsh(tensor)
    phi = math.radians(material.friction_angle_deg)
    return float((e[2] - e[0]) + (e[2] + e[0]) * math.sin(phi)
                 - 2 * material.cohesion * math.cos(phi))


def update_mohr_coulomb_stress(particles, strain_increment, materials, state):
    """Atomic multisurface MC update with admissible face/edge/apex returns.

    No commit if any particle has an incompatible plastic-flow return,
    invalid input, or nonfinite result; solver also rolls back kinematics.
    """
    from geopympm.core.particles import ParticleSet
    if not isinstance(particles, ParticleSet) or particles.ndim != 2:
        raise TypeError("MohrCoulomb requires a 2D plane-strain ParticleSet")
    n = particles.n_particles
    strain_increment = np.asarray(strain_increment, dtype=np.float64)
    if strain_increment.shape != (n, 6) or not np.all(np.isfinite(strain_increment)):
        raise ValueError("Invalid strain increment")
    if np.any(strain_increment[:, [2, 4, 5]] != 0.0):
        raise ValueError("Plane strain requires zero zz/yz/xz increments")
    if not np.all(np.isfinite(particles.stress)) or np.any(particles.stress[:, [4, 5]] != 0.0):
        raise ValueError("Invalid plane-strain initial stress")
    if not isinstance(state, MaterialState):
        raise TypeError("MohrCoulomb needs MaterialState")
    state.validate(n)
    if any(name not in state.names for name in ("plastic_strain", "equivalent_plastic_strain")):
        raise ValueError("Missing MohrCoulomb plastic history")
    lam, mu, cohesion, sin_phi, cos_phi, sin_psi = _particle_mc_parameters(particles, materials)
    sigma, epsp, eqp, errors, _branches = _mc_multisurface_kernel(
        np.ascontiguousarray(particles.stress), np.ascontiguousarray(strain_increment),
        lam, mu, cohesion, sin_phi, cos_phi, sin_psi,
        np.ascontiguousarray(state["plastic_strain"]),
        np.ascontiguousarray(state["equivalent_plastic_strain"]),
    )
    if np.any(errors):
        i = int(np.flatnonzero(errors)[0])
        raise ValueError(f"MohrCoulomb particle {i}: " + (
            "edge/apex has no admissible multisurface return for the chosen dilation angle" if errors[i] == 1
            else "invalid plastic multiplier denominator"))
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(epsp))
            and np.all(np.isfinite(eqp)) and np.all(eqp >= -1e-14)):
        raise ValueError("Nonfinite/invalid MohrCoulomb update")
    particles.stress[:] = sigma
    state["plastic_strain"][:] = epsp
    state["equivalent_plastic_strain"][:] = eqp


__all__ = ["MohrCoulomb", "MC_HISTORY", "mohr_coulomb_yield", "update_mohr_coulomb_stress"]
