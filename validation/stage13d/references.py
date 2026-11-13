"""Closed-form Mohr–Coulomb reference solutions; independent of GeoPyMPM kernels.

Units: SI, stress compression negative (tension-positive convention).
'Pure shear' reference is the perfectly plastic phi=psi=0 (Tresca) limit.
These equations are derived independently, NOT from a call to the MPM solver.
"""
from __future__ import annotations

import math


def _validate_moduli(young_modulus: float, poisson_ratio: float) -> None:
    if not (math.isfinite(young_modulus) and young_modulus > 0.0):
        raise ValueError("E must be finite and positive")
    if not (math.isfinite(poisson_ratio) and -1.0 < poisson_ratio < 0.5):
        raise ValueError("Poisson's ratio must be finite and in (-1, 0.5)")


def shear_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Isotropic elastic shear modulus G=E/(2(1+nu))."""
    _validate_moduli(young_modulus, poisson_ratio)
    return young_modulus / (2.0 * (1.0 + poisson_ratio))


def tresca_pure_shear(gamma: float, young_modulus: float,
                      poisson_ratio: float, cohesion: float) -> dict[str, float]:
    """Exact small-strain simple-shear response for MC with phi=psi=0.

    gamma is engineering shear strain. Because no volumetric plastic flow is
    generated, the solution is tau=sign(gamma)*min(G*abs(gamma),c),
    gamma_plastic=gamma-tau/G and eq_plastic=abs(gamma_plastic)/sqrt(3).
    Applicable to *monotonic* shear from virgin stress/history only.
    """
    g = shear_modulus(young_modulus, poisson_ratio)
    if not (math.isfinite(gamma) and math.isfinite(cohesion) and cohesion > 0.0):
        raise ValueError("gamma and positive cohesion must be finite")
    tau = math.copysign(min(g * abs(gamma), cohesion), gamma)
    plastic = gamma - tau / g
    return {"gamma": float(gamma), "tau_pa": tau, "gamma_plastic": plastic,
            "equivalent_plastic_strain": abs(plastic) / math.sqrt(3.0),
            "yield_gamma": cohesion / g}


def triaxial_compression_strength(confinement_pa: float, cohesion_pa: float,
                                   friction_angle_deg: float) -> float:
    """Mohr–Coulomb axisymmetric compressive deviator strength q=σ1c−σ3c.

    q=2(c cos(phi)+sigma3_compression sin(phi))/(1-sin(phi)); compression
    magnitudes are POSITIVE inputs. The returned q is positive.  This is a
    STRESS-SPACE yield reference, not a claim to simulate true triaxial
    loading using the 2D plane-strain solver.
    """
    if not (math.isfinite(confinement_pa) and confinement_pa >= 0
            and math.isfinite(cohesion_pa) and cohesion_pa >= 0
            and math.isfinite(friction_angle_deg)
            and 0.0 <= friction_angle_deg < 90.0):
        raise ValueError("invalid triaxial reference parameters")
    angle = math.radians(friction_angle_deg)
    return 2.0 * (cohesion_pa * math.cos(angle)
                  + confinement_pa * math.sin(angle)) / (1.0 - math.sin(angle))


def plane_strain_confined_yield_strain(young_modulus: float,
                                       poisson_ratio: float,
                                       cohesion_pa: float,
                                       friction_angle_deg: float) -> float:
    """First yield for eps_xx=-e, eps_yy=eps_zz=0, virgin isotropic stress.

    σxx=-(λ+2G)e and σyy=σzz=-λe until yield; therefore
    f=2e[G-(λ+G)sin(phi)]-2c*cos(phi).
    Only defined when the denominator is positive, else the specified
    proportional elastic loading path does not reach yielding for e>=0.
    """
    g = shear_modulus(young_modulus, poisson_ratio)
    if not (math.isfinite(cohesion_pa) and cohesion_pa > 0.0
            and math.isfinite(friction_angle_deg)
            and 0.0 <= friction_angle_deg < 90.0):
        raise ValueError("invalid confined compression reference parameters")
    lam = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    a = math.radians(friction_angle_deg)
    denom = g - (lam + g) * math.sin(a)
    if denom <= 0.0:
        raise ValueError("no yield onset on this elastic confined-compression path")
    return cohesion_pa * math.cos(a) / denom
