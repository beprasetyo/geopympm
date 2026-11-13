"""Validated Stage-11 material model dispatch, homogeneous constitutive family.

Material-ID mappings may contain multiple parameter sets, but this stage
explicitly rejects mixing LinearElastic and CompressibleNeoHookean in one solver.
Mixed-model particle dispatch is left for a later stage.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .linear_elastic import LinearElastic, _particle_moduli
from .neo_hookean import CompressibleNeoHookean, _particle_cnh_moduli
from .mohr_coulomb import MohrCoulomb, _particle_mc_parameters


def material_family(particles, materials) -> str:
    if type(materials) is LinearElastic:
        _particle_moduli(particles, materials)
        return "linear"
    if type(materials) is CompressibleNeoHookean:
        _particle_cnh_moduli(particles, materials)
        return "cnh"
    if type(materials) is MohrCoulomb:
        _particle_mc_parameters(particles, materials)
        return "mc"
    if not isinstance(materials, Mapping) or not materials:
        raise TypeError("materials must be a supported model or nonempty ID mapping")
    types = {type(m) for m in materials.values()}
    if types == {MohrCoulomb}:
        _particle_mc_parameters(particles, materials)
        return "mc"
    if types == {LinearElastic}:
        _particle_moduli(particles, materials)
        return "linear"
    if types == {CompressibleNeoHookean}:
        _particle_cnh_moduli(particles, materials)
        return "cnh"
    raise TypeError("Stage 11 supports homogeneous LinearElastic or CNH material families; no mixed families")


def particle_wave_moduli(particles, materials):
    """Return elastic Lamé lambda and shear for conservative small-strain CFL screening."""
    if material_family(particles, materials) == "linear":
        return _particle_moduli(particles, materials)
    if material_family(particles, materials) == "mc":
        lam, mu, *_ = _particle_mc_parameters(particles, materials)
        return lam, mu
    K, G = _particle_cnh_moduli(particles, materials)
    return K - (2.0 / 3.0)*G, G
