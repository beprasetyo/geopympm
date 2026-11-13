"""Material definitions and constitutive stress-update implementations."""

from .linear_elastic import LinearElastic, update_linear_elastic_stress
from .mohr_coulomb import MohrCoulomb, mohr_coulomb_yield, update_mohr_coulomb_stress, MC_HISTORY
from .objective_elastic import update_objective_linear_elastic_stress
from .material_state import MaterialState, StateFieldSpec
from .interface import ConstitutiveAdapter, resolve_constitutive_adapter
from .neo_hookean import (CompressibleNeoHookean, update_neo_hookean_stress,
                          evaluate_neo_hookean, neo_hookean_strain_energy)

__all__ = [
    "MohrCoulomb", "mohr_coulomb_yield", "update_mohr_coulomb_stress", "MC_HISTORY",
    "MaterialState", "StateFieldSpec", "ConstitutiveAdapter",
    "resolve_constitutive_adapter",
    "LinearElastic", "update_linear_elastic_stress",
    "update_objective_linear_elastic_stress", "CompressibleNeoHookean",
    "update_neo_hookean_stress", "evaluate_neo_hookean",
    "neo_hookean_strain_energy",
]
