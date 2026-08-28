"""Solid-mechanics solution procedures.

The namespace stays importable on every supported platform. Stateful global
procedures are loaded only when requested, allowing the native Windows core to
report their PETSc capability boundary before importing solver internals.
"""

from importlib import import_module as _import_module


_LAZY_EXPORTS = {
    "CreepEnergyFrame": "creep",
    "CreepIncrementInfo": "creep",
    "CreepPathInfo": "creep",
    "ImplicitCreepStep": "creep",
    "implicit_creep_step": "creep",
    "ExperimentalFiniteStrainPlasticityStep": "finite_strain_plasticity",
    "FiniteStrainJ2AffineTransaction": "finite_strain_plasticity",
    "FiniteStrainPlasticityIncrementInfo": "finite_strain_plasticity",
    "experimental_finite_strain_j2_step": "finite_strain_plasticity",
    "finite_strain_j2_affine_problem": "finite_strain_plasticity",
    "J2IncrementInfo": "plasticity",
    "J2LoadPathInfo": "plasticity",
    "J2PlasticityStep": "plasticity",
    "j2_plasticity_step": "plasticity",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = tuple(_LAZY_EXPORTS)
