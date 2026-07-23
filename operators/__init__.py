"""Engineering-level finite-element operators.

This package gives users K/M/C/F language while keeping weak forms available in
``agentfem.forms`` and low-level assembly in ``agentfem.assembly``.
"""

from . import elasticity
from .core import (
    OperatorForm,
    assemble_matrix,
    assemble_vector,
    body_force_vector,
    boundary_load_vector,
    boundary_force_vector,
    compile_form,
    damping_operator,
    diffusion_operator,
    force_vector,
    lumped_mass,
    lumped_operator,
    mass_operator,
)
from .elasticity import internal_force_vector, stiffness_operator
from .system import LinearSystem, SecondOrderSystem

assemble_lumped_mass = lumped_mass
assemble_lumped_operator = lumped_operator

__all__ = [
    "LinearSystem",
    "OperatorForm",
    "SecondOrderSystem",
    "assemble_lumped_mass",
    "assemble_lumped_operator",
    "assemble_matrix",
    "assemble_vector",
    "body_force_vector",
    "boundary_load_vector",
    "boundary_force_vector",
    "compile_form",
    "damping_operator",
    "diffusion_operator",
    "elasticity",
    "force_vector",
    "internal_force_vector",
    "lumped_mass",
    "lumped_operator",
    "mass_operator",
    "stiffness_operator",
]
