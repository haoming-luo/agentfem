"""Engineering-level finite-element operators.

This package gives users K/M/C/F language while keeping weak forms available in
``agentfem.forms`` and low-level assembly in ``agentfem.assembly``.
"""

from . import elasticity
from .core import (
    OperatorForm,
    action,
    assemble_matrix,
    assemble_vector,
    bilinear_form,
    body_force_vector,
    boundary_load_vector,
    boundary_force_vector,
    boundary_model_vector,
    capacity_operator,
    combine,
    compile_form,
    conduction_operator,
    damping_operator,
    diffusion_operator,
    force_vector,
    heat_capacity_operator,
    heat_capacity_vector,
    heat_conduction_operator,
    heat_source_vector,
    load_vector,
    lumped_mass,
    lumped_operator,
    mass_action_vector,
    mass_operator,
    scale,
    source_vector,
    stiffness,
    quadratic_form,
    xtmx,
    xtmy,
)
from .elasticity import elastic_stiffness, internal_force_vector, stiffness_operator
from .system import LinearSystem, SecondOrderSystem, linear_system

assemble_lumped_mass = lumped_mass
assemble_lumped_operator = lumped_operator

__all__ = [
    "LinearSystem",
    "OperatorForm",
    "SecondOrderSystem",
    "action",
    "assemble_lumped_mass",
    "assemble_lumped_operator",
    "assemble_matrix",
    "assemble_vector",
    "bilinear_form",
    "body_force_vector",
    "boundary_load_vector",
    "boundary_force_vector",
    "boundary_model_vector",
    "capacity_operator",
    "combine",
    "compile_form",
    "conduction_operator",
    "damping_operator",
    "diffusion_operator",
    "elastic_stiffness",
    "elasticity",
    "force_vector",
    "heat_capacity_operator",
    "heat_capacity_vector",
    "heat_conduction_operator",
    "heat_source_vector",
    "internal_force_vector",
    "linear_system",
    "load_vector",
    "lumped_mass",
    "lumped_operator",
    "mass_action_vector",
    "mass_operator",
    "scale",
    "source_vector",
    "stiffness",
    "stiffness_operator",
    "quadratic_form",
    "xtmx",
    "xtmy",
]
