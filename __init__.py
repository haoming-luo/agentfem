"""Reusable finite-element workflow tools built on DOLFINx/PETSc."""

__version__ = "0.1.0"

from . import amplitudes
from . import assembly
from . import boundary_models
from . import benchmarks
from . import constraints
from . import constitutive
from . import diagnostics
from . import fields
from . import elements
from . import forms
from . import io
from . import loads
from . import materials
from . import mesh
from . import models
from . import operators
from . import problems
from . import solvers
from . import spaces
from . import studies
from . import time

PUBLIC_WORKFLOW_MODULES = (
    "studies",
    "mesh",
    "models",
    "fields",
    "materials",
    "constitutive",
    "constraints",
    "amplitudes",
    "loads",
    "operators",
    "problems",
    "solvers",
    "time",
    "io",
)


def public_api() -> tuple[str, ...]:
    """Return the stable beginner-facing AgentFEM workflow modules."""

    return PUBLIC_WORKFLOW_MODULES


__all__ = [
    "PUBLIC_WORKFLOW_MODULES",
    "__version__",
    "amplitudes",
    "assembly",
    "boundary_models",
    "benchmarks",
    "constraints",
    "constitutive",
    "diagnostics",
    "fields",
    "elements",
    "forms",
    "io",
    "loads",
    "materials",
    "mesh",
    "models",
    "operators",
    "problems",
    "solvers",
    "spaces",
    "studies",
    "time",
    "public_api",
]
