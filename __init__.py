"""Reusable finite-element workflow tools built on DOLFINx/PETSc."""

__version__ = "0.2.0a1"

from . import amplitudes
from . import assembly
from . import backends
from . import boundary_models
from . import benchmarks
from . import campaigns
from . import constraints
from . import constitutive
from . import datasets
from . import dependencies
from . import diagnostics
from . import fields
from . import elements
from . import forms
from . import io
from . import ir
from . import loads
from . import materials
from . import mechanics
from . import mesh
from . import models
from . import operators
from . import platforms
from . import problems
from . import procedures
from . import results
from . import solvers
from . import spaces
from . import steps
from . import studies
from . import surrogates
from . import time
from . import validation
from . import verification

PUBLIC_WORKFLOW_MODULES = (
    "studies",
    "mesh",
    "models",
    "fields",
    "materials",
    "mechanics",
    "constitutive",
    "constraints",
    "amplitudes",
    "loads",
    "operators",
    "problems",
    "platforms",
    "procedures",
    "results",
    "solvers",
    "steps",
    "time",
    "io",
    "diagnostics",
    "ir",
    "campaigns",
    "datasets",
    "surrogates",
    "validation",
    "verification",
)


def public_api() -> tuple[str, ...]:
    """Return the stable beginner-facing AgentFEM workflow modules."""

    return PUBLIC_WORKFLOW_MODULES


__all__ = [
    "PUBLIC_WORKFLOW_MODULES",
    "__version__",
    "amplitudes",
    "assembly",
    "backends",
    "boundary_models",
    "benchmarks",
    "campaigns",
    "constraints",
    "constitutive",
    "datasets",
    "dependencies",
    "diagnostics",
    "fields",
    "elements",
    "forms",
    "io",
    "ir",
    "loads",
    "materials",
    "mechanics",
    "mesh",
    "models",
    "operators",
    "platforms",
    "problems",
    "procedures",
    "results",
    "solvers",
    "spaces",
    "steps",
    "studies",
    "surrogates",
    "time",
    "validation",
    "verification",
    "public_api",
]
