"""Reusable finite-element workflow tools built on DOLFINx/PETSc."""

__version__ = "0.2.0a1"

from . import amplitudes
from . import assembly
from . import backends
from . import boundary_models
from . import benchmarks
from . import campaigns
from . import checkpointing
from . import constraints
from . import constitutive
from . import coordinates
from . import datasets
from . import dependencies
from . import diagnostics
from . import fields
from . import fracture
from . import elements
from . import extensions
from . import forms
from . import io
from . import ir
from . import interfaces
from . import loads
from . import materials
from . import mechanics
from . import mesh
from . import models
from . import operators
from . import platforms
from . import problems
from . import project
from . import provenance
from . import procedures
from . import results
from . import solvers
from . import spaces
from . import steps
from . import studies
from . import surrogates
from . import time
from . import units
from . import upgrades
from . import validation
from . import verification

PUBLIC_WORKFLOW_MODULES = (
    "studies",
    "mesh",
    "models",
    "fields",
    "fracture",
    "materials",
    "mechanics",
    "constitutive",
    "coordinates",
    "constraints",
    "amplitudes",
    "loads",
    "operators",
    "problems",
    "project",
    "provenance",
    "platforms",
    "procedures",
    "results",
    "solvers",
    "steps",
    "time",
    "units",
    "upgrades",
    "io",
    "diagnostics",
    "extensions",
    "ir",
    "interfaces",
    "campaigns",
    "checkpointing",
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
    "checkpointing",
    "constraints",
    "constitutive",
    "coordinates",
    "datasets",
    "dependencies",
    "diagnostics",
    "fields",
    "elements",
    "extensions",
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
    "project",
    "provenance",
    "procedures",
    "results",
    "solvers",
    "spaces",
    "steps",
    "studies",
    "surrogates",
    "time",
    "units",
    "upgrades",
    "validation",
    "verification",
    "public_api",
]
