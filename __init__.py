"""Reusable finite-element workflow tools built on DOLFINx/PETSc."""

__version__ = "0.2.1"

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
from . import fatigue_fracture
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

CORE_WORKFLOW_MODULES = (
    "studies",
    "mesh",
    "models",
    "fields",
    "materials",
    "constitutive",
    "coordinates",
    "constraints",
    "amplitudes",
    "loads",
    "project",
    "procedures",
    "results",
    "solvers",
    "steps",
    "units",
    "io",
    "verification",
)

ADVANCED_WORKFLOW_MODULES = (
    "boundary_models",
    "campaigns",
    "checkpointing",
    "datasets",
    "fatigue_fracture",
    "fracture",
    "interfaces",
    "mechanics",
    "operators",
    "problems",
    "surrogates",
    "time",
)

EXPERT_WORKFLOW_MODULES = (
    "assembly",
    "backends",
    "benchmarks",
    "dependencies",
    "diagnostics",
    "elements",
    "extensions",
    "forms",
    "ir",
    "platforms",
    "provenance",
    "spaces",
    "upgrades",
    "validation",
)

# Compatibility inventory retained throughout the 0.2.x convergence series.
PUBLIC_WORKFLOW_MODULES = tuple(
    dict.fromkeys(
        CORE_WORKFLOW_MODULES
        + ADVANCED_WORKFLOW_MODULES
        + EXPERT_WORKFLOW_MODULES
    )
)


def public_api(level: str = "all") -> tuple[str, ...]:
    """Return public modules at a declared discovery level.

    ``public_api()`` preserves the complete 0.2.0 inventory. New frontends and
    tutorials should query ``level="core"`` first, then disclose advanced or
    expert modules only when the workflow requires them.
    """

    selected = str(level).lower().replace("-", "_").strip()
    levels = {
        "core": CORE_WORKFLOW_MODULES,
        "advanced": ADVANCED_WORKFLOW_MODULES,
        "expert": EXPERT_WORKFLOW_MODULES,
        "all": PUBLIC_WORKFLOW_MODULES,
    }
    try:
        return levels[selected]
    except KeyError as exc:
        raise ValueError(
            "public_api level must be core, advanced, expert, or all."
        ) from exc


__all__ = [
    "PUBLIC_WORKFLOW_MODULES",
    "CORE_WORKFLOW_MODULES",
    "ADVANCED_WORKFLOW_MODULES",
    "EXPERT_WORKFLOW_MODULES",
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
    "fatigue_fracture",
    "fracture",
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
