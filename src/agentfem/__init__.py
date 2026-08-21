"""Reusable finite-element workflow tools built on DOLFINx/PETSc."""

from importlib import import_module as _import_module

__version__ = "0.2.2"

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
    "events",
    "expressions",
    "solvers",
    "steps",
    "units",
    "io",
    "verification",
)

ADVANCED_WORKFLOW_MODULES = (
    "boundary_models",
    "campaigns",
    "convergence",
    "checkpointing",
    "datasets",
    "fatigue_fracture",
    "fracture",
    "histories",
    "interfaces",
    "learning",
    "mechanics",
    "operators",
    "problems",
    "responses",
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
    "integrations",
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


def __getattr__(name: str):
    """Load public workflow modules only when first requested.

    The public surface remains ``from agentfem import mesh, models, ...`` while
    short scripts, CLI discovery, and agent workers avoid importing unrelated
    fracture, learning, campaign, and visualization stacks at startup.
    """

    if name in PUBLIC_WORKFLOW_MODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
    "convergence",
    "checkpointing",
    "constraints",
    "constitutive",
    "coordinates",
    "datasets",
    "dependencies",
    "diagnostics",
    "events",
    "expressions",
    "fields",
    "fatigue_fracture",
    "fracture",
    "elements",
    "extensions",
    "forms",
    "integrations",
    "io",
    "ir",
    "loads",
    "learning",
    "materials",
    "mechanics",
    "mesh",
    "models",
    "operators",
    "platforms",
    "problems",
    "project",
    "provenance",
    "responses",
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
