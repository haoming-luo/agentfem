"""Reusable finite-element workflow tools built on DOLFINx/PETSc."""

from importlib import import_module as _import_module

from ._api_contract import (
    ADVANCED_WORKFLOW_MODULES,
    CORE_WORKFLOW_MODULES,
    EXPERT_WORKFLOW_MODULES,
    PUBLIC_WORKFLOW_MODULES,
    workflow_modules as _workflow_modules,
)

__version__ = "0.2.3"


def public_api(level: str = "all") -> tuple[str, ...]:
    """Return public modules at a declared discovery level.

    ``public_api()`` preserves the complete 0.2.0 inventory. New frontends and
    tutorials should query ``level="core"`` first, then disclose advanced or
    expert modules only when the workflow requires them.
    """

    return _workflow_modules(level)


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


__all__ = (
    "PUBLIC_WORKFLOW_MODULES",
    "CORE_WORKFLOW_MODULES",
    "ADVANCED_WORKFLOW_MODULES",
    "EXPERT_WORKFLOW_MODULES",
    "__version__",
    "public_api",
    *PUBLIC_WORKFLOW_MODULES,
)
