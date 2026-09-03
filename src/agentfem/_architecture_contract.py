"""Dependency-free ownership contract for AgentFEM's stable middle layer.

The contract says who owns a scientific decision.  It is intentionally
smaller than the package inventory: modules may grow, but they must continue
to lower through these boundaries instead of turning ``Model`` or one solver
class into a numerical god object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OwnershipBoundary:
    """One stable responsibility in the AgentFEM execution architecture."""

    name: str
    question: str
    owns: tuple[str, ...]
    excludes: tuple[str, ...]
    modules: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "question": self.question,
            "owns": self.owns,
            "excludes": self.excludes,
            "modules": self.modules,
        }


OWNERSHIP_BOUNDARIES = (
    OwnershipBoundary(
        name="model",
        question="What engineering problem is being solved?",
        owns=(
            "study and geometry",
            "regions and fields",
            "material assignments",
            "loads and constraints",
            "inspectable engineering steps",
        ),
        excludes=(
            "nonlinear iteration",
            "time integration",
            "constitutive history",
            "result acceptance",
        ),
        modules=(
            "studies",
            "models",
            "mesh",
            "fields",
            "materials",
            "loads",
            "constraints",
            "boundary_models",
        ),
    ),
    OwnershipBoundary(
        name="constitutive",
        question="How does material response map kinematics and history to response?",
        owns=(
            "material-point update laws",
            "stress and consistent tangent",
            "declared internal-variable schema",
        ),
        excludes=(
            "global equilibrium",
            "mesh traversal policy",
            "accepted state lifetime",
        ),
        modules=("constitutive", "mechanics"),
    ),
    OwnershipBoundary(
        name="state",
        question="Which accepted and trial quantities must survive numerical evolution?",
        owns=(
            "accepted and trial history",
            "commit and rollback boundary",
            "restart snapshots",
            "time-level fields",
        ),
        excludes=(
            "material equations",
            "solver selection",
            "file-format policy",
        ),
        modules=("state", "checkpointing"),
    ),
    OwnershipBoundary(
        name="operator",
        question="What mathematical contribution is assembled or applied?",
        owns=(
            "residual, tangent, mass, damping, and source identity",
            "operator composition",
            "form provenance",
        ),
        excludes=(
            "analysis sequencing",
            "state acceptance",
            "verification claims",
        ),
        modules=("operators", "forms", "assembly"),
    ),
    OwnershipBoundary(
        name="procedure",
        question="How is the problem advanced and solved?",
        owns=(
            "solution family and algorithm",
            "increment and iteration policy",
            "provider dispatch and option contract",
        ),
        excludes=(
            "engineering model definition",
            "backend algebra implementation",
            "scientific acceptance",
        ),
        modules=("procedures", "step_providers", "_step_builders", "problems"),
    ),
    OwnershipBoundary(
        name="backend",
        question="Which numerical runtime compiles, assembles, and executes the forms?",
        owns=(
            "form compilation",
            "finite-element assembly",
            "DOF and linear-algebra execution",
            "runtime capability identity",
        ),
        excludes=(
            "engineering semantics",
            "material selection",
            "verification policy",
        ),
        modules=("backends", "kernel"),
    ),
    OwnershipBoundary(
        name="result_verification",
        question="What was computed, and what evidence makes it usable?",
        owns=(
            "quantities, fields, histories, and artifacts",
            "provenance and failure semantics",
            "verification evidence and acceptance status",
        ),
        excludes=(
            "solver mutation",
            "constitutive state evolution",
            "model construction",
        ),
        modules=(
            "results",
            "verification",
            "provenance",
            "convergence",
            "_work_energy",
        ),
    ),
)


# These are architectural impossibilities, not a complete import allow-list.
# The rules remain deliberately small so a useful implementation detail does
# not become an artificial abstraction layer.
FORBIDDEN_IMPORTS = {
    "models": ("problems", "results", "solvers", "time", "kernel"),
    "state": ("models", "problems", "step_providers", "_step_builders", "results"),
    "operators": ("models", "problems", "step_providers", "_step_builders", "results"),
    "procedures": ("models", "problems", "backends", "results"),
    "backends": (
        "models",
        "constitutive",
        "mechanics",
        "operators",
        "procedures",
        "problems",
        "results",
        "step_providers",
        "_step_builders",
    ),
}


def ownership_contract() -> tuple[dict[str, object], ...]:
    """Return the stable, machine-readable ownership inventory."""

    return tuple(item.as_dict() for item in OWNERSHIP_BOUNDARIES)


__all__ = (
    "FORBIDDEN_IMPORTS",
    "OWNERSHIP_BOUNDARIES",
    "OwnershipBoundary",
    "ownership_contract",
)
