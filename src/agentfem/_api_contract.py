"""Single source of truth for AgentFEM's discoverable product language.

This module deliberately imports no numerical dependency.  Package discovery,
the CLI, documentation generation, IDE integrations, and future GUI clients
must see the same module, Model-verb, command, and workflow inventories.
"""

from __future__ import annotations


CORE_WORKFLOW_MODULES = (
    "studies",
    "mesh",
    "models",
    "fields",
    "materials",
    "constitutive",
    "constraints",
    "amplitudes",
    "loads",
    "project",
    "results",
    "steps",
    "units",
    "verification",
)

ADVANCED_WORKFLOW_MODULES = (
    "assessments",
    "boundary_models",
    "campaigns",
    "convergence",
    "checkpointing",
    "coordinates",
    "datasets",
    "events",
    "expressions",
    "fatigue_fracture",
    "fracture",
    "histories",
    "interfaces",
    "learning",
    "mechanics",
    "operators",
    "procedures",
    "responses",
    "solvers",
    "surrogates",
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
    "io",
    "integrations",
    "ir",
    "platforms",
    "problems",
    "provenance",
    "spaces",
    "time",
    "upgrades",
    "validation",
)

CORE_MODEL_API = (
    "field",
    "amplitude",
    "material",
    "constraint",
    "fix",
    "prescribe",
    "clamp",
    "prescribed_temperature",
    "periodic",
    "load",
    "traction",
    "surface_force",
    "body_force",
    "heat_flux",
    "heat_source",
    "gravity",
    "pressure",
    "symmetry",
    "roller",
    "convection",
    "stage",
    "step",
    "validate",
    "check",
    "summary",
    "tree",
)

ADVANCED_MODEL_API = (
    "remote_displacement",
    "distributing_coupling",
    "remote_force",
    "centrifugal",
    "hydrostatic_pressure",
    "absorbing_boundary",
    "elastic_foundation",
    "stiffness",
    "mass",
    "damping",
    "conduction",
    "heat_capacity",
    "thermal_expansion",
    "lumped_mass",
    "load_vector",
    "external_force",
    "internal_force",
    "boundary_force",
    "force_balance",
    "operator",
    "add_region",
    "bcs",
    "audit_boundaries",
    "manifest",
    "to_ir",
    "write_ir",
)

COMPATIBILITY_MODEL_API = (
    "add_field",
    "add_amplitude",
    "add_material",
    "add_constraint",
    "add_load",
    "add_step",
    "add_boundary_model",
    "internal_force_vector",
    "linear_static_step",
    "heat_transfer_step",
    "hyperelastic_step",
    "mixed_hyperelastic_step",
    "j2_plasticity_step",
    "creep_step",
    "explicit_dynamics_step",
    "finite_strain_explicit_dynamics_step",
    "implicit_dynamics_step",
)

COMPATIBILITY_MODEL_REPLACEMENTS = {
    "add_field": "field",
    "add_amplitude": "amplitude",
    "add_material": "material",
    "add_constraint": "constraint",
    "add_load": "load",
    "add_step": "step",
    "add_boundary_model": "convection or another named boundary-model verb",
    "internal_force_vector": "internal_force",
    "linear_static_step": "step",
    "heat_transfer_step": "step",
    "hyperelastic_step": "step",
    "mixed_hyperelastic_step": "step",
    "j2_plasticity_step": "step",
    "creep_step": "step",
    "explicit_dynamics_step": "step",
    "finite_strain_explicit_dynamics_step": "step",
    "implicit_dynamics_step": "step",
}

CLI_COMMANDS = (
    "doctor",
    "init",
    "templates",
    "check",
    "upgrade",
    "run",
    "inspect",
    "inspect-abaqus",
    "migrate-abaqus",
    "lower-abaqus",
    "verify",
    "capabilities",
    "extensions",
)

MACHINE_COMMANDS = {
    "environment_check": "agentfem doctor --json",
    "capabilities": "agentfem capabilities --json",
    "project_check": "agentfem check --json",
    "run": "agentfem run --json",
    "inspect": "agentfem inspect --json",
    "abaqus_inspect": "agentfem inspect-abaqus model.inp --json",
    "abaqus_migrate": "agentfem migrate-abaqus model.inp ./project --json",
    "abaqus_lower": (
        "agentfem lower-abaqus ./project --reviewed-by REVIEWER "
        "--unit-system SI --json"
    ),
    "verify": "agentfem verify --json",
}

WORKFLOW_STAGES = (
    "study",
    "model",
    "mesh_and_regions",
    "fields",
    "materials",
    "loads_and_constraints",
    "step",
    "solve",
    "result_and_verification",
)

CAPABILITIES_SCHEMA_VERSION = "0.2.2"


def _all(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


PUBLIC_WORKFLOW_MODULES = _all(
    CORE_WORKFLOW_MODULES,
    ADVANCED_WORKFLOW_MODULES,
    EXPERT_WORKFLOW_MODULES,
)

PUBLIC_MODEL_API = _all(
    CORE_MODEL_API,
    ADVANCED_MODEL_API,
    COMPATIBILITY_MODEL_API,
)


def workflow_modules(level: str = "all") -> tuple[str, ...]:
    """Return public workflow modules at one progressive-disclosure level."""

    return _level(
        level,
        {
            "core": CORE_WORKFLOW_MODULES,
            "advanced": ADVANCED_WORKFLOW_MODULES,
            "expert": EXPERT_WORKFLOW_MODULES,
            "all": PUBLIC_WORKFLOW_MODULES,
        },
        noun="public_api",
    )


def model_methods(level: str = "core") -> tuple[str, ...]:
    """Return Model facade methods at one discovery level."""

    return _level(
        level,
        {
            "core": CORE_MODEL_API,
            "advanced": ADVANCED_MODEL_API,
            "compatibility": COMPATIBILITY_MODEL_API,
            "all": PUBLIC_MODEL_API,
        },
        noun="model_api",
    )


def model_method_contract(level: str = "all") -> tuple[dict[str, object], ...]:
    """Return lifecycle metadata for the discoverable Model vocabulary.

    Compatibility methods remain executable.  The record makes their status
    and preferred spelling explicit without emitting runtime warnings or
    rewriting scientific Python behind the user's back.
    """

    selected = model_methods(level)
    return tuple(_model_method_record(name) for name in selected)


def _model_method_record(name: str) -> dict[str, object]:
    if name in CORE_MODEL_API:
        return {
            "name": name,
            "tier": "core",
            "lifecycle": "recommended",
            "replacement": None,
            "semantic_review": False,
        }
    if name in ADVANCED_MODEL_API:
        return {
            "name": name,
            "tier": "advanced",
            "lifecycle": "supported",
            "replacement": None,
            "semantic_review": False,
        }
    replacement = COMPATIBILITY_MODEL_REPLACEMENTS[name]
    return {
        "name": name,
        "tier": "compatibility",
        "lifecycle": "compatibility",
        "replacement": (
            f"model.{replacement}(...)" if " " not in replacement else replacement
        ),
        "semantic_review": name.endswith("_step"),
        "automatic_rewrite": False,
    }


def _level(
    level: str,
    levels: dict[str, tuple[str, ...]],
    *,
    noun: str,
) -> tuple[str, ...]:
    selected = str(level).lower().replace("-", "_").strip()
    try:
        return levels[selected]
    except KeyError as exc:
        choices = ", ".join((*tuple(levels)[:-1], f"or {tuple(levels)[-1]}"))
        raise ValueError(f"{noun} level must be {choices}.") from exc
