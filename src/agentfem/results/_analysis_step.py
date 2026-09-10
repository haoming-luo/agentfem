"""Private result assembly for the common linear analysis Step."""

from __future__ import annotations

from .. import constraints as constraint_api
from .. import fields as field_api
from .core import from_solution
from .lifecycle import complete_result
from .quantities import static_force_balance, static_work_balance


def from_analysis_step(
    step,
    solution,
    *,
    output=None,
    fields=(),
    field_variables=None,
    strict_output: bool = False,
    metadata=None,
):
    """Build and complete a result from an already solved analysis Step.

    The procedure layer supplies the converged field.  This private result
    boundary owns field records, constraint-dual evidence, static balance
    diagnostics, and output completion without mutating solver state.
    """

    if fields and field_variables is not None:
        raise ValueError("Choose explicit live fields or field_variables, not both.")
    if field_variables is not None and step.result_field_factory is None:
        raise ValueError(
            "This analysis step does not provide declarative derived fields."
        )
    generated = ()
    if not fields and step.result_field_factory is not None:
        generated = tuple(step.result_field_factory(field_variables))
    generated_ids = {id(item) for item in generated}
    selected_fields = tuple(fields) if fields else (solution, *generated)
    if not selected_fields:
        selected_fields = (solution,)
    result = from_solution(
        solution,
        name=step.name,
        metadata={
            "step": step.summary(),
            "study": (
                _describe_asset(step.study) if step.study is not None else None
            ),
        },
    )
    for item in selected_fields:
        function = field_api.unwrap(item)
        if function is solution:
            continue
        result.add_field(
            getattr(function, "name", type(function).__name__),
            function,
            location=_field_location(function),
            description=(
                "Constitutive result projected to a discontinuous finite-"
                "element space; no nodal extrapolation or interelement "
                "smoothing is applied."
                if id(item) in generated_ids
                else ""
            ),
            processing=(
                _projected_field_processing(function)
                if id(item) in generated_ids
                else None
            ),
        )
    is_static_solid = bool(
        step.study is not None
        and getattr(step.study, "is_solid_mechanics", False)
        and len(tuple(getattr(solution, "ufl_shape", ()))) == 1
    )
    if is_static_solid:
        _add_static_balance_evidence(step, result)
    return complete_result(
        step,
        result,
        output=output,
        strict_output=strict_output,
        metadata=metadata,
    )


def _add_static_balance_evidence(step, result) -> None:
    callback_duals = ()
    if step.constraint_dual_provider is not None:
        callback_duals = tuple(step.constraint_dual_provider(step.problem) or ())
    provider_duals = constraint_api.collect_provider_duals(
        step.constraint_assets,
        step.problem,
        extra=callback_duals,
    )
    balance_contract = constraint_api.constraint_balance_contract(
        step.constraint_assets,
        provider_duals=provider_duals,
    )
    result.metadata["constraint_balance_contract"] = balance_contract
    result.metadata["constraint_duals"] = tuple(
        item.summary() for item in provider_duals
    )
    for item in provider_duals:
        distribution = item.distribution
        if distribution is None:
            continue
        result.add_field(
            getattr(distribution, "name", f"{item.constraint_name}_reaction"),
            distribution,
            location="nodes",
            description=(
                "Provider-owned nodal reaction distribution reconstructed from "
                "the converged constraint dual."
            ),
            processing={
                "source": item.source,
                "constraint": item.constraint_name,
                "role": item.role,
                "method": "provider_dual_reaction_distribution",
            },
        )
    try:
        equilibrium = static_force_balance(
            step.problem,
            constraints=step.constraint_assets,
            provider_duals=provider_duals,
        )
    except NotImplementedError as exc:
        equilibrium = None
        result.metadata["static_equilibrium"] = {
            "status": "unavailable",
            "reason": str(exc),
            "reaction_scope": balance_contract["reaction_scope"],
        }
    try:
        work = static_work_balance(
            step.problem,
            constraints=step.constraint_assets,
            provider_duals=provider_duals,
        )
    except NotImplementedError as exc:
        work = None
        result.metadata["static_work"] = {
            "status": "unavailable",
            "reason": str(exc),
        }
    if equilibrium is not None:
        result.add_quantities(
            {
                "external_force_resultant": equilibrium.external,
                "reaction_force_resultant": equilibrium.reaction,
                "provider_reaction_force_resultant": equilibrium.provider_reaction,
                "force_balance_residual": equilibrium.residual,
                "relative_force_balance_error": equilibrium.relative_error,
            },
            kind="diagnostic",
            descriptions={
                "external_force_resultant": (
                    "Resultant of the assembled linear-system right-hand side."
                ),
                "reaction_force_resultant": (
                    "Resultant of all declared constraint reactions included "
                    "by the balance contract."
                ),
                "provider_reaction_force_resultant": (
                    "Physical-space resultant supplied by MPC, weak, or "
                    "contact providers."
                ),
                "force_balance_residual": (
                    "Reaction plus external-force resultant."
                ),
                "relative_force_balance_error": (
                    "Norm of the force-balance residual divided by the larger "
                    "external or reaction resultant norm."
                ),
            },
        )
        result.metadata["static_equilibrium"] = equilibrium.as_dict()
    if work is not None:
        result.add_quantities(
            {
                "strain_energy": work.strain_energy,
                "natural_load_work": work.natural_load_work,
                "prescribed_motion_work": work.prescribed_motion_work,
                "provider_constraint_work": work.provider_constraint_work,
                "external_work": work.external_work,
                "energy_balance_error": work.balance_error,
            },
            kind="diagnostic",
        )
        result.metadata["static_work"] = work.as_dict()


def _field_location(field) -> str:
    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    discontinuous = bool(
        getattr(element, "discontinuous", False)
        or getattr(basix_element, "discontinuous", False)
    )
    return "cells" if discontinuous else "nodes"


def _projected_field_processing(field) -> dict[str, object]:
    """Describe the post-processing contract of a projected result field."""

    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    degree = getattr(basix_element, "degree", None)
    family = getattr(basix_element, "family", None)
    selected_degree = None if degree is None else int(degree)
    return {
        "source_position": "constitutive_expression",
        "method": "global_l2_projection",
        "representation": (
            "cell_average" if selected_degree == 0 else "discontinuous_field"
        ),
        "space_family": (
            None if family is None else str(getattr(family, "name", family))
        ),
        "space_degree": selected_degree,
        "nodal_extrapolation": False,
        "interelement_smoothing": False,
        "material_boundary_averaging": False,
    }


def _describe_asset(asset) -> object:
    if hasattr(asset, "as_dict"):
        return asset.as_dict()
    if hasattr(asset, "summary"):
        return asset.summary()
    return getattr(asset, "name", repr(asset))


__all__ = ()
