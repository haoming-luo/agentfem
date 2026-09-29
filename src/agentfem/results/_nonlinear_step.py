# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private result assembly for nonlinear load-path procedures."""

from __future__ import annotations

from .. import constraints as constraint_api
from .. import fields as field_api
from ._field_metadata import field_location, generated_field_processing
from .core import from_solution
from .execution import add_execution_trace
from .lifecycle import complete_result
from .quantities import static_force_balance


def from_incremental_nonlinear_step(
    step,
    solution,
    *,
    output=None,
    fields=(),
    strict_output: bool = False,
    metadata=None,
):
    """Build a result after an ordinary nonlinear load path has converged."""

    result = _base_nonlinear_result(step, solution, fields=fields)
    for checkpoint in step.checkpoints:
        result.add_checkpoint(checkpoint)
    _add_nonlinear_constraint_evidence(step, result)
    add_execution_trace(result, step.execution_events)
    return complete_result(
        step,
        result,
        output=output,
        fields=fields,
        strict_output=strict_output,
        metadata=metadata,
    )


def _add_nonlinear_constraint_evidence(step, result) -> None:
    assets = tuple(getattr(step, "constraint_assets", ()))
    if not assets:
        return
    provider_duals = constraint_api.collect_provider_duals(assets, step)
    result.metadata["constraint_duals"] = tuple(
        item.summary() for item in provider_duals
    )
    for item in provider_duals:
        distribution = item.distribution
        if distribution is not None:
            result.add_field(
                getattr(distribution, "name", f"{item.constraint_name}_reaction"),
                distribution,
                location="nodes",
                description=(
                    "Provider-owned reaction distribution reconstructed from "
                    "the converged nonlinear constraint contribution."
                ),
                processing={
                    "source": item.source,
                    "constraint": item.constraint_name,
                    "role": item.role,
                    "method": "provider_dual_reaction_distribution",
                },
            )
        prefix = item.constraint_name.lower().replace(" ", "_")
        if item.resultant is not None:
            result.add_quantity(
                f"{prefix}_reaction_resultant",
                item.resultant,
                kind="diagnostic",
            )
        for key, value in item.diagnostics.items():
            if key in {
                "contact_energy",
                "contact_moment",
                "penetration_l2_norm",
                "active_contact_measure",
            }:
                result.add_quantity(f"{prefix}_{key}", value, kind="diagnostic")

    constraint_work, work_evidence = _add_nonlinear_constraint_path_work(
        step,
        result,
    )
    contract = constraint_api.constraint_balance_contract(
        assets,
        provider_duals=provider_duals,
        provider_work=work_evidence,
    )
    result.metadata["constraint_balance_contract"] = contract
    result.metadata["constraint_work_evidence"] = tuple(
        item.summary() for item in work_evidence
    )
    try:
        equilibrium = static_force_balance(
            step,
            constraints=assets,
            provider_duals=provider_duals,
        )
    except (NotImplementedError, TypeError) as exc:
        result.metadata["static_equilibrium"] = {
            "status": "unavailable",
            "reason": str(exc),
            "reaction_scope": contract["reaction_scope"],
        }
    else:
        result.add_quantities(
            {
                "external_force_resultant": equilibrium.external,
                "reaction_force_resultant": equilibrium.reaction,
                "provider_reaction_force_resultant": equilibrium.provider_reaction,
                "force_balance_residual": equilibrium.residual,
                "relative_force_balance_error": equilibrium.relative_error,
            },
            kind="diagnostic",
        )
        result.metadata["static_equilibrium"] = equilibrium.as_dict()
    _add_nonlinear_energy_evidence(
        step,
        result,
        constraint_work=constraint_work,
    )


def _add_nonlinear_constraint_path_work(
    step,
    result,
) -> tuple[dict[str, object], tuple[object, ...]]:
    history = step.constraint_dual_history
    complete = history.complete(accepted_factor=step.accepted_load_factor)
    channels = {}
    work_evidence = []
    total = 0.0
    assets_by_name = {
        str(getattr(item, "name", type(item).__name__)): item
        for item in constraint_api.constraint_assets(step.constraint_assets)
    }
    for name in history.constraint_names:
        channel = history.channel(name)
        prefix = name.lower().replace(" ", "_")
        coordinate = channel["coordinate"]
        if coordinate is None:
            channels[name] = {
                "status": "unavailable",
                "reason": "provider did not publish a work-conjugate coordinate",
            }
            complete = False
            continue
        work = history.work(name)
        total += work
        channels[name] = {
            "status": "complete",
            "value": work,
            "role": channel["role"],
            "source": channel["source"],
        }
        result.add_history(
            f"{prefix}_generalized_force",
            history.factors,
            channel["force"],
            abscissa_name="load_factor",
            abscissa_unit=None,
            description=(
                "Provider-owned generalized constraint force at accepted "
                "nonlinear stations."
            ),
        )
        result.add_history(
            f"{prefix}_generalized_coordinate",
            history.factors,
            coordinate,
            abscissa_name="load_factor",
            abscissa_unit=None,
            description=(
                "Work-conjugate constraint coordinate at accepted nonlinear "
                "stations."
            ),
        )
        result.add_quantity(
            f"{prefix}_path_work",
            work,
            kind="diagnostic",
            description=(
                "Trapezoidal provider-dual work over accepted nonlinear states."
            ),
        )
        if complete:
            work_evidence.append(
                constraint_api.constraint_work(
                    assets_by_name[name],
                    value=work,
                    integration="accepted_force_coordinate_trapezoidal",
                    sample_count=len(history.records),
                    role=channel["role"],
                    source=channel["source"],
                )
            )
    record = {
        "status": "complete" if complete else "unavailable",
        "sample_count": len(history.records),
        "integration": "accepted_force_coordinate_trapezoidal",
        "channels": channels,
        "total": total if complete else None,
    }
    result.metadata["constraint_path_work"] = record
    return record, tuple(work_evidence)


def _add_nonlinear_energy_evidence(
    step,
    result,
    *,
    constraint_work: dict[str, object],
) -> None:
    recorder = getattr(step, "accepted_history_recorders", {}).get(
        "conservative_energy"
    )
    if recorder is None:
        result.metadata["static_work"] = {
            "status": "unavailable",
            "reason": (
                "The nonlinear provider did not publish accepted natural-load "
                "and stored-energy histories."
            ),
        }
        return
    evidence = dict(recorder.evidence(accepted_factor=step.accepted_load_factor))
    frames = tuple(recorder.frames)
    factors = tuple(item.load_factor for item in frames)
    result.add_history(
        "natural_load_generalized_coordinate",
        factors,
        tuple(item.natural_load_coordinate for item in frames),
        abscissa_name="load_factor",
        abscissa_unit=None,
        description=(
            "Dual product of the fixed dead-load pattern with the accepted "
            "displacement field."
        ),
    )
    for name in recorder.stored_energy_evaluators:
        result.add_history(
            name,
            factors,
            tuple(item.stored_energy_components[name] for item in frames),
            abscissa_name="load_factor",
            abscissa_unit=None,
            description=f"Accepted conservative energy component {name}.",
        )
    result.add_history(
        "stored_energy",
        factors,
        tuple(item.stored_energy for item in frames),
        abscissa_name="load_factor",
        abscissa_unit=None,
        description="Total recoverable energy at accepted nonlinear boundaries.",
    )

    reasons = []
    if evidence.get("status") != "complete":
        reasons.append(str(evidence.get("reason") or "energy history is incomplete"))
    if constraint_work.get("status") != "complete":
        reasons.append("constraint provider work history is incomplete")
    provider_work = (
        0.0
        if constraint_work.get("total") is None
        else float(constraint_work["total"])
    )
    natural_work = float(evidence.get("natural_load_work", 0.0))
    stored_change = float(evidence.get("stored_energy_change", 0.0))
    external_work = natural_work + provider_work
    balance_error = external_work - stored_change
    scale = max(abs(external_work), abs(stored_change), 1.0e-30)
    relative_error = abs(balance_error) / scale
    record = {
        **evidence,
        "status": "complete" if not reasons else "unavailable",
        "reason": None if not reasons else "; ".join(reasons),
        "prescribed_motion_work": 0.0 if recorder.zero_prescribed_motion else None,
        "provider_constraint_work": provider_work,
        "external_work": external_work,
        "energy_balance_error": balance_error,
        "relative_energy_balance_error": relative_error,
        "reaction_scope": "accepted nonlinear natural and provider-dual paths",
    }
    result.metadata["static_work"] = record
    if record["status"] != "complete":
        return
    quantities = {
        "natural_load_work": natural_work,
        "provider_constraint_work": provider_work,
        "external_work": external_work,
        "stored_energy_change": stored_change,
        "energy_balance_error": balance_error,
        "relative_energy_balance_error": relative_error,
    }
    quantities.update(evidence.get("stored_energy_components", {}))
    result.add_quantities(quantities, kind="diagnostic")


def from_affine_nonlinear_step(
    step,
    solution,
    *,
    output=None,
    fields=(),
    strict_output: bool = False,
    metadata=None,
):
    """Build a result with state, checkpoint and affine-dual evidence."""

    result = _base_nonlinear_result(
        step,
        solution,
        fields=fields,
        extra_metadata={"state": _state_summary(step.state_transaction)},
    )
    if step.state_transaction is not None and hasattr(
        step.state_transaction, "populate_result"
    ):
        step.state_transaction.populate_result(result)
    for checkpoint in step.checkpoints:
        result.add_checkpoint(checkpoint)
    _add_affine_constraint_evidence(step, result)
    add_execution_trace(result, step.execution_events)
    return complete_result(
        step,
        result,
        output=output,
        strict_output=strict_output,
        metadata=metadata,
    )


def _base_nonlinear_result(
    step,
    solution,
    *,
    fields=(),
    extra_metadata=None,
):
    generated = (
        () if step.result_field_factory is None else tuple(step.result_field_factory())
    )
    primary = solution if not generated else generated[0]
    selected_metadata = {
        "problem": step.summary(),
        "solve": step.last_solve_info.as_dict(),
    }
    if extra_metadata:
        selected_metadata.update(dict(extra_metadata))
    result = from_solution(primary, name=step.name, metadata=selected_metadata)
    for generated_field in generated[1:]:
        result.add_field(
            getattr(generated_field, "name", type(generated_field).__name__),
            generated_field,
            location=field_location(generated_field),
            processing=generated_field_processing(step, generated_field),
        )
    for selected in fields:
        function = field_api.unwrap(selected)
        result.add_field(
            getattr(function, "name", type(function).__name__),
            function,
            location=field_location(function),
        )
    return result


def _add_affine_constraint_evidence(step, result) -> None:
    provider_duals = constraint_api.collect_provider_duals(
        (step.constraint,),
        step,
    )
    work_evidence = _add_affine_path_work(step, result)
    result.metadata["constraint_balance_contract"] = (
        constraint_api.constraint_balance_contract(
            (step.constraint,),
            provider_duals=provider_duals,
            provider_work=work_evidence,
        )
    )
    result.metadata["constraint_duals"] = tuple(
        item.summary() for item in provider_duals
    )
    result.metadata["constraint_work_evidence"] = tuple(
        item.summary() for item in work_evidence
    )
    if len(provider_duals) == 1:
        dual = provider_duals[0]
        result.add_quantities(
            {
                "affine_path_generalized_reaction": float(dual.force[0]),
                "affine_constraint_force_resultant": dual.resultant,
            },
            kind="diagnostic",
            descriptions={
                "affine_path_generalized_reaction": (
                    "Virtual work of the converged full residual against a "
                    "unit increment of the prescribed affine path."
                ),
                "affine_constraint_force_resultant": (
                    "Physical-space resultant of the converged displacement "
                    "residual owned by the affine constraint provider."
                ),
            },
        )


def _add_affine_path_work(step, result) -> tuple[object, ...]:
    history = tuple(step.constraint_dual_history.records)
    complete = bool(
        len(history) >= 2
        and abs(float(history[0]["load_factor"])) <= 1.0e-12
        and abs(float(history[-1]["load_factor"]) - step.accepted_load_factor)
        <= 1.0e-12
    )
    contract = {
        "status": "complete" if complete else "unavailable",
        "sample_count": len(history),
        "integration": "accepted_path_trapezoidal",
        "reason": (
            None
            if complete
            else "Accepted generalized-force history does not start at zero."
        ),
    }
    result.metadata["affine_constraint_path_work"] = contract
    if not complete:
        return ()
    factors = step.constraint_dual_history.factors
    forces = step.constraint_dual_history.forces
    path_work = step.constraint_dual_history.work()
    result.add_history(
        "affine_path_generalized_reaction",
        factors,
        forces,
        abscissa_name="load_factor",
        description=(
            "Full-residual generalized reaction conjugate to the accepted "
            "affine path coordinate."
        ),
    )
    result.add_history(
        "affine_path_outgoing_generalized_reaction",
        factors,
        step.constraint_dual_history.outgoing_forces,
        abscissa_name="load_factor",
        description=(
            "Right-sided generalized reaction at affine path knots; equal to "
            "the incoming value on a smooth path."
        ),
    )
    result.add_quantity(
        "affine_constraint_path_work",
        path_work,
        kind="diagnostic",
        description=(
            "Trapezoidal work of the affine generalized reaction over all "
            "accepted load-path increments."
        ),
    )
    contract["value"] = path_work
    return (
        constraint_api.constraint_work(
            step.constraint,
            value=path_work,
            integration="accepted_path_trapezoidal",
            sample_count=len(history),
            role="mpc_constraint",
            source=str(history[-1]["source"]),
            diagnostics={
                "accepted_factor": float(step.accepted_load_factor),
                "segment_count": len(history) - 1,
            },
        ),
    )


def _state_summary(transaction):
    if transaction is None:
        return None
    if hasattr(transaction, "summary"):
        return transaction.summary()
    return {"kind": type(transaction).__name__}


__all__ = ()
