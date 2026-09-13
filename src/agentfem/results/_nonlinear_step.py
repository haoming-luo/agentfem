"""Private result assembly for nonlinear load-path procedures."""

from __future__ import annotations

from .. import constraints as constraint_api
from .. import fields as field_api
from ._field_metadata import field_location, generated_field_processing
from .core import from_solution
from .execution import add_execution_trace
from .lifecycle import complete_result


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
    add_execution_trace(result, step.execution_events)
    return complete_result(
        step,
        result,
        output=output,
        fields=fields,
        strict_output=strict_output,
        metadata=metadata,
    )


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
    result.metadata["constraint_balance_contract"] = (
        constraint_api.constraint_balance_contract(
            (step.constraint,),
            provider_duals=provider_duals,
        )
    )
    result.metadata["constraint_duals"] = tuple(
        item.summary() for item in provider_duals
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
    _add_affine_path_work(step, result)


def _add_affine_path_work(step, result) -> None:
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
        return
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


def _state_summary(transaction):
    if transaction is None:
        return None
    if hasattr(transaction, "summary"):
        return transaction.summary()
    return {"kind": type(transaction).__name__}


__all__ = ()
