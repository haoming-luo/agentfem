# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared completion helpers for model-owned analysis Steps."""

from __future__ import annotations

from collections.abc import Mapping


def execution_context(step):
    """Return the context bound by :meth:`Model.step`, when available."""

    return getattr(step, "execution_context", None)


def attach_checkpoint_contract(step, result):
    """Publish the exact checkpoint contract preflighted for *step*.

    Model-owned Steps carry the canonical record in their execution context.
    Low-level Steps remain inspectable through their own declaration without
    pretending that a policy was requested.
    """

    context = execution_context(step)
    contract = (
        None if context is None else getattr(context, "checkpoint_contract", None)
    )
    if contract is None:
        from .. import checkpointing

        contract = checkpointing.preflight_contract(
            step,
            getattr(step, "checkpoint_policy", None),
        )
    if contract is not None:
        result.add_checkpoint_contract(contract)
    return result


def complete_result(
    step,
    result,
    *,
    output=None,
    fields=(),
    strict_output: bool = False,
    deformation_scale: float = 0.0,
    time: float = 0.0,
    metadata: Mapping[str, object] | None = None,
):
    """Complete output and metadata through one compatibility-safe path.

    A path-like ``output`` writes the final live field collection.  A
    declarative :class:`OutputPlan` owns finite-strain frames, histories,
    presentation products, IR, and the result manifest.  When a plan was
    supplied to ``model.step(output=...)``, ``solve_result()`` consumes it
    automatically.  ``OutputPlan.finalize`` is idempotent so existing callers
    that still finalize manually remain safe during the 0.2.x migration.
    """

    context = execution_context(step)
    selected_output = output
    if selected_output is None and context is not None:
        selected_output = context.configured_output

    if metadata:
        result.metadata.update(dict(metadata))
    if context is not None:
        result.metadata.setdefault("execution_context", context.summary())
    engineering = getattr(step, "engineering_step_summary", None)
    if engineering is not None:
        from copy import deepcopy
        result.metadata["engineering_step"] = deepcopy(engineering)
    attach_checkpoint_contract(step, result)
    if selected_output is None:
        return result

    from .plan import OutputPlan

    if isinstance(selected_output, OutputPlan):
        if context is None:
            raise ValueError(
                "Declarative OutputPlan requires a Step created by model.step(...)."
            )
        if context.material is None:
            raise ValueError(
                "Declarative finite-strain output requires one resolved material."
            )
        return selected_output.finalize(
            model=context.model,
            step=step,
            result=result,
            target=context.output_target,
            material=context.material,
            metadata=metadata,
        )

    from .output import attach_result_field_output

    attach_result_field_output(
        result,
        selected_output,
        time=time,
        names=tuple(fields),
        deformation_scale=float(deformation_scale),
        strict=bool(strict_output),
    )
    return result
