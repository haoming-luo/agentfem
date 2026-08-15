"""Shared completion helpers for model-owned analysis Steps."""

from __future__ import annotations

from collections.abc import Mapping


def execution_context(step):
    """Return the context bound by :meth:`Model.step`, when available."""

    return getattr(step, "execution_context", None)


def complete_result(
    step,
    result,
    *,
    output=None,
    fields=(),
    strict_output: bool = False,
    deformation_scale: float = 0.0,
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
        names=tuple(fields),
        deformation_scale=float(deformation_scale),
        strict=bool(strict_output),
    )
    return result
