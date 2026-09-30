# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Procedure-owned closest-point evaluation lifecycle.

The lifecycle calls an analytical surface or reviewed search backend for every
nonlinear evaluation, stages successful evidence in State, and exposes explicit
increment acceptance/rejection.  It deliberately does not assemble a contact
law, residual, or tangent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_state import (
    ContactProjectionRecord,
    ContactProjectionState,
)
from .rigid import SurfaceProjection, _readonly_array


def _validated_point_ids(value) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("Contact point IDs must be one-dimensional.")
    if raw.size and raw.dtype.kind not in {"i", "u"}:
        raise TypeError("Contact point IDs must be explicit integers.")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("Contact point IDs must fit signed 64-bit identity.")
    selected = raw.astype(np.int64, copy=False)
    if np.any(selected < 0):
        raise ValueError("Contact point IDs must be non-negative.")
    if np.unique(selected).size != selected.size:
        raise ValueError("Contact point IDs must be unique.")
    return _readonly_array(selected, dtype=np.int64)


@dataclass(frozen=True)
class ContactProjectionEvaluation:
    """One successful projection evaluation and optional search evidence."""

    attempt: int
    record: ContactProjectionRecord
    diagnostics: object | None = None

    def __post_init__(self) -> None:
        if int(self.attempt) < 1:
            raise ValueError("Contact projection attempt must be positive.")
        if not isinstance(self.record, ContactProjectionRecord):
            raise TypeError("Contact projection evaluation requires one record.")
        object.__setattr__(self, "attempt", int(self.attempt))

    def summary(self) -> dict[str, object]:
        diagnostics = self.diagnostics
        return {
            "attempt": self.attempt,
            "projection": self.record.summary(),
            "search_diagnostics": (
                None
                if diagnostics is None
                else (
                    diagnostics.summary()
                    if callable(getattr(diagnostics, "summary", None))
                    else {"kind": type(diagnostics).__name__}
                )
            ),
        }


class ContactProjectionLifecycle:
    """Coordinate search/projection trials around accepted increments.

    ``projector`` must expose ``project(points, **options)``. Search backends
    may additionally expose ``project_with_diagnostics``; the lifecycle keeps
    that evidence without making it part of State. Exact projection is repeated
    for every call. Candidate warm-starting may be implemented inside a search
    backend, but this lifecycle never reuses stale signed gaps or normals.
    """

    def __init__(
        self,
        projector,
        point_ids,
        *,
        state: ContactProjectionState | None = None,
        require_all_valid: bool = True,
    ) -> None:
        if not callable(getattr(projector, "project", None)):
            raise TypeError("Contact projector must provide project().")
        self.projector = projector
        self.point_ids = _validated_point_ids(point_ids)
        self.state = ContactProjectionState() if state is None else state
        if not isinstance(self.state, ContactProjectionState):
            raise TypeError("Contact projection lifecycle requires its State owner.")
        if self.state.accepted is not None and not np.array_equal(
            self.state.accepted.point_ids,
            np.sort(self.point_ids),
        ):
            raise ValueError("Lifecycle contact point identity differs from State.")
        self.require_all_valid = bool(require_all_valid)
        self.projection_attempts = 0
        self.successful_projections = 0
        self.rejected_projections = 0
        self.accepted_increments = 0
        self.rejected_increments = 0
        self.last_evaluation: ContactProjectionEvaluation | None = None

    def evaluate(self, points, **projection_options) -> ContactProjectionEvaluation:
        """Project one nonlinear trial and replace State only on success."""

        self.projection_attempts += 1
        diagnostics = None
        try:
            with_diagnostics = getattr(
                self.projector,
                "project_with_diagnostics",
                None,
            )
            if callable(with_diagnostics):
                outcome = with_diagnostics(points, **projection_options)
                projection = getattr(outcome, "projection", None)
                diagnostics = getattr(outcome, "diagnostics", None)
            else:
                projection = self.projector.project(points, **projection_options)
            if not isinstance(projection, SurfaceProjection):
                raise TypeError("Contact projector returned no SurfaceProjection.")
        except Exception:
            self.state.rollback()
            self.last_evaluation = None
            self.rejected_projections += 1
            raise

        post_error: Exception | None = None
        record: ContactProjectionRecord | None = None
        try:
            if projection.point_count != self.point_ids.size:
                raise ValueError(
                    "Projected point count differs from the lifecycle identity."
                )
            if self.require_all_valid and not projection.all_valid:
                counts = projection.summary()["status_counts"]
                raise ValueError(
                    "Contact projection contains invalid points: "
                    f"status_counts={counts}."
                )
            record = self.state.begin(self.point_ids, projection)
        except (TypeError, ValueError) as exc:
            post_error = exc

        communicator = getattr(self.projector, "communicator", None)
        if communicator is not None:
            messages = tuple(
                communicator.allgather(
                    None
                    if post_error is None
                    else f"{type(post_error).__name__}: {post_error}"
                )
            )
            failures = tuple(
                f"rank {rank}: {message}"
                for rank, message in enumerate(messages)
                if message is not None
            )
            if failures:
                self.state.rollback()
                self.last_evaluation = None
                self.rejected_projections += 1
                raise ValueError(
                    "Contact projection rejected collectively; " + "; ".join(failures)
                )
        elif post_error is not None:
            self.state.rollback()
            self.last_evaluation = None
            self.rejected_projections += 1
            raise post_error

        if record is None:  # pragma: no cover - collective consensus guards this
            raise RuntimeError("Contact projection consensus produced no trial record.")
        self.successful_projections += 1
        evaluation = ContactProjectionEvaluation(
            attempt=self.projection_attempts,
            record=record,
            diagnostics=diagnostics,
        )
        self.last_evaluation = evaluation
        return evaluation

    def commit_increment(self) -> ContactProjectionRecord:
        """Accept the most recent exact projection with its owning increment."""

        self.state.commit()
        self.accepted_increments += 1
        accepted = self.state.accepted
        if accepted is None:  # pragma: no cover - guarded by State.commit()
            raise RuntimeError("Contact projection commit produced no accepted state.")
        return accepted

    def rollback_increment(self) -> None:
        """Discard any trial projection while preserving accepted evidence."""

        self.state.rollback()
        self.rejected_increments += 1

    def summary(self) -> dict[str, object]:
        projector_summary = getattr(self.projector, "summary", None)
        return {
            "kind": "contact_projection_lifecycle",
            "point_count": int(self.point_ids.size),
            "projection_update": "every_evaluation",
            "candidate_warm_start": False,
            "require_all_valid": self.require_all_valid,
            "projection_attempts": self.projection_attempts,
            "successful_projections": self.successful_projections,
            "rejected_projections": self.rejected_projections,
            "accepted_increments": self.accepted_increments,
            "rejected_increments": self.rejected_increments,
            "projector": (
                projector_summary()
                if callable(projector_summary)
                else {"kind": type(self.projector).__name__}
            ),
            "state": self.state.summary(),
        }


__all__ = ["ContactProjectionEvaluation", "ContactProjectionLifecycle"]
