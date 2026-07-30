"""Structured validation results for human and agent repair workflows.

Validation is intentionally separate from numerical execution.  A report can
be inspected, serialized, or turned into an exception before any form is
assembled.  Issue paths address scientific objects rather than Python stack
frames so an agent can revise the smallest relevant part of a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal


Severity = Literal["error", "warning", "info"]
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class ValidationIssue:
    """One addressable model, numerical, or execution issue."""

    code: str
    path: str
    message: str
    severity: Severity = "error"
    hint: str | None = None
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_ORDER:
            raise ValueError(
                f"Unknown validation severity {self.severity!r}; "
                "expected 'error', 'warning', or 'info'."
            )
        if not self.code:
            raise ValueError("ValidationIssue.code must not be empty.")
        if not self.path:
            raise ValueError("ValidationIssue.path must not be empty.")
        if not self.message:
            raise ValueError("ValidationIssue.message must not be empty.")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe issue record."""

        result: dict[str, object] = {
            "code": self.code,
            "path": self.path,
            "severity": self.severity,
            "message": self.message,
        }
        if self.hint is not None:
            result["hint"] = self.hint
        if self.context:
            result["context"] = dict(self.context)
        return result

    def format(self) -> str:
        """Return a compact human-readable issue line."""

        text = f"[{self.severity.upper()} {self.code}] {self.path}: {self.message}"
        if self.hint:
            text += f" Hint: {self.hint}"
        return text


@dataclass(frozen=True)
class ValidationReport:
    """Immutable collection of structured validation issues."""

    issues: tuple[ValidationIssue, ...] = ()
    scope: str = "model"

    @classmethod
    def from_issues(
        cls,
        issues: Iterable[ValidationIssue],
        *,
        scope: str = "model",
    ) -> "ValidationReport":
        """Build a deterministically ordered report."""

        ordered = sorted(
            tuple(issues),
            key=lambda item: (
                _SEVERITY_ORDER[item.severity],
                item.path,
                item.code,
            ),
        )
        return cls(issues=tuple(ordered), scope=scope)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def infos(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "info")

    @property
    def is_valid(self) -> bool:
        """Return true when the report contains no errors."""

        return not self.errors

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe validation record."""

        return {
            "kind": "validation_report",
            "scope": self.scope,
            "valid": self.is_valid,
            "counts": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.infos),
            },
            "issues": tuple(issue.as_dict() for issue in self.issues),
        }

    def summary(self) -> dict[str, object]:
        """Compatibility alias for model and agent inspection."""

        return self.as_dict()

    def format(self) -> str:
        """Return a multiline report for terminals and notebooks."""

        if not self.issues:
            return f"ValidationReport(scope={self.scope!r}): valid"
        header = (
            f"ValidationReport(scope={self.scope!r}): "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        return "\n".join((header, *(issue.format() for issue in self.issues)))

    def raise_if_errors(self) -> None:
        """Raise ``ModelValidationError`` when errors are present."""

        if self.errors:
            raise ModelValidationError(self)


class ModelValidationError(ValueError):
    """Raised when a structured model validation report contains errors."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.format())


def issue(
    code: str,
    path: str,
    message: str,
    *,
    severity: Severity = "error",
    hint: str | None = None,
    **context,
) -> ValidationIssue:
    """Concise constructor used by model validators and backend adapters."""

    return ValidationIssue(
        code=code,
        path=path,
        message=message,
        severity=severity,
        hint=hint,
        context=context,
    )


__all__ = [
    "ModelValidationError",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "issue",
]
