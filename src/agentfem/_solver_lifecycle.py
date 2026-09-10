"""Minimal internal contract for retained numerical solve allocations."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable


SolveValue = TypeVar("SolveValue", covariant=True)


@runtime_checkable
class PreparedSolve(Protocol[SolveValue]):
    """One reusable numerical allocation with an explicit terminal lifetime."""

    @property
    def closed(self) -> bool:
        """Whether owned backend resources have been released."""

        ...

    def solve(self) -> SolveValue:
        """Execute one solve while the allocation is open."""

        ...

    def summary(self) -> dict[str, object]:
        """Return evidence that remains readable after close."""

        ...

    def close(self) -> None:
        """Release owned resources exactly once."""

        ...


__all__ = ["PreparedSolve"]
