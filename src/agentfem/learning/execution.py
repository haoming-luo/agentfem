"""Framework-neutral execution boundary for user-owned neural-field solvers.

The core deliberately does not define a training framework or neural-network
base class.  A user implementation receives one immutable scientific request
and returns an ordinary :class:`agentfem.results.SimulationResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .. import results
from .core import NeuralFieldSpec


@dataclass(frozen=True)
class NeuralFieldExecutionRequest:
    """Immutable input supplied to a user- or package-owned executor."""

    specification: NeuralFieldSpec
    model: object
    analysis: str
    name: str
    output_directory: Path | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.specification, NeuralFieldSpec):
            raise TypeError(
                "NeuralFieldExecutionRequest.specification must be a "
                "learning.NeuralFieldSpec."
            )
        selected_name = str(self.name).strip()
        if not selected_name:
            raise ValueError("NeuralFieldExecutionRequest.name must be non-empty.")
        selected_output = self.output_directory
        if selected_output is not None:
            selected_output = Path(selected_output).expanduser().resolve()
        selected_options = dict(self.options)
        if any(not isinstance(key, str) or not key for key in selected_options):
            raise TypeError("Executor option names must be non-empty strings.")
        object.__setattr__(self, "name", selected_name)
        object.__setattr__(self, "output_directory", selected_output)
        object.__setattr__(self, "options", MappingProxyType(selected_options))

    def option(self, name: str, default=None):
        """Read one executor-owned option without exposing mutable state."""

        return self.options.get(name, default)

    @property
    def comm(self):
        """Return the model communicator when a mesh-backed model owns one."""

        domain = getattr(self.model, "domain", None)
        return getattr(domain, "comm", None)

    @property
    def is_root(self) -> bool:
        """Return whether this process owns scalar result-manifest output."""

        communicator = self.comm
        return communicator is None or int(getattr(communicator, "rank", 0)) == 0

    def summary(self) -> dict[str, object]:
        """Return a JSON-shaped scientific summary without the live model."""

        return {
            "kind": "neural_field_execution_request",
            "analysis": self.analysis,
            "name": self.name,
            "model": getattr(self.model, "name", type(self.model).__name__),
            # Do not leak a workstation path into portable result evidence.
            "managed_output": self.output_directory is not None,
            "distributed": (
                self.comm is not None
                and int(getattr(self.comm, "size", 1)) > 1
            ),
            "option_names": tuple(sorted(self.options)),
            "specification": self.specification.summary(),
        }


class CallableNeuralFieldStep:
    """Execute one neural-field specification through a user-owned callable.

    No inheritance is required. ``executor`` may be a callable accepting one
    :class:`NeuralFieldExecutionRequest`, or an object exposing
    ``solve(request)``. The executor retains full ownership of its framework,
    architecture, optimization, and artifacts, but must return AgentFEM's
    common scientific result.
    """

    def __init__(
        self,
        request: NeuralFieldExecutionRequest,
        executor,
        *,
        executor_name: str,
        executor_version: str | None = None,
    ) -> None:
        if not _is_executor(executor):
            raise TypeError(
                "A neural-field executor must be callable or expose solve(request)."
            )
        self.request = request
        self.executor = executor
        self.executor_name = _non_empty(executor_name, "executor_name")
        self.executor_version = (
            None
            if executor_version is None
            else _non_empty(executor_version, "executor_version")
        )
        self.name = request.name
        self.step_number = 0
        self.execution_context = None
        self.last_result = None

    @property
    def specification(self) -> NeuralFieldSpec:
        return self.request.specification

    def solve(self):
        """Execute once and return the standard scientific result."""

        return self.solve_result()

    def solve_result(self, *, name: str | None = None):
        """Run the executor, bind provenance, and write the common manifest."""

        if self.last_result is not None:
            return self.last_result
        if name is not None and str(name).strip() != self.request.name:
            raise ValueError(
                "The execution request name is immutable. Set name= in model.step()."
            )
        self._prepare_output()
        solve = getattr(self.executor, "solve", None)
        produced = solve(self.request) if callable(solve) else self.executor(self.request)
        if not isinstance(produced, results.SimulationResult):
            raise TypeError(
                "A neural-field executor must return results.SimulationResult; "
                f"received {type(produced).__name__}."
            )
        provenance = {
            "contract": "agentfem.learning.neural_field_executor",
            "executor": {
                "name": self.executor_name,
                "version": self.executor_version,
            },
            "request": self.request.summary(),
        }
        existing = produced.metadata.get("learning_execution")
        if existing is not None and existing != provenance:
            raise ValueError(
                "SimulationResult.metadata['learning_execution'] is reserved for "
                "the AgentFEM execution boundary."
            )
        produced.metadata["learning_execution"] = provenance
        if self.request.output_directory is not None and self.request.is_root:
            produced.write_manifest(
                self.request.output_directory / "result.json",
                include_histories=True,
            )
        self.last_result = produced
        return produced

    def _prepare_output(self) -> None:
        output = self.request.output_directory
        if output is None:
            return
        if self.request.is_root:
            output.mkdir(parents=True, exist_ok=True)
        communicator = self.request.comm
        if (
            communicator is not None
            and int(getattr(communicator, "size", 1)) > 1
        ):
            communicator.barrier()


def executor_identity(
    executor,
    *,
    name: str | None = None,
    version: str | None = None,
) -> tuple[str, str | None]:
    """Return stable, serializable identity without retaining executable code."""

    if not _is_executor(executor):
        raise TypeError(
            "A neural-field executor must be callable or expose solve(request)."
        )
    selected = name or getattr(executor, "executor_name", None)
    if selected is None:
        candidate = executor if hasattr(executor, "__qualname__") else type(executor)
        module = getattr(candidate, "__module__", "")
        qualified = getattr(candidate, "__qualname__", type(executor).__name__)
        selected = f"{module}.{qualified}" if module else qualified
    selected_version = version
    if selected_version is None:
        selected_version = getattr(executor, "executor_version", None)
    return (
        _non_empty(selected, "executor_name"),
        None
        if selected_version is None
        else _non_empty(selected_version, "executor_version"),
    )


def _is_executor(candidate) -> bool:
    return callable(candidate) or callable(getattr(candidate, "solve", None))


def _non_empty(value, label: str) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError(f"{label} must be non-empty.")
    return selected


__all__ = [
    "NeuralFieldExecutionRequest",
]
