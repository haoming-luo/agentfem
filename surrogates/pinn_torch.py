"""Optional PyTorch execution adapter for explicit AgentFEM PINN contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PINNTrainingRecord:
    """In-memory training evidence without serializing a PyTorch pickle."""

    module: object
    losses: tuple[float, ...]
    spec_summary: Mapping[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "kind": "torch_pinn_training",
            "status": "trained_in_memory",
            "epochs": len(self.losses),
            "initial_loss": self.losses[0],
            "final_loss": self.losses[-1],
            "spec": dict(self.spec_summary),
        }


@dataclass(frozen=True)
class TorchPINNAdapter:
    """Bind explicit residual/condition callables to a :class:`PINNSpec`.

    Each residual callable receives ``(module, points)`` and returns its
    pointwise violation. Each condition callable uses the same signature.
    Callables normally use ``torch.autograd.grad``. AgentFEM validates the
    scientific names and weights; PyTorch owns autodiff and optimization.
    """

    spec: object
    residual_functions: Mapping[str, object]
    condition_functions: Mapping[str, object]

    def __post_init__(self) -> None:
        residual_names = {item.name for item in self.spec.residuals}
        condition_names = {item.name for item in self.spec.conditions}
        if set(self.residual_functions) != residual_names:
            raise ValueError(
                "PINN residual functions must exactly match the spec; "
                f"expected={sorted(residual_names)}."
            )
        if set(self.condition_functions) != condition_names:
            raise ValueError(
                "PINN condition functions must exactly match the spec; "
                f"expected={sorted(condition_names)}."
            )
        if any(not callable(value) for value in self.residual_functions.values()):
            raise TypeError("Every PINN residual implementation must be callable.")
        if any(not callable(value) for value in self.condition_functions.values()):
            raise TypeError("Every PINN condition implementation must be callable.")
        object.__setattr__(self, "residual_functions", dict(self.residual_functions))
        object.__setattr__(self, "condition_functions", dict(self.condition_functions))

    def loss(
        self,
        module,
        *,
        residual_points: Mapping[str, object],
        condition_points: Mapping[str, object],
    ):
        """Return total differentiable loss and named detached diagnostics."""

        torch = _torch()
        _require_batches(residual_points, self.residual_functions, "residual")
        _require_batches(condition_points, self.condition_functions, "condition")
        total = None
        diagnostics = {}
        for residual in self.spec.residuals:
            violation = self.residual_functions[residual.name](
                module,
                residual_points[residual.name],
            )
            term = float(residual.weight) * torch.mean(violation**2)
            total = term if total is None else total + term
            diagnostics[f"residual:{residual.name}"] = float(term.detach().cpu())
        for condition in self.spec.conditions:
            violation = self.condition_functions[condition.name](
                module,
                condition_points[condition.name],
            )
            term = float(condition.weight) * torch.mean(violation**2)
            total = term if total is None else total + term
            diagnostics[f"condition:{condition.name}"] = float(term.detach().cpu())
        if total is None:
            raise RuntimeError("PINN contract produced no loss terms.")
        diagnostics["total"] = float(total.detach().cpu())
        return total, diagnostics

    def train(
        self,
        module,
        *,
        residual_points: Mapping[str, object],
        condition_points: Mapping[str, object],
        epochs: int = 1000,
        learning_rate: float = 1.0e-3,
    ) -> PINNTrainingRecord:
        """Run a minimal Adam loop while retaining PyTorch model ownership."""

        torch = _torch()
        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("epochs and learning_rate must be positive.")
        optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
        losses = []
        module.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            loss, _ = self.loss(
                module,
                residual_points=residual_points,
                condition_points=condition_points,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        module.eval()
        return PINNTrainingRecord(module, tuple(losses), self.spec.summary())

    def summary(self) -> dict[str, object]:
        return {
            "kind": "torch_pinn_adapter",
            "status": "executable_for_bound_residuals",
            "residuals": tuple(self.residual_functions),
            "conditions": tuple(self.condition_functions),
            "automatic_ufl_translation": False,
            "spec": self.spec.summary(),
        }


def _require_batches(actual, implementations, kind: str) -> None:
    if set(actual) != set(implementations):
        raise ValueError(
            f"PINN {kind} point batches must match implementations; "
            f"expected={sorted(implementations)}."
        )


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "TorchPINNAdapter requires optional dependency 'torch'."
        ) from exc
    return torch


__all__ = ["PINNTrainingRecord", "TorchPINNAdapter"]
