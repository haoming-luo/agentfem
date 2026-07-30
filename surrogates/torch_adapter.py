"""Optional, lazily imported PyTorch MLP template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..datasets import ScientificDataset
from .base import Prediction, validate_predictions
from .linear import _Schema, _require_compatible_schema, _standardize


@dataclass(frozen=True)
class TorchMLPSurrogate:
    """Configurable dense-network baseline for parameter-to-QoI learning.

    PyTorch is optional and imported only by :meth:`fit`. Field-to-field
    neural operators use the separate, stricter ``NeuralOperatorSpec``.
    """

    hidden_layers: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    verbose_every: int | None = None

    def __post_init__(self) -> None:
        if not self.hidden_layers or any(width <= 0 for width in self.hidden_layers):
            raise ValueError("TorchMLPSurrogate.hidden_layers must be positive.")
        if self.activation.lower() not in {"relu", "tanh", "gelu", "silu"}:
            raise ValueError("Supported activations are relu, tanh, gelu, and silu.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")

    def fit(self, dataset: ScientificDataset) -> "TrainedTorchMLP":
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "TorchMLPSurrogate requires optional dependency 'torch'. "
                "Install PyTorch in the active AgentFEM environment."
            ) from exc

        torch.manual_seed(self.seed)
        X = dataset.x_matrix(normalized=True)
        Y = dataset.y_matrix()
        x_mean, x_scale, Xs = _standardize(X, enabled=True)
        y_mean, y_scale, Ys = _standardize(Y, enabled=True)
        module = _build_module(
            torch,
            input_size=X.shape[1],
            output_size=Y.shape[1],
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        ).to(self.device)
        features = torch.as_tensor(Xs, dtype=torch.float32)
        targets = torch.as_tensor(Ys, dtype=torch.float32)
        data = torch.utils.data.TensorDataset(features, targets)
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=min(self.batch_size, len(data)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = torch.nn.MSELoss()
        history = []
        module.train()
        for epoch in range(self.epochs):
            total = 0.0
            count = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_function(module(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch_x)
                count += len(batch_x)
            epoch_loss = total / max(count, 1)
            history.append(epoch_loss)
            if self.verbose_every and (epoch + 1) % self.verbose_every == 0:
                print(f"TorchMLP epoch {epoch + 1}: loss={epoch_loss:.6e}")
        module.eval()
        with torch.no_grad():
            fitted = module(features.to(self.device)).cpu().numpy()
        residual_std = np.sqrt(np.mean((fitted - Ys) ** 2, axis=0)) * y_scale
        return TrainedTorchMLP(
            schema=_Schema.from_dataset(dataset),
            module=module,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            residual_std=residual_std,
            config=self,
            loss_history=tuple(history),
        )


@dataclass
class TrainedTorchMLP:
    """In-memory trained PyTorch adapter."""

    schema: _Schema
    module: object
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    residual_std: np.ndarray
    config: TorchMLPSurrogate
    loss_history: tuple[float, ...]
    kind: str = "torch_mlp"

    def predict_matrix(self, values) -> np.ndarray:
        import torch

        X = self.schema.encode_inputs(values)
        Xs = (X - self.x_mean) / self.x_scale
        self.module.eval()
        with torch.no_grad():
            prediction = (
                self.module(
                    torch.as_tensor(
                        Xs,
                        dtype=torch.float32,
                        device=self.config.device,
                    )
                )
                .cpu()
                .numpy()
            )
        return prediction * self.y_scale + self.y_mean

    def predict(self, values):
        decoded = self.schema.decode(self.predict_matrix(values))
        return decoded[0] if isinstance(values, Mapping) else decoded

    def predict_with_uncertainty(self, values):
        rows = self.predict_matrix(values)
        outputs = self.schema.decode(rows)
        uncertainties = self.schema.decode(
            np.repeat(self.residual_std.reshape(1, -1), len(rows), axis=0)
        )
        predictions = [
            Prediction(
                outputs=output,
                uncertainty=uncertainty,
                source=self.kind,
                diagnostics={
                    "uncertainty_kind": "training_residual_scale",
                    "epistemic_uncertainty": False,
                    "training_loss": self.loss_history[-1],
                },
            )
            for output, uncertainty in zip(outputs, uncertainties, strict=True)
        ]
        return predictions[0] if isinstance(values, Mapping) else predictions

    def validate(self, dataset: ScientificDataset, *, thresholds=None):
        _require_compatible_schema(self.schema, dataset)
        return validate_predictions(
            model_kind=self.kind,
            dataset=dataset,
            predictions=self.predict_matrix(
                [sample.inputs for sample in dataset.samples]
            ),
            thresholds=thresholds,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "status": "trained_in_memory",
            "hidden_layers": self.config.hidden_layers,
            "activation": self.config.activation,
            "epochs": self.config.epochs,
            "final_training_loss": self.loss_history[-1],
            **self.schema.summary(),
        }


def _build_module(torch, *, input_size, output_size, hidden_layers, activation):
    activation_types = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
    }
    layers = []
    previous = input_size
    for width in hidden_layers:
        layers.extend((torch.nn.Linear(previous, width), activation_types[activation.lower()]()))
        previous = width
    layers.append(torch.nn.Linear(previous, output_size))
    return torch.nn.Sequential(*layers)


__all__ = ["TorchMLPSurrogate", "TrainedTorchMLP"]
