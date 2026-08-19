"""Deterministic NumPy baselines for scalar and field surrogates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..datasets import Quantity, ScientificDataset, decode_quantities
from ..datasets.core import _parameter_space_from_summary, _quantity_from_summary
from ..ir.schema import to_json_safe
from .base import Prediction, validate_predictions


SURROGATE_SCHEMA = "agentfem.surrogate"
SURROGATE_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class _Schema:
    parameter_space: object
    quantities: tuple[Quantity, ...]
    training_case_ids: tuple[str, ...]

    @classmethod
    def from_dataset(cls, dataset: ScientificDataset):
        return cls(
            parameter_space=dataset.parameter_space,
            quantities=dataset.quantities,
            training_case_ids=tuple(sample.case_id for sample in dataset.samples),
        )

    def encode_inputs(self, values) -> np.ndarray:
        if isinstance(values, Mapping):
            return self.parameter_space.encode(values).reshape(1, -1)
        selected = tuple(values)
        if not selected:
            raise ValueError("Prediction requires at least one input sample.")
        return np.vstack(
            [self.parameter_space.encode(sample) for sample in selected]
        )

    def decode(self, rows: np.ndarray):
        return [decode_quantities(self.quantities, row) for row in rows]

    def summary(self) -> dict[str, object]:
        return {
            "parameter_space": self.parameter_space.summary(),
            "quantities": [quantity.summary() for quantity in self.quantities],
            "training_case_ids": self.training_case_ids,
        }


@dataclass(frozen=True)
class RidgeSurrogate:
    """Multi-output ridge regression baseline.

    This inexpensive baseline is intentionally built in: a neural model should
    have to outperform a transparent reference before adding complexity.
    """

    alpha: float = 1.0e-10
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.alpha < 0.0:
            raise ValueError("RidgeSurrogate.alpha must be non-negative.")

    def fit(self, dataset: ScientificDataset) -> "TrainedRidge":
        X = dataset.x_matrix(normalized=True)
        Y = dataset.y_matrix()
        x_mean, x_scale, Xs = _standardize(X, enabled=self.standardize)
        y_mean, y_scale, Ys = _standardize(Y, enabled=self.standardize)
        design = np.column_stack((np.ones(len(Xs)), Xs))
        penalty = np.eye(design.shape[1]) * float(self.alpha)
        penalty[0, 0] = 0.0
        coefficients = (
            np.linalg.lstsq(design, Ys, rcond=None)[0]
            if self.alpha == 0.0
            else np.linalg.solve(
                design.T @ design + penalty,
                design.T @ Ys,
            )
        )
        fitted = design @ coefficients
        residual_std = np.sqrt(np.mean((fitted - Ys) ** 2, axis=0)) * y_scale
        return TrainedRidge(
            schema=_Schema.from_dataset(dataset),
            coefficients=coefficients,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            residual_std=residual_std,
            alpha=float(self.alpha),
            standardize=self.standardize,
        )


@dataclass(frozen=True)
class TrainedRidge:
    """Fitted ridge surrogate with named prediction and validation methods."""

    schema: _Schema
    coefficients: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    residual_std: np.ndarray
    alpha: float
    standardize: bool
    kind: str = "ridge"

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("TrainedRidge.alpha must be finite and non-negative.")
        input_size = len(self.schema.parameter_space.feature_names)
        output_size = sum(quantity.size for quantity in self.schema.quantities)
        coefficients = _state_array(
            self.coefficients,
            name="coefficients",
            shape=(input_size + 1, output_size),
        )
        x_mean = _state_array(self.x_mean, name="x_mean", shape=(input_size,))
        x_scale = _state_array(self.x_scale, name="x_scale", shape=(input_size,))
        y_mean = _state_array(self.y_mean, name="y_mean", shape=(output_size,))
        y_scale = _state_array(self.y_scale, name="y_scale", shape=(output_size,))
        residual_std = _state_array(
            self.residual_std,
            name="residual_std",
            shape=(output_size,),
        )
        if np.any(x_scale <= 0.0) or np.any(y_scale <= 0.0):
            raise ValueError("TrainedRidge standardization scales must be positive.")
        if np.any(residual_std < 0.0):
            raise ValueError("TrainedRidge residual scales must be non-negative.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "x_mean", x_mean)
        object.__setattr__(self, "x_scale", x_scale)
        object.__setattr__(self, "y_mean", y_mean)
        object.__setattr__(self, "y_scale", y_scale)
        object.__setattr__(self, "residual_std", residual_std)

    def predict_matrix(self, values) -> np.ndarray:
        X = self.schema.encode_inputs(values)
        Xs = (X - self.x_mean) / self.x_scale
        design = np.column_stack((np.ones(len(Xs)), Xs))
        return (design @ self.coefficients) * self.y_scale + self.y_mean

    def predict(self, values):
        rows = self.predict_matrix(values)
        decoded = self.schema.decode(rows)
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
            "schema": SURROGATE_SCHEMA,
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "kind": self.kind,
            "alpha": self.alpha,
            "standardize": self.standardize,
            **self.schema.summary(),
        }

    def write(self, path: str | Path) -> Path:
        return _write_model(
            path,
            manifest=self.summary(),
            arrays={
                "coefficients": self.coefficients,
                "x_mean": self.x_mean,
                "x_scale": self.x_scale,
                "y_mean": self.y_mean,
                "y_scale": self.y_scale,
                "residual_std": self.residual_std,
            },
        )

    @classmethod
    def read(cls, path: str | Path) -> "TrainedRidge":
        manifest, arrays = _read_model(path, expected_kind="ridge")
        return cls(
            schema=_schema_from_manifest(manifest),
            coefficients=arrays["coefficients"],
            x_mean=arrays["x_mean"],
            x_scale=arrays["x_scale"],
            y_mean=arrays["y_mean"],
            y_scale=arrays["y_scale"],
            residual_std=arrays["residual_std"],
            alpha=float(manifest["alpha"]),
            standardize=bool(manifest["standardize"]),
        )


@dataclass(frozen=True)
class PODRidgeSurrogate:
    """Proper-orthogonal-decomposition outputs plus ridge latent dynamics."""

    max_modes: int | None = None
    energy: float = 0.999
    alpha: float = 1.0e-10
    standardize_inputs: bool = True

    def __post_init__(self) -> None:
        if self.max_modes is not None and self.max_modes <= 0:
            raise ValueError("PODRidgeSurrogate.max_modes must be positive.")
        if not 0.0 < self.energy <= 1.0:
            raise ValueError("PODRidgeSurrogate.energy must lie in (0, 1].")
        if self.alpha < 0.0:
            raise ValueError("PODRidgeSurrogate.alpha must be non-negative.")

    def fit(self, dataset: ScientificDataset) -> "TrainedPODRidge":
        X = dataset.x_matrix(normalized=True)
        Y = dataset.y_matrix()
        x_mean, x_scale, Xs = _standardize(X, enabled=self.standardize_inputs)
        output_mean = np.mean(Y, axis=0)
        centered = Y - output_mean
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        rank = _pod_rank(
            singular_values,
            energy=self.energy,
            max_modes=self.max_modes,
        )
        basis = right[:rank]
        latent = centered @ basis.T
        design = np.column_stack((np.ones(len(Xs)), Xs))
        penalty = np.eye(design.shape[1]) * float(self.alpha)
        penalty[0, 0] = 0.0
        coefficients = (
            np.linalg.lstsq(design, latent, rcond=None)[0]
            if self.alpha == 0.0
            else np.linalg.solve(
                design.T @ design + penalty,
                design.T @ latent,
            )
        )
        reconstruction = (design @ coefficients) @ basis + output_mean
        residual_std = np.sqrt(np.mean((reconstruction - Y) ** 2, axis=0))
        total_energy = float(np.sum(singular_values**2))
        retained_energy = (
            1.0
            if total_energy == 0.0
            else float(np.sum(singular_values[:rank] ** 2) / total_energy)
        )
        return TrainedPODRidge(
            schema=_Schema.from_dataset(dataset),
            basis=basis,
            output_mean=output_mean,
            latent_coefficients=coefficients,
            x_mean=x_mean,
            x_scale=x_scale,
            residual_std=residual_std,
            alpha=float(self.alpha),
            retained_energy=retained_energy,
            singular_values=singular_values,
        )


@dataclass(frozen=True)
class TrainedPODRidge:
    """Fitted POD-ridge field/curve surrogate."""

    schema: _Schema
    basis: np.ndarray
    output_mean: np.ndarray
    latent_coefficients: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    residual_std: np.ndarray
    alpha: float
    retained_energy: float
    singular_values: np.ndarray
    kind: str = "pod_ridge"

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("TrainedPODRidge.alpha must be finite and non-negative.")
        input_size = len(self.schema.parameter_space.feature_names)
        output_size = sum(quantity.size for quantity in self.schema.quantities)
        basis = np.asarray(self.basis, dtype=float)
        if basis.ndim != 2 or basis.shape[1] != output_size or basis.shape[0] == 0:
            raise ValueError(
                "TrainedPODRidge basis must have shape (positive_modes, output_size)."
            )
        mode_count = basis.shape[0]
        basis = _state_array(
            basis,
            name="basis",
            shape=(mode_count, output_size),
        )
        output_mean = _state_array(
            self.output_mean,
            name="output_mean",
            shape=(output_size,),
        )
        latent_coefficients = _state_array(
            self.latent_coefficients,
            name="latent_coefficients",
            shape=(input_size + 1, mode_count),
        )
        x_mean = _state_array(self.x_mean, name="x_mean", shape=(input_size,))
        x_scale = _state_array(self.x_scale, name="x_scale", shape=(input_size,))
        residual_std = _state_array(
            self.residual_std,
            name="residual_std",
            shape=(output_size,),
        )
        singular_values = _state_array(
            self.singular_values,
            name="singular_values",
        )
        if singular_values.ndim != 1 or singular_values.size < mode_count:
            raise ValueError(
                "TrainedPODRidge singular_values must contain every retained mode."
            )
        if np.any(x_scale <= 0.0):
            raise ValueError("TrainedPODRidge input scales must be positive.")
        if np.any(residual_std < 0.0) or np.any(singular_values < 0.0):
            raise ValueError("POD residual scales and singular values must be non-negative.")
        if not 0.0 <= self.retained_energy <= 1.0 + 1.0e-12:
            raise ValueError("POD retained_energy must lie in [0, 1].")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "output_mean", output_mean)
        object.__setattr__(self, "latent_coefficients", latent_coefficients)
        object.__setattr__(self, "x_mean", x_mean)
        object.__setattr__(self, "x_scale", x_scale)
        object.__setattr__(self, "residual_std", residual_std)
        object.__setattr__(self, "singular_values", singular_values)

    @property
    def mode_count(self) -> int:
        return int(self.basis.shape[0])

    def predict_matrix(self, values) -> np.ndarray:
        X = self.schema.encode_inputs(values)
        Xs = (X - self.x_mean) / self.x_scale
        design = np.column_stack((np.ones(len(Xs)), Xs))
        return (design @ self.latent_coefficients) @ self.basis + self.output_mean

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
                    "uncertainty_kind": "training_reconstruction_residual_scale",
                    "epistemic_uncertainty": False,
                    "pod_modes": self.mode_count,
                    "retained_energy": self.retained_energy,
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
            "schema": SURROGATE_SCHEMA,
            "schema_version": SURROGATE_SCHEMA_VERSION,
            "kind": self.kind,
            "alpha": self.alpha,
            "mode_count": self.mode_count,
            "retained_energy": self.retained_energy,
            **self.schema.summary(),
        }

    def write(self, path: str | Path) -> Path:
        return _write_model(
            path,
            manifest=self.summary(),
            arrays={
                "basis": self.basis,
                "output_mean": self.output_mean,
                "latent_coefficients": self.latent_coefficients,
                "x_mean": self.x_mean,
                "x_scale": self.x_scale,
                "residual_std": self.residual_std,
                "singular_values": self.singular_values,
            },
        )

    @classmethod
    def read(cls, path: str | Path) -> "TrainedPODRidge":
        manifest, arrays = _read_model(path, expected_kind="pod_ridge")
        return cls(
            schema=_schema_from_manifest(manifest),
            basis=arrays["basis"],
            output_mean=arrays["output_mean"],
            latent_coefficients=arrays["latent_coefficients"],
            x_mean=arrays["x_mean"],
            x_scale=arrays["x_scale"],
            residual_std=arrays["residual_std"],
            alpha=float(manifest["alpha"]),
            retained_energy=float(manifest["retained_energy"]),
            singular_values=arrays["singular_values"],
        )


def _standardize(values: np.ndarray, *, enabled: bool):
    if not enabled:
        mean = np.zeros(values.shape[1])
        scale = np.ones(values.shape[1])
        return mean, scale, values.copy()
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return mean, scale, (values - mean) / scale


def _state_array(values, *, name: str, shape=None) -> np.ndarray:
    selected = np.array(values, dtype=float, copy=True)
    if shape is not None and selected.shape != shape:
        raise ValueError(
            f"Surrogate state {name!r} requires shape {shape}, got {selected.shape}."
        )
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"Surrogate state {name!r} contains non-finite values.")
    selected.setflags(write=False)
    return selected


def _pod_rank(
    singular_values: np.ndarray,
    *,
    energy: float,
    max_modes: int | None,
) -> int:
    if singular_values.size == 0:
        return 1
    total = float(np.sum(singular_values**2))
    if total == 0.0:
        rank = 1
    else:
        cumulative = np.cumsum(singular_values**2) / total
        rank = int(np.searchsorted(cumulative, energy, side="left") + 1)
    if max_modes is not None:
        rank = min(rank, int(max_modes))
    return max(1, min(rank, singular_values.size))


def _require_compatible_schema(schema: _Schema, dataset: ScientificDataset) -> None:
    if schema.parameter_space.summary() != dataset.parameter_space.summary():
        raise ValueError("Dataset parameter space differs from the trained surrogate.")
    if [item.summary() for item in schema.quantities] != [
        item.summary() for item in dataset.quantities
    ]:
        raise ValueError("Dataset output schema differs from the trained surrogate.")


def _write_model(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "state.npz"
    manifest_path = output / "manifest.json"
    np.savez_compressed(arrays_path, **arrays)
    selected = {**manifest, "state": arrays_path.name}
    manifest_path.write_text(
        json.dumps(
            to_json_safe(selected),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _read_model(path: str | Path, *, expected_kind: str):
    location = Path(path)
    manifest_path = location if location.is_file() else location / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SURROGATE_SCHEMA:
        raise ValueError(f"Unsupported surrogate schema {manifest.get('schema')!r}.")
    if manifest.get("schema_version") != SURROGATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported surrogate version {manifest.get('schema_version')!r}."
        )
    if manifest.get("kind") != expected_kind:
        raise ValueError(
            f"Expected surrogate kind {expected_kind!r}, got {manifest.get('kind')!r}."
        )
    with np.load(manifest_path.parent / manifest["state"]) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return manifest, arrays


def _schema_from_manifest(manifest: Mapping[str, object]) -> _Schema:
    return _Schema(
        parameter_space=_parameter_space_from_summary(manifest["parameter_space"]),
        quantities=tuple(
            _quantity_from_summary(item) for item in manifest["quantities"]
        ),
        training_case_ids=tuple(manifest.get("training_case_ids", ())),
    )


__all__ = [
    "PODRidgeSurrogate",
    "RidgeSurrogate",
    "SURROGATE_SCHEMA",
    "SURROGATE_SCHEMA_VERSION",
    "TrainedPODRidge",
    "TrainedRidge",
]
