"""Validated, portable datasets for surrogate and reduced-order models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..campaigns.parameters import (
    ChoiceParameter,
    IntegerParameter,
    ParameterSpace,
    RealParameter,
)
from ..ir.schema import to_json_safe
from .schema import Quantity, Sample, decode_quantities


DATASET_SCHEMA = "agentfem.scientific-dataset"
DATASET_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class DatasetSplit:
    """Reproducible train/validation partition."""

    train: "ScientificDataset"
    validation: "ScientificDataset"
    seed: int
    validation_fraction: float


@dataclass(frozen=True)
class ScientificDataset:
    """A numeric dataset whose columns retain scientific meaning."""

    parameter_space: ParameterSpace
    quantities: tuple[Quantity, ...]
    samples: tuple[Sample, ...]
    name: str = "dataset"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        quantities = tuple(self.quantities)
        samples = tuple(self.samples)
        if not quantities:
            raise ValueError("ScientificDataset requires at least one output quantity.")
        if len({quantity.name for quantity in quantities}) != len(quantities):
            raise ValueError("ScientificDataset quantity names must be unique.")
        if not samples:
            raise ValueError("ScientificDataset requires at least one successful sample.")
        if len({sample.case_id for sample in samples}) != len(samples):
            raise ValueError("ScientificDataset case IDs must be unique.")
        expected_outputs = {quantity.name for quantity in quantities}
        for sample in samples:
            self.parameter_space.validate(sample.inputs)
            actual_outputs = set(sample.outputs)
            if actual_outputs != expected_outputs:
                raise ValueError(
                    f"Sample {sample.case_id!r} output keys differ from schema; "
                    f"missing={sorted(expected_outputs - actual_outputs)}, "
                    f"extra={sorted(actual_outputs - expected_outputs)}."
                )
            for quantity in quantities:
                quantity.validate(sample.outputs[quantity.name])
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "name", str(self.name).strip() or "dataset")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.parameter_space.names

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(quantity.name for quantity in self.quantities)

    @property
    def input_size(self) -> int:
        return len(self.parameter_space.feature_names)

    @property
    def output_size(self) -> int:
        return sum(quantity.size for quantity in self.quantities)

    def x_matrix(self, *, normalized: bool = True) -> np.ndarray:
        """Return the design matrix in parameter order.

        Normalized coordinates are the portable default because categorical and
        log-scaled parameters do not share a meaningful raw numeric scale.
        """

        if normalized:
            return np.vstack(
                [self.parameter_space.encode(sample.inputs) for sample in self.samples]
            )
        rows = []
        for sample in self.samples:
            values = []
            for parameter in self.parameter_space.parameters:
                value = parameter.validate(sample.inputs[parameter.name])
                if isinstance(parameter, ChoiceParameter):
                    raise TypeError(
                        "Raw design matrices cannot encode categorical parameters; "
                        "use normalized=True."
                    )
                values.append(float(value))
            rows.append(values)
        return np.asarray(rows, dtype=float)

    def y_matrix(self) -> np.ndarray:
        """Return flattened outputs in declared quantity order."""

        rows = []
        for sample in self.samples:
            rows.append(
                np.concatenate(
                    [
                        quantity.validate(sample.outputs[quantity.name]).reshape(-1)
                        for quantity in self.quantities
                    ]
                )
            )
        return np.asarray(rows, dtype=float)

    def to_torch(self, **kwargs):
        """Create the optional PyTorch tensor bundle for this dataset."""

        from .torch import to_torch

        return to_torch(self, **kwargs)

    def decode_outputs(self, row: Sequence[float]) -> dict[str, object]:
        """Restore one flattened prediction to named scalar/array quantities."""

        return decode_quantities(self.quantities, row)

    def subset(self, indices: Iterable[int], *, name: str | None = None):
        selected = tuple(self.samples[int(index)] for index in indices)
        return ScientificDataset(
            parameter_space=self.parameter_space,
            quantities=self.quantities,
            samples=selected,
            name=name or self.name,
            metadata=self.metadata,
        )

    def split(
        self,
        *,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> DatasetSplit:
        """Create a deterministic random train/validation split."""

        fraction = float(validation_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1.")
        if len(self.samples) < 2:
            raise ValueError("At least two samples are required for a split.")
        validation_count = min(
            len(self.samples) - 1,
            max(1, int(round(len(self.samples) * fraction))),
        )
        order = np.random.default_rng(seed).permutation(len(self.samples))
        validation_indices = order[:validation_count]
        train_indices = order[validation_count:]
        return DatasetSplit(
            train=self.subset(train_indices, name=f"{self.name}_train"),
            validation=self.subset(
                validation_indices,
                name=f"{self.name}_validation",
            ),
            seed=int(seed),
            validation_fraction=fraction,
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": DATASET_SCHEMA,
            "schema_version": DATASET_SCHEMA_VERSION,
            "name": self.name,
            "sample_count": len(self.samples),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "parameter_space": self.parameter_space.summary(),
            "quantities": [quantity.summary() for quantity in self.quantities],
            "metadata": dict(self.metadata or {}),
        }

    def write(self, path: str | Path) -> Path:
        """Write a manifest plus compact numeric arrays to a directory."""

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        arrays_path = output / "arrays.npz"
        manifest_path = output / "manifest.json"
        np.savez_compressed(
            arrays_path,
            X=self.x_matrix(normalized=True),
            Y=self.y_matrix(),
        )
        manifest = {
            **self.summary(),
            "input_encoding": "normalized_numeric_plus_one_hot_choices",
            "arrays": arrays_path.name,
            "samples": [
                {
                    "case_id": sample.case_id,
                    "provenance": sample.provenance,
                    "artifacts": sample.artifacts,
                }
                for sample in self.samples
            ],
        }
        manifest_path.write_text(
            json.dumps(
                to_json_safe(manifest),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def read(cls, path: str | Path):
        """Load a dataset written by :meth:`write` with strict version checks."""

        location = Path(path)
        manifest_path = location if location.is_file() else location / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != DATASET_SCHEMA:
            raise ValueError(f"Unsupported dataset schema {manifest.get('schema')!r}.")
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported dataset schema version "
                f"{manifest.get('schema_version')!r}; expected {DATASET_SCHEMA_VERSION!r}."
            )
        if (
            manifest.get("input_encoding")
            != "normalized_numeric_plus_one_hot_choices"
        ):
            raise ValueError(
                f"Unsupported dataset input encoding "
                f"{manifest.get('input_encoding')!r}."
            )
        space = _parameter_space_from_summary(manifest["parameter_space"])
        quantities = tuple(_quantity_from_summary(item) for item in manifest["quantities"])
        with np.load(manifest_path.parent / manifest["arrays"]) as arrays:
            X = np.asarray(arrays["X"], dtype=float)
            Y = np.asarray(arrays["Y"], dtype=float)
        records = manifest["samples"]
        if X.shape[0] != len(records) or Y.shape[0] != len(records):
            raise ValueError("Dataset manifest and array sample counts disagree.")
        samples = []
        for index, record in enumerate(records):
            samples.append(
                Sample(
                    case_id=record["case_id"],
                    inputs=space.decode(X[index]),
                    outputs=decode_quantities(quantities, Y[index]),
                    provenance=record.get("provenance", {}),
                    artifacts=record.get("artifacts", {}),
                )
            )
        return cls(
            parameter_space=space,
            quantities=quantities,
            samples=tuple(samples),
            name=manifest.get("name", "dataset"),
            metadata=manifest.get("metadata", {}),
        )


def _parameter_space_from_summary(record: Mapping[str, object]) -> ParameterSpace:
    parameters = []
    for item in record["parameters"]:
        kind = item["kind"]
        if kind == "real":
            parameters.append(
                RealParameter(
                    name=item["name"],
                    lower=item["lower"],
                    upper=item["upper"],
                    unit=item.get("unit"),
                    description=item.get("description", ""),
                    scale=item.get("scale", "linear"),
                    nominal=item.get("nominal"),
                )
            )
        elif kind == "integer":
            parameters.append(
                IntegerParameter(
                    name=item["name"],
                    lower=item["lower"],
                    upper=item["upper"],
                    unit=item.get("unit"),
                    description=item.get("description", ""),
                    nominal=item.get("nominal"),
                )
            )
        elif kind == "choice":
            parameters.append(
                ChoiceParameter(
                    name=item["name"],
                    choices=tuple(item["choices"]),
                    description=item.get("description", ""),
                    nominal=item.get("nominal"),
                )
            )
        else:
            raise ValueError(f"Unknown parameter kind {kind!r}.")
    return ParameterSpace(
        parameters=tuple(parameters),
        name=record.get("name", "parameter_space"),
    )


def _quantity_from_summary(record: Mapping[str, object]) -> Quantity:
    return Quantity(
        name=record["name"],
        shape=tuple(record.get("shape", ())),
        unit=record.get("unit"),
        kind=record.get("kind", "quantity_of_interest"),
        description=record.get("description", ""),
        field_encoding=record.get("field_encoding"),
    )


__all__ = [
    "DATASET_SCHEMA",
    "DATASET_SCHEMA_VERSION",
    "DatasetSplit",
    "ScientificDataset",
]
