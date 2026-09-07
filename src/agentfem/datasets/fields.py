"""Portable field collections for neural-operator and field-learning workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..ir.schema import to_json_safe


FIELD_DATASET_SCHEMA = "agentfem.scientific-field-dataset"
FIELD_DATASET_SCHEMA_VERSION = "0.1.0"


def _encoding_summary(encoding: object) -> dict[str, object]:
    if isinstance(encoding, Mapping):
        record = dict(encoding)
    else:
        summary = getattr(encoding, "summary", None)
        if not callable(summary):
            raise TypeError(
                "Field encodings must be mappings or expose a summary() method."
            )
        record = dict(summary())
    name = str(record.get("name", "")).strip()
    role = str(record.get("role", "")).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("Every field encoding requires a non-empty name.")
    if role not in {"input", "output", "condition", "coordinate"}:
        raise ValueError(
            f"Field encoding {name!r} has unsupported role {role!r}."
        )
    record["name"] = name
    record["role"] = role
    if record.get("shape") is not None:
        shape = tuple(int(value) for value in record["shape"])
        if any(value <= 0 for value in shape):
            raise ValueError(f"Field encoding {name!r} has an invalid shape.")
        record["shape"] = shape
    return record


def _numeric_array(name: str, value: object, *, leading_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[0] != leading_size:
        raise ValueError(
            f"Array {name!r} must have leading case dimension {leading_size}; "
            f"got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Array {name!r} must be numeric, got {array.dtype}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"Array {name!r} contains non-finite values.")
    return np.array(array, copy=True)


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FieldDatasetSplit:
    """Reproducible train/validation/test partition of field cases."""

    train: "ScientificFieldDataset"
    validation: "ScientificFieldDataset"
    test: "ScientificFieldDataset | None"
    seed: int
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class ScientificFieldDataset:
    """A scientific collection whose samples are complete physical fields.

    Every field array uses the leading axis for cases.  Its remaining axes are
    declared by a field encoding, so training libraries can change without
    changing the scientific meaning, units, geometry policy, or provenance.
    The reference on-disk format is compressed NumPy plus a JSON manifest;
    larger storage providers may implement the same public contract later.
    """

    case_ids: tuple[str, ...]
    encodings: tuple[object, ...]
    fields: Mapping[str, np.ndarray]
    coordinates: Mapping[str, np.ndarray] = field(default_factory=dict)
    parameters: Mapping[str, np.ndarray] = field(default_factory=dict)
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    case_metadata: tuple[Mapping[str, object], ...] = ()
    name: str = "field_dataset"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_ids = tuple(str(item).strip() for item in self.case_ids)
        if not case_ids or any(not item for item in case_ids):
            raise ValueError("ScientificFieldDataset requires non-empty case IDs.")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("ScientificFieldDataset case IDs must be unique.")

        encodings = tuple(_encoding_summary(item) for item in self.encodings)
        if not encodings:
            raise ValueError("ScientificFieldDataset requires field encodings.")
        encoding_names = tuple(str(item["name"]) for item in encodings)
        if len(set(encoding_names)) != len(encoding_names):
            raise ValueError("ScientificFieldDataset encoding names must be unique.")
        roles = {str(item["role"]) for item in encodings}
        if "output" not in roles:
            raise ValueError("ScientificFieldDataset requires at least one output field.")
        if not roles.intersection({"input", "condition", "coordinate"}):
            raise ValueError("ScientificFieldDataset requires at least one input field.")

        selected_fields = {
            str(key): _numeric_array(str(key), value, leading_size=len(case_ids))
            for key, value in self.fields.items()
        }
        if set(selected_fields) != set(encoding_names):
            raise ValueError(
                "Field arrays must match encoding names exactly; "
                f"missing={sorted(set(encoding_names) - set(selected_fields))}, "
                f"extra={sorted(set(selected_fields) - set(encoding_names))}."
            )
        for encoding in encodings:
            expected = encoding.get("shape")
            actual = selected_fields[str(encoding["name"])].shape[1:]
            if expected is not None and tuple(expected) != actual:
                raise ValueError(
                    f"Field {encoding['name']!r} declares per-case shape "
                    f"{tuple(expected)}, got {actual}."
                )

        coordinates = {
            str(key): _numeric_array(str(key), value, leading_size=len(case_ids))
            for key, value in self.coordinates.items()
        }
        parameters = {
            str(key): _numeric_array(str(key), value, leading_size=len(case_ids))
            for key, value in self.parameters.items()
        }
        masks: dict[str, np.ndarray] = {}
        for key, value in self.masks.items():
            name = str(key)
            if name not in selected_fields:
                raise ValueError(f"Mask {name!r} has no corresponding field.")
            mask = np.asarray(value, dtype=bool)
            field_shape = selected_fields[name].shape
            valid_shapes = {field_shape}
            if len(field_shape) >= 3:
                valid_shapes.add((field_shape[0], *field_shape[2:]))
            if mask.shape not in valid_shapes:
                component_free_shape = (
                    (field_shape[0], *field_shape[2:])
                    if len(field_shape) >= 3
                    else field_shape
                )
                raise ValueError(
                    f"Mask {name!r} shape {mask.shape} does not match field "
                    f"shape {field_shape} or its component-free shape "
                    f"{component_free_shape}."
                )
            masks[name] = np.array(mask, copy=True)

        records = tuple(dict(item) for item in self.case_metadata)
        if not records:
            records = tuple({} for _ in case_ids)
        if len(records) != len(case_ids):
            raise ValueError("case_metadata must contain one record per case.")

        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "encodings", encodings)
        object.__setattr__(self, "fields", selected_fields)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "case_metadata", records)
        object.__setattr__(self, "name", str(self.name).strip() or "field_dataset")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(
            str(item["name"])
            for item in self.encodings
            if item["role"] in {"input", "condition", "coordinate"}
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(
            str(item["name"])
            for item in self.encodings
            if item["role"] == "output"
        )

    def encoding(self, name: str) -> Mapping[str, object]:
        selected = str(name)
        for encoding in self.encodings:
            if encoding["name"] == selected:
                return dict(encoding)
        raise KeyError(selected)

    def subset(self, indices: Iterable[int], *, name: str | None = None):
        selected = np.asarray(tuple(int(item) for item in indices), dtype=int)
        if selected.size == 0:
            raise ValueError("A field-dataset subset must contain at least one case.")
        return ScientificFieldDataset(
            case_ids=tuple(self.case_ids[index] for index in selected),
            encodings=self.encodings,
            fields={key: value[selected] for key, value in self.fields.items()},
            coordinates={key: value[selected] for key, value in self.coordinates.items()},
            parameters={key: value[selected] for key, value in self.parameters.items()},
            masks={key: value[selected] for key, value in self.masks.items()},
            case_metadata=tuple(self.case_metadata[index] for index in selected),
            name=name or self.name,
            metadata=self.metadata,
        )

    def split(
        self,
        *,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.0,
        seed: int = 0,
    ) -> FieldDatasetSplit:
        """Create deterministic, disjoint train/validation/test partitions."""

        validation = float(validation_fraction)
        test = float(test_fraction)
        if validation <= 0.0 or test < 0.0 or validation + test >= 1.0:
            raise ValueError(
                "validation_fraction must be positive, test_fraction non-negative, "
                "and their sum must be below one."
            )
        minimum = 3 if test > 0.0 else 2
        if self.case_count < minimum:
            raise ValueError(f"At least {minimum} cases are required for this split.")
        validation_count = max(1, int(round(self.case_count * validation)))
        test_count = max(1, int(round(self.case_count * test))) if test > 0.0 else 0
        if validation_count + test_count >= self.case_count:
            raise ValueError("The requested split leaves no training cases.")
        order = np.random.default_rng(seed).permutation(self.case_count)
        validation_indices = order[:validation_count]
        test_indices = order[validation_count : validation_count + test_count]
        train_indices = order[validation_count + test_count :]
        return FieldDatasetSplit(
            train=self.subset(train_indices, name=f"{self.name}_train"),
            validation=self.subset(
                validation_indices, name=f"{self.name}_validation"
            ),
            test=(
                self.subset(test_indices, name=f"{self.name}_test")
                if test_count
                else None
            ),
            seed=int(seed),
            validation_fraction=validation,
            test_fraction=test,
        )

    @property
    def fingerprint(self) -> str:
        evidence = {
            "schema": FIELD_DATASET_SCHEMA,
            "schema_version": FIELD_DATASET_SCHEMA_VERSION,
            "case_ids": self.case_ids,
            "encodings": self.encodings,
            "fields": {
                key: _array_digest(value) for key, value in sorted(self.fields.items())
            },
            "coordinates": {
                key: _array_digest(value)
                for key, value in sorted(self.coordinates.items())
            },
            "parameters": {
                key: _array_digest(value)
                for key, value in sorted(self.parameters.items())
            },
            "masks": {
                key: _array_digest(value) for key, value in sorted(self.masks.items())
            },
            "case_metadata": self.case_metadata,
            "metadata": self.metadata,
        }
        payload = json.dumps(
            to_json_safe(evidence),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "schema": FIELD_DATASET_SCHEMA,
            "schema_version": FIELD_DATASET_SCHEMA_VERSION,
            "name": self.name,
            "case_count": self.case_count,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "encodings": [dict(item) for item in self.encodings],
            "field_arrays": {
                key: {"shape": value.shape, "dtype": str(value.dtype)}
                for key, value in sorted(self.fields.items())
            },
            "coordinate_arrays": {
                key: {"shape": value.shape, "dtype": str(value.dtype)}
                for key, value in sorted(self.coordinates.items())
            },
            "parameter_arrays": {
                key: {"shape": value.shape, "dtype": str(value.dtype)}
                for key, value in sorted(self.parameters.items())
            },
            "mask_arrays": {
                key: {"shape": value.shape, "dtype": str(value.dtype)}
                for key, value in sorted(self.masks.items())
            },
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }

    def write(self, path: str | Path) -> Path:
        """Write a portable manifest and lossless compressed arrays."""

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        arrays_path = output / "arrays.npz"
        manifest_path = output / "manifest.json"
        arrays: dict[str, np.ndarray] = {}
        storage: dict[str, dict[str, str]] = {
            "fields": {},
            "coordinates": {},
            "parameters": {},
            "masks": {},
        }
        groups = (
            ("fields", self.fields),
            ("coordinates", self.coordinates),
            ("parameters", self.parameters),
            ("masks", self.masks),
        )
        for group_name, group in groups:
            for index, (name, value) in enumerate(sorted(group.items())):
                storage_key = f"{group_name}_{index}"
                arrays[storage_key] = value
                storage[group_name][name] = storage_key
        np.savez_compressed(arrays_path, **arrays)
        manifest = {
            **self.summary(),
            "storage": {
                "format": "npz",
                "arrays": arrays_path.name,
                "keys": storage,
            },
            "case_ids": self.case_ids,
            "case_metadata": self.case_metadata,
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
        """Load a field dataset with strict schema and fingerprint checks."""

        location = Path(path)
        manifest_path = location if location.is_file() else location / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != FIELD_DATASET_SCHEMA:
            raise ValueError(
                f"Unsupported field-dataset schema {manifest.get('schema')!r}."
            )
        if manifest.get("schema_version") != FIELD_DATASET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported field-dataset schema version "
                f"{manifest.get('schema_version')!r}; expected "
                f"{FIELD_DATASET_SCHEMA_VERSION!r}."
            )
        storage = manifest.get("storage", {})
        if storage.get("format") != "npz":
            raise ValueError(f"Unsupported field storage {storage.get('format')!r}.")
        groups: dict[str, dict[str, np.ndarray]] = {
            "fields": {},
            "coordinates": {},
            "parameters": {},
            "masks": {},
        }
        with np.load(
            manifest_path.parent / str(storage["arrays"]), allow_pickle=False
        ) as saved:
            for group_name, keys in storage["keys"].items():
                if group_name not in groups:
                    raise ValueError(f"Unknown field storage group {group_name!r}.")
                groups[group_name] = {
                    str(name): np.asarray(saved[str(storage_key)])
                    for name, storage_key in keys.items()
                }
        dataset = cls(
            case_ids=tuple(manifest["case_ids"]),
            encodings=tuple(manifest["encodings"]),
            fields=groups["fields"],
            coordinates=groups["coordinates"],
            parameters=groups["parameters"],
            masks=groups["masks"],
            case_metadata=tuple(manifest.get("case_metadata", ())),
            name=manifest.get("name", "field_dataset"),
            metadata=manifest.get("metadata", {}),
        )
        if dataset.fingerprint != manifest.get("fingerprint"):
            raise ValueError("Field-dataset fingerprint does not match stored arrays.")
        return dataset


__all__ = [
    "FIELD_DATASET_SCHEMA",
    "FIELD_DATASET_SCHEMA_VERSION",
    "FieldDatasetSplit",
    "ScientificFieldDataset",
]
