"""Data-backed material library.

Each JSON file under ``materials/data`` describes one material entity. A
material can contain multiple model entries, such as isotropic elasticity,
anisotropic elasticity, thermal conduction, or future viscoelastic data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

from .schemas import validate_material_record


@dataclass(frozen=True)
class MaterialRecord:
    """Material-library record before conversion to a constitutive law."""

    name: str
    data: dict

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def display_name(self) -> str:
        return str(self.data.get("display_name", self.name))

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.data["models"]))

    def as_dict(self) -> dict:
        return dict(self.data)


_REGISTRY: dict[str, dict] | None = None


def list_materials(*, model: str | None = None) -> tuple[str, ...]:
    """List available material names, optionally filtered by model."""

    records = _records()
    names = sorted(records)
    if model is None:
        return tuple(names)
    return tuple(name for name in names if model in records[name].get("models", {}))


def list_material_models(name: str) -> tuple[str, ...]:
    """List model names available for one material."""

    return material_record(name).model_names


def material_record(name: str) -> MaterialRecord:
    """Return a validated material record without constructing a model object."""

    records = _records()
    if name not in records:
        raise KeyError(f"unknown material {name!r}. Available: {sorted(records)}.")
    record = dict(records[name])
    validate_material_record(name, record)
    return MaterialRecord(name=name, data=record)


def load_material(name: str, model: str | None = None):
    """Load one material model and return a constitutive material object.

    If ``model`` is omitted, the material must contain exactly one model entry.
    """

    record = material_record(name)
    model = _select_model(record, model)
    data = record.data["models"][model]
    if model == "isotropic_linear_elastic":
        from agentfem.constitutive import isotropic_elastic

        return isotropic_elastic(
            name=record.name,
            young=float(data["young"]),
            poisson=float(data["poisson"]),
            density=float(data["density"]),
        )
    if model == "anisotropic_linear_elastic_2d":
        from agentfem.constitutive import anisotropic_elastic_2d

        return anisotropic_elastic_2d(
            name=record.name,
            stiffness_voigt=np.asarray(data["stiffness_voigt"], dtype=float),
            density=float(data["density"]),
        )
    if model == "orthotropic_plane_stress_2d":
        from agentfem.constitutive import orthotropic_plane_stress_2d

        return orthotropic_plane_stress_2d(
            name=record.name,
            ex=float(data["ex"]),
            ey=float(data["ey"]),
            nuxy=float(data["nuxy"]),
            gxy=float(data["gxy"]),
            density=float(data["density"]),
        )
    raise ValueError(f"unsupported material model {model!r}.")


def register_material(name: str, data: dict, *, overwrite: bool = False) -> None:
    """Register or override a material record in memory.

    This does not write to disk. Edit one material-centered JSON file under
    ``materials/data`` for persistent library entries.
    """

    records = _records()
    if name in records and not overwrite:
        raise KeyError(f"material {name!r} already exists; pass overwrite=True.")
    validate_material_record(name, data)
    records[name] = dict(data)


def _records() -> dict[str, dict]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        for filename in _data_filenames():
            record = _read_json_data(filename)
            name = str(record.get("id", filename.removesuffix(".json")))
            if name in _REGISTRY:
                raise ValueError(f"duplicate material id {name!r}.")
            _REGISTRY[name] = record
        for name, record in _REGISTRY.items():
            validate_material_record(name, record)
    return _REGISTRY


def _data_filenames() -> tuple[str, ...]:
    data_dir = resources.files("agentfem.materials").joinpath("data")
    return tuple(sorted(path.name for path in data_dir.iterdir() if path.name.endswith(".json")))


def _read_json_data(filename: str) -> dict:
    data_path = resources.files("agentfem.materials").joinpath("data", filename)
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"material data file {filename!r} must contain an object.")
    return data


def _select_model(record: MaterialRecord, model: str | None) -> str:
    models = record.model_names
    if model is not None:
        if model not in models:
            raise KeyError(
                f"material {record.name!r} has no model {model!r}. "
                f"Available models: {models}."
            )
        return model
    if len(models) == 1:
        return models[0]
    raise ValueError(
        f"material {record.name!r} has multiple models; choose one explicitly. "
        f"Available models: {models}."
    )
