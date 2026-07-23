"""Lightweight validation for material-centered library records."""

from __future__ import annotations

import numpy as np


SUPPORTED_MODELS = {
    "isotropic_linear_elastic",
    "anisotropic_linear_elastic_2d",
    "orthotropic_plane_stress_2d",
}


def validate_material_record(name: str, record: dict) -> None:
    """Validate one material-centered library record.

    A record describes one material entity. Its ``models`` mapping contains one
    or more constitutive/transport model parameter sets.
    """

    if not isinstance(record, dict):
        raise TypeError(f"material {name!r} must be a dictionary.")
    material_id = record.get("id")
    if material_id != name:
        raise ValueError(f"material file {name!r} must declare id={name!r}.")
    if record.get("unit_system", "SI") != "SI":
        raise ValueError(f"material {name!r} must use unit_system='SI'.")
    if not record.get("source"):
        raise ValueError(f"material {name!r} must include a source note.")
    models = record.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"material {name!r} must define a nonempty models mapping.")
    for model_name, model_record in models.items():
        validate_material_model(name, model_name, model_record)


def validate_material_model(material_name: str, model: str, record: dict) -> None:
    """Validate one model entry inside a material record."""

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"material {material_name!r} has unsupported model {model!r}. "
            f"Supported models: {sorted(SUPPORTED_MODELS)}."
        )
    if not isinstance(record, dict):
        raise TypeError(f"material {material_name!r} model {model!r} must be a dictionary.")
    if model == "isotropic_linear_elastic":
        _require_positive(record, material_name, model, "young")
        _require_positive(record, material_name, model, "density")
        poisson = float(record["poisson"])
        if not (-1.0 < poisson < 0.5):
            raise ValueError(
                f"material {material_name!r} model {model!r} has invalid poisson={poisson}."
            )
        return
    if model == "anisotropic_linear_elastic_2d":
        _require_positive(record, material_name, model, "density")
        C = np.asarray(record.get("stiffness_voigt"), dtype=float)
        if C.shape != (3, 3):
            raise ValueError(
                f"material {material_name!r} model {model!r} stiffness_voigt must be 3x3."
            )
        return
    if model == "orthotropic_plane_stress_2d":
        for key in ("ex", "ey", "gxy", "density"):
            _require_positive(record, material_name, model, key)
        nuxy = float(record["nuxy"])
        if not (-1.0 < nuxy < 0.5):
            raise ValueError(
                f"material {material_name!r} model {model!r} has invalid nuxy={nuxy}."
            )


def _require_positive(record: dict, material_name: str, model: str, key: str) -> None:
    if key not in record:
        raise ValueError(
            f"material {material_name!r} model {model!r} is missing required field {key!r}."
        )
    value = float(record[key])
    if value <= 0.0:
        raise ValueError(
            f"material {material_name!r} model {model!r} field {key!r} must be positive."
        )
