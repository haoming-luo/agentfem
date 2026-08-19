"""Material-library helpers for AgentFEM."""

from .library import (
    MaterialRecord,
    list_material_models,
    list_materials,
    load_material,
    material_record,
    register_material,
)
from .properties import (
    ElasticAnisotropic2DProperties,
    ElasticIsotropicProperties,
    ThermoElasticIsotropicProperties,
)
from .schemas import validate_material_record

__all__ = [
    "ElasticAnisotropic2DProperties",
    "ElasticIsotropicProperties",
    "ThermoElasticIsotropicProperties",
    "MaterialRecord",
    "list_material_models",
    "list_materials",
    "load_material",
    "material_record",
    "register_material",
    "validate_material_record",
]
