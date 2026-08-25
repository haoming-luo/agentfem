"""Material-library helpers for AgentFEM."""

from .assets import MaterialAssetError, load, load_python

from .definitions import (
    MaterialBehavior,
    MaterialCompatibility,
    MaterialDefinition,
    define,
)

from .library import (
    MaterialRecord,
    list_material_models,
    list_materials,
    load_material,
    load_definition,
    material_record,
    register_material,
)
from .properties import (
    ElasticAnisotropic2DProperties,
    ElasticIsotropicProperties,
    ThermoElasticIsotropicProperties,
    TemperatureDependentThermoElasticProperties,
    TemperaturePropertyTable,
    temperature_property,
)
from .schemas import validate_material_record

__all__ = [
    "ElasticAnisotropic2DProperties",
    "ElasticIsotropicProperties",
    "ThermoElasticIsotropicProperties",
    "TemperatureDependentThermoElasticProperties",
    "TemperaturePropertyTable",
    "temperature_property",
    "MaterialRecord",
    "MaterialAssetError",
    "MaterialBehavior",
    "MaterialCompatibility",
    "MaterialDefinition",
    "define",
    "list_material_models",
    "list_materials",
    "load",
    "load_python",
    "load_material",
    "load_definition",
    "material_record",
    "register_material",
    "validate_material_record",
]
