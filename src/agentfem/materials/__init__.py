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
    ElasticAnisotropic3DProperties,
    ElasticIsotropicProperties,
    ThermoElasticIsotropicProperties,
    TemperatureDependentThermoElasticProperties,
    TemperaturePropertyTable,
    temperature_property,
)
from .orientations import (
    FiberFrame,
    MaterialFrame,
    OrientedMaterial,
    fiber_frame,
    material_frame,
    oriented,
)
from .sections import (
    LaminateResponse,
    LaminateSection,
    Ply,
    PlyPointResult,
    SectionPoint,
    laminate,
    laminate_from_abaqus_section,
    ply,
    transformed_reduced_stiffness,
)
from .schemas import validate_material_record

__all__ = [
    "ElasticAnisotropic2DProperties",
    "ElasticAnisotropic3DProperties",
    "ElasticIsotropicProperties",
    "ThermoElasticIsotropicProperties",
    "TemperatureDependentThermoElasticProperties",
    "TemperaturePropertyTable",
    "FiberFrame",
    "MaterialFrame",
    "OrientedMaterial",
    "LaminateResponse",
    "LaminateSection",
    "Ply",
    "PlyPointResult",
    "SectionPoint",
    "fiber_frame",
    "material_frame",
    "oriented",
    "laminate",
    "laminate_from_abaqus_section",
    "ply",
    "transformed_reduced_stiffness",
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
