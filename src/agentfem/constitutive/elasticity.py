"""Linear-elastic constitutive relations."""

from __future__ import annotations

import numpy as np
import ufl

from agentfem import fields as field_api
from agentfem.materials.properties import (
    ElasticAnisotropic2DProperties,
    ElasticIsotropicProperties,
    ThermoElasticIsotropicProperties,
    TemperatureDependentThermoElasticProperties,
)


def strain(displacement):
    """Small-strain tensor, ``sym(grad(u))``."""

    displacement = field_api.unwrap(displacement)
    return ufl.sym(ufl.grad(displacement))


def engineering_strain_voigt_2d(displacement):
    """2D engineering-strain Voigt vector: [eps_xx, eps_yy, gamma_xy]."""

    displacement = field_api.unwrap(displacement)
    eps = strain(displacement)
    return ufl.as_vector([eps[0, 0], eps[1, 1], 2.0 * eps[0, 1]])


def stress_voigt_to_tensor_2d(stress_voigt):
    """Convert [sig_xx, sig_yy, sig_xy] to a 2D symmetric stress tensor."""

    return ufl.as_tensor(
        [
            [stress_voigt[0], stress_voigt[2]],
            [stress_voigt[2], stress_voigt[1]],
        ]
    )


def isotropic_pressure_wave_speed(young: float, poisson: float, density: float) -> float:
    """Longitudinal wave speed for a 3D isotropic elastic solid."""

    numerator = (1.0 - poisson) * young
    denominator = (1.0 + poisson) * (1.0 - 2.0 * poisson)
    return float(np.sqrt((numerator / denominator) / density))


def isotropic_shear_wave_speed(young: float, poisson: float, density: float) -> float:
    """Shear wave speed for an isotropic elastic solid."""

    mu = young / (2.0 * (1.0 + poisson))
    return float(np.sqrt(mu / density))


def estimate_elastic_wave_speeds(material) -> tuple[float, float]:
    """Return approximate ``(pressure_speed, shear_speed)`` for a material.

    For anisotropic materials this is a conservative scalar estimate, not a
    direction-dependent Christoffel analysis.
    """

    if isinstance(material, ElasticIsotropicProperties):
        return material.pressure_wave_speed, material.shear_wave_speed
    if isinstance(material, ElasticAnisotropic2DProperties):
        return material.pressure_wave_speed, material.shear_wave_speed
    if hasattr(material, "pressure_wave_speed"):
        pressure = float(material.pressure_wave_speed)
        shear = float(getattr(material, "shear_wave_speed", pressure / np.sqrt(3.0)))
        return pressure, shear
    raise TypeError("material does not provide enough elastic data to estimate wave speeds.")


def isotropic_stress(displacement, properties: ElasticIsotropicProperties, *, study=None, temperature=None):
    """Small-strain isotropic stress, ``sigma(u)``.

    Without a study, this uses the classical isotropic relation in the
    displacement field's geometric dimension. With a 2D solid-mechanics study,
    the study assumption selects plane strain or plane stress.
    """

    displacement = field_api.unwrap(displacement)
    if study is not None:
        _require_elastic_study_supported(study)
        if study.dimension == 2 and study.assumption == "plane_stress":
            return isotropic_plane_stress_2d(displacement, properties, temperature=temperature)
        if study.dimension == 2 and study.assumption == "plane_strain":
            return isotropic_plane_strain_2d(displacement, properties, temperature=temperature)

    eps = strain(displacement)
    young, poisson = _elastic_coefficients(properties, temperature)
    lambda_ = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return (
        lambda_ * ufl.tr(eps) * ufl.Identity(len(displacement))
        + 2.0 * mu * eps
    )


def thermal_strain(temperature, properties, *, dimension: int):
    """Return isotropic free thermal strain ``alpha (T-T_ref) I``."""

    selected = field_api.unwrap(temperature)
    increment = selected - properties.reference_temperature
    alpha = _coefficient(properties, "thermal_expansion", selected)
    return alpha * increment * ufl.Identity(int(dimension))


def thermoelastic_stress(displacement, temperature, properties, *, study=None):
    """Small-strain isotropic stress including thermal eigenstrain.

    Plane strain retains the constrained out-of-plane thermal strain in the
    three-dimensional trace. Plane stress uses the reduced in-plane law.
    """

    displacement = field_api.unwrap(displacement)
    selected_temperature = field_api.unwrap(temperature)
    if study is not None:
        _require_elastic_study_supported(study)
    dimension = len(displacement)
    delta_temperature = selected_temperature - properties.reference_temperature
    young, poisson = _elastic_coefficients(properties, selected_temperature)
    lambda_ = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    alpha_delta = _coefficient(properties, "thermal_expansion", selected_temperature) * delta_temperature
    eps = strain(displacement)
    if dimension == 2 and getattr(study, "assumption", None) == "plane_stress":
        mechanical = eps - alpha_delta * ufl.Identity(2)
        plane_stress_lambda = young * poisson / (
            1.0 - poisson**2
        )
        return (
            plane_stress_lambda * ufl.tr(mechanical) * ufl.Identity(2)
            + 2.0 * mu * mechanical
        )
    if dimension == 2 and getattr(study, "assumption", None) == "plane_strain":
        return (
            lambda_
            * (ufl.tr(eps) - 3.0 * alpha_delta)
            * ufl.Identity(2)
            + 2.0 * mu * (eps - alpha_delta * ufl.Identity(2))
        )
    mechanical = eps - alpha_delta * ufl.Identity(dimension)
    return (
        lambda_ * ufl.tr(mechanical) * ufl.Identity(dimension)
        + 2.0 * mu * mechanical
    )


def thermal_expansion_stress(temperature, properties, *, study=None, dimension=None):
    """Return positive ``C:epsilon_thermal`` for an equivalent thermal load."""

    selected = field_api.unwrap(temperature)
    selected_dimension = int(
        dimension if dimension is not None else getattr(study, "dimension", 3)
    )
    young, poisson = _elastic_coefficients(properties, selected)
    lambda_ = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    alpha_delta = _coefficient(properties, "thermal_expansion", selected) * (
        selected - properties.reference_temperature
    )
    if selected_dimension == 2 and getattr(study, "assumption", None) == "plane_stress":
        plane_stress_lambda = young * poisson / (
            1.0 - poisson**2
        )
        factor = 2.0 * (mu + plane_stress_lambda)
    elif selected_dimension == 2 and getattr(study, "assumption", None) == "plane_strain":
        factor = 2.0 * mu + 3.0 * lambda_
    else:
        factor = 2.0 * mu + selected_dimension * lambda_
    return factor * alpha_delta * ufl.Identity(selected_dimension)


def isotropic_plane_strain_2d(displacement, properties: ElasticIsotropicProperties, *, temperature=None):
    """2D isotropic plane-strain stress."""

    displacement = field_api.unwrap(displacement)
    eps = strain(displacement)
    young, poisson = _elastic_coefficients(properties, temperature)
    lambda_ = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return lambda_ * ufl.tr(eps) * ufl.Identity(2) + 2.0 * mu * eps


def isotropic_plane_stress_2d(displacement, properties: ElasticIsotropicProperties, *, temperature=None):
    """2D isotropic plane-stress stress."""

    displacement = field_api.unwrap(displacement)
    eps = strain(displacement)
    young, poisson = _elastic_coefficients(properties, temperature)
    mu = young / (2.0 * (1.0 + poisson))
    plane_stress_lambda = young * poisson / (
        1.0 - poisson**2
    )
    return (
        plane_stress_lambda * ufl.tr(eps) * ufl.Identity(2)
        + 2.0 * mu * eps
    )


def anisotropic_stress_2d(displacement, properties: ElasticAnisotropic2DProperties, *, study=None):
    """2D anisotropic stress from engineering-strain Voigt stiffness."""

    displacement = field_api.unwrap(displacement)
    if study is not None:
        _require_2d_study_supported(study)
    strain_voigt = engineering_strain_voigt_2d(displacement)
    stress_voigt = ufl.dot(ufl.as_matrix(properties.stiffness_voigt.tolist()), strain_voigt)
    return stress_voigt_to_tensor_2d(stress_voigt)


def stress(displacement, properties, *, study=None, temperature=None):
    """Dispatch to the matching elastic stress relation."""

    if isinstance(properties, ElasticIsotropicProperties):
        return isotropic_stress(displacement, properties, study=study, temperature=temperature)
    if isinstance(properties, ElasticAnisotropic2DProperties):
        return anisotropic_stress_2d(displacement, properties, study=study)
    if hasattr(properties, "stiffness_voigt"):
        return anisotropic_stress_2d(displacement, properties, study=study)
    if hasattr(properties, "young") and hasattr(properties, "poisson"):
        return isotropic_stress(displacement, properties, study=study, temperature=temperature)
    raise TypeError(f"unsupported elastic properties object: {type(properties)!r}")


def _coefficient(properties, name: str, temperature=None):
    if hasattr(properties, "coefficient"):
        if temperature is None:
            raise ValueError(
                f"Temperature-dependent {name} requires a known temperature field."
            )
        return properties.coefficient(name, temperature)
    return getattr(properties, name)


def _elastic_coefficients(properties, temperature=None):
    return (
        _coefficient(properties, "young", temperature),
        _coefficient(properties, "poisson", temperature),
    )


def _require_elastic_study_supported(study) -> None:
    if hasattr(study, "require"):
        study.require(physics="solid_mechanics")
    if getattr(study, "dimension", None) == 2:
        _require_2d_study_supported(study)


def _require_2d_study_supported(study) -> None:
    assumption = getattr(study, "assumption", None)
    if assumption == "axisymmetric":
        raise NotImplementedError(
            "Axisymmetric elasticity requires radial strain and weighted "
            "integration; it is not implemented yet."
        )
    if assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError(
            "2D elasticity requires study.assumption='plane_stress' or "
            "'plane_strain'."
        )


def isotropic_elastic(
    *,
    young: float,
    density: float,
    poisson: float,
    name: str = "isotropic elastic",
) -> ElasticIsotropicProperties:
    """Create isotropic linear-elastic properties."""

    return ElasticIsotropicProperties(name=name, young=young, density=density, poisson=poisson)


def thermoelastic(
    *,
    young: float,
    density: float,
    poisson: float,
    thermal_expansion: float,
    conductivity: float,
    specific_heat: float,
    reference_temperature: float = 293.15,
    name: str = "isotropic thermoelastic",
) -> ThermoElasticIsotropicProperties:
    """Create one material record for sequential thermal-stress workflows."""

    return ThermoElasticIsotropicProperties(
        name=name,
        young=young,
        density=density,
        poisson=poisson,
        thermal_expansion=thermal_expansion,
        conductivity=conductivity,
        specific_heat=specific_heat,
        reference_temperature=reference_temperature,
    )


def temperature_dependent_thermoelastic(
    *,
    young,
    density: float,
    poisson,
    thermal_expansion,
    conductivity,
    specific_heat,
    reference_temperature: float = 293.15,
    name: str = "temperature-dependent isotropic thermoelastic",
) -> TemperatureDependentThermoElasticProperties:
    """Create tabulated properties for sequential thermo-mechanics."""

    return TemperatureDependentThermoElasticProperties(
        name=name,
        young=young,
        density=density,
        poisson=poisson,
        thermal_expansion=thermal_expansion,
        conductivity=conductivity,
        specific_heat=specific_heat,
        reference_temperature=reference_temperature,
    )


def anisotropic_elastic_2d(
    *,
    stiffness_voigt,
    density: float,
    name: str = "anisotropic elastic 2D",
) -> ElasticAnisotropic2DProperties:
    """Create 2D anisotropic linear-elastic properties."""

    return ElasticAnisotropic2DProperties(
        name=name,
        stiffness_voigt=np.asarray(stiffness_voigt, dtype=float),
        density=density,
    )


def orthotropic_plane_stress_2d(
    *,
    ex: float,
    ey: float,
    nuxy: float,
    gxy: float,
    density: float,
    name: str = "orthotropic plane-stress elastic 2D",
) -> ElasticAnisotropic2DProperties:
    """Create 2D orthotropic plane-stress elastic properties."""

    nuyx = nuxy * ey / ex
    denom = 1.0 - nuxy * nuyx
    C = np.array(
        [
            [ex / denom, nuyx * ex / denom, 0.0],
            [nuxy * ey / denom, ey / denom, 0.0],
            [0.0, 0.0, gxy],
        ],
        dtype=float,
    )
    return ElasticAnisotropic2DProperties(
        name=name,
        stiffness_voigt=C,
        density=density,
        model="orthotropic_plane_stress_2d",
    )


IsotropicElasticMaterial = ElasticIsotropicProperties
AnisotropicElasticMaterial2D = ElasticAnisotropic2DProperties
