"""Linear-elastic constitutive relations."""

from __future__ import annotations

import numpy as np
import ufl

from agentfem import fields as field_api
from agentfem.materials.properties import (
    ElasticAnisotropic2DProperties,
    ElasticIsotropicProperties,
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


def isotropic_stress(displacement, properties: ElasticIsotropicProperties, *, study=None):
    """Small-strain isotropic stress, ``sigma(u)``.

    Without a study, this uses the classical isotropic relation in the
    displacement field's geometric dimension. With a 2D solid-mechanics study,
    the study assumption selects plane strain or plane stress.
    """

    displacement = field_api.unwrap(displacement)
    if study is not None:
        _require_elastic_study_supported(study)
        if study.dimension == 2 and study.assumption == "plane_stress":
            return isotropic_plane_stress_2d(displacement, properties)
        if study.dimension == 2 and study.assumption == "plane_strain":
            return isotropic_plane_strain_2d(displacement, properties)

    eps = strain(displacement)
    return (
        properties.lambda_ * ufl.tr(eps) * ufl.Identity(len(displacement))
        + 2.0 * properties.mu * eps
    )


def isotropic_plane_strain_2d(displacement, properties: ElasticIsotropicProperties):
    """2D isotropic plane-strain stress."""

    displacement = field_api.unwrap(displacement)
    eps = strain(displacement)
    return (
        properties.lambda_ * ufl.tr(eps) * ufl.Identity(2)
        + 2.0 * properties.mu * eps
    )


def isotropic_plane_stress_2d(displacement, properties: ElasticIsotropicProperties):
    """2D isotropic plane-stress stress."""

    displacement = field_api.unwrap(displacement)
    eps = strain(displacement)
    plane_stress_lambda = properties.young * properties.poisson / (
        1.0 - properties.poisson**2
    )
    return (
        plane_stress_lambda * ufl.tr(eps) * ufl.Identity(2)
        + 2.0 * properties.mu * eps
    )


def anisotropic_stress_2d(displacement, properties: ElasticAnisotropic2DProperties, *, study=None):
    """2D anisotropic stress from engineering-strain Voigt stiffness."""

    displacement = field_api.unwrap(displacement)
    if study is not None:
        _require_2d_study_supported(study)
    strain_voigt = engineering_strain_voigt_2d(displacement)
    stress_voigt = ufl.dot(ufl.as_matrix(properties.stiffness_voigt.tolist()), strain_voigt)
    return stress_voigt_to_tensor_2d(stress_voigt)


def stress(displacement, properties, *, study=None):
    """Dispatch to the matching elastic stress relation."""

    if isinstance(properties, ElasticIsotropicProperties):
        return isotropic_stress(displacement, properties, study=study)
    if isinstance(properties, ElasticAnisotropic2DProperties):
        return anisotropic_stress_2d(displacement, properties, study=study)
    if hasattr(properties, "stiffness_voigt"):
        return anisotropic_stress_2d(displacement, properties, study=study)
    if hasattr(properties, "young") and hasattr(properties, "poisson"):
        return isotropic_stress(displacement, properties, study=study)
    raise TypeError(f"unsupported elastic properties object: {type(properties)!r}")


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
