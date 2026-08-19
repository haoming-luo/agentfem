"""Finite-strain hyperelastic constitutive relations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

import numpy as np
import ufl

from agentfem import fields as field_api


@dataclass(frozen=True)
class NeoHookeanProperties:
    """Compressible Neo-Hookean parameters derived from ``E`` and ``nu``.

    The strain-energy density is

    ``psi = mu/2 (I_C - d) - mu ln(J) + lambda/2 ln(J)^2``.

    In two dimensions this is the plane-strain restriction with the
    out-of-plane stretch fixed to one.  Plane stress requires an additional
    local solve and is not implied by this model.
    """

    young: float
    poisson: float
    density: float | None = None
    name: str = "compressible Neo-Hookean"

    def __post_init__(self) -> None:
        if not isfinite(float(self.young)) or self.young <= 0.0:
            raise ValueError("NeoHookeanProperties.young must be finite and positive.")
        if not isfinite(float(self.poisson)) or not (-1.0 < self.poisson < 0.5):
            raise ValueError(
                "NeoHookeanProperties.poisson must satisfy -1 < poisson < 0.5."
            )
        if self.density is not None and (
            not isfinite(float(self.density)) or self.density <= 0.0
        ):
            raise ValueError(
                "NeoHookeanProperties.density must be finite and positive when set."
            )

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lambda_(self) -> float:
        return self.young * self.poisson / (
            (1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)
        )

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "compressible_neo_hookean",
            "kinematics": "finite_strain",
            "two_dimensional_assumption": "plane_strain",
            "young": self.young,
            "poisson": self.poisson,
            "density": self.density,
            "mu": self.mu,
            "lambda": self.lambda_,
            "bulk_modulus": self.bulk_modulus,
            "maturity": "fem_form_available",
        }


@dataclass(frozen=True)
class PlaneStressNeoHookeanProperties(NeoHookeanProperties):
    """Compressible Neo-Hookean membrane with locally relaxed thickness.

    The in-plane deformation is embedded in a three-dimensional diagonal
    deformation gradient.  At every evaluation the thickness stretch is
    condensed by enforcing ``P33 = 0``.  This is a finite-strain plane-stress
    reduction, not the two-dimensional plane-strain energy with renamed
    metadata.
    """

    name: str = "plane-stress compressible Neo-Hookean"

    def as_dict(self) -> dict[str, object]:
        values = super().as_dict()
        values.update(
            {
                "model": "plane_stress_compressible_neo_hookean",
                "two_dimensional_assumption": "plane_stress",
                "thickness_condition": "P33=0_local_condensation",
                "maturity": "experimental_verified_homogeneous_paths",
            }
        )
        return values


@dataclass(frozen=True)
class MooneyRivlinProperties:
    """Two-parameter isotropic Mooney-Rivlin finite-strain solid.

    The three-dimensional compressible form uses the decoupled energy

    ``C10*(I1_bar-3) + C01*(I2_bar-3) + K/2*(J-1)^2``.

    With ``plane_stress_incompressible=True`` it instead consumes the exact
    reduced sheet energy used in Wang, Fineberg and Needleman (Eq. 17):

    ``mu/2[c*(I+J^-2-3) + (1-c)*(J^2+I*J^-2-3)]``.

    Here ``c`` is ``first_invariant_fraction``, ``C10=mu*c/2`` and
    ``C01=mu*(1-c)/2``.  The reduced form enforces thickness stretch ``1/J``
    and therefore must only be used in a two-dimensional plane-stress study.
    """

    shear_modulus: float
    first_invariant_fraction: float
    bulk_modulus_value: float = float("inf")
    density: float | None = None
    plane_stress_incompressible: bool = False
    name: str = "Mooney-Rivlin"

    def __post_init__(self) -> None:
        if not isfinite(float(self.shear_modulus)) or self.shear_modulus <= 0.0:
            raise ValueError("MooneyRivlinProperties.shear_modulus must be positive.")
        fraction = float(self.first_invariant_fraction)
        if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "first_invariant_fraction must be finite and lie in [0, 1]."
            )
        if self.plane_stress_incompressible:
            if not np.isinf(float(self.bulk_modulus_value)):
                raise ValueError(
                    "The incompressible plane-stress reduction requires "
                    "bulk_modulus_value=inf."
                )
        elif (
            not isfinite(float(self.bulk_modulus_value))
            or self.bulk_modulus_value <= 0.0
        ):
            raise ValueError("Compressible Mooney-Rivlin bulk modulus must be positive.")
        if self.density is not None and (
            not isfinite(float(self.density)) or self.density <= 0.0
        ):
            raise ValueError("Mooney-Rivlin density must be finite and positive.")

    @property
    def mu(self) -> float:
        return float(self.shear_modulus)

    @property
    def c10(self) -> float:
        return 0.5 * self.mu * float(self.first_invariant_fraction)

    @property
    def c01(self) -> float:
        return 0.5 * self.mu * (1.0 - float(self.first_invariant_fraction))

    @property
    def bulk_modulus(self) -> float:
        return float(self.bulk_modulus_value)

    @property
    def young(self) -> float:
        if self.plane_stress_incompressible:
            return 3.0 * self.mu
        return 9.0 * self.bulk_modulus * self.mu / (
            3.0 * self.bulk_modulus + self.mu
        )

    @property
    def poisson(self) -> float:
        if self.plane_stress_incompressible:
            return 0.5
        return (3.0 * self.bulk_modulus - 2.0 * self.mu) / (
            2.0 * (3.0 * self.bulk_modulus + self.mu)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": (
                "incompressible_plane_stress_mooney_rivlin"
                if self.plane_stress_incompressible
                else "compressible_mooney_rivlin"
            ),
            "kinematics": "finite_strain",
            "spatial_formulation": (
                "incompressible_plane_stress_sheet"
                if self.plane_stress_incompressible
                else "three_dimensional_compressible_solid"
            ),
            "two_dimensional_assumption": (
                "plane_stress" if self.plane_stress_incompressible else None
            ),
            "shear_modulus": self.mu,
            "first_invariant_fraction": self.first_invariant_fraction,
            "C10": self.c10,
            "C01": self.c01,
            "bulk_modulus": (
                None if self.plane_stress_incompressible else self.bulk_modulus
            ),
            "incompressibility": (
                "exact_sheet_reduction"
                if self.plane_stress_incompressible
                else "finite_bulk_penalty"
            ),
            "young": self.young,
            "poisson": self.poisson,
            "density": self.density,
            "source_equation": (
                "Wang-Fineberg-Needleman Eq. 17"
                if self.plane_stress_incompressible
                else None
            ),
            "maturity": "experimental_verified_material_paths",
        }


def mooney_rivlin(
    *,
    shear_modulus: float,
    first_invariant_fraction: float,
    bulk_modulus: float,
    density: float | None = None,
    name: str = "compressible Mooney-Rivlin",
) -> MooneyRivlinProperties:
    """Create a three-dimensional compressible Mooney-Rivlin solid."""

    return MooneyRivlinProperties(
        shear_modulus=shear_modulus,
        first_invariant_fraction=first_invariant_fraction,
        bulk_modulus_value=bulk_modulus,
        density=density,
        name=name,
    )


def mooney_rivlin_plane_stress(
    *,
    shear_modulus: float,
    first_invariant_fraction: float,
    density: float | None = None,
    name: str = "incompressible plane-stress Mooney-Rivlin",
) -> MooneyRivlinProperties:
    """Create the exact incompressible sheet reduction of Eq. (17)."""

    return MooneyRivlinProperties(
        shear_modulus=shear_modulus,
        first_invariant_fraction=first_invariant_fraction,
        bulk_modulus_value=float("inf"),
        density=density,
        plane_stress_incompressible=True,
        name=name,
    )


def is_finite_strain_hyperelastic(properties) -> bool:
    """Return whether a material is consumed by the displacement formulation."""

    return isinstance(properties, (NeoHookeanProperties, MooneyRivlinProperties))


def is_plane_stress_hyperelastic(properties) -> bool:
    return isinstance(properties, PlaneStressNeoHookeanProperties) or (
        isinstance(properties, MooneyRivlinProperties)
        and properties.plane_stress_incompressible
    )


def supports_hyperelastic_study(properties, *, dimension: int, assumption) -> bool:
    """Return whether one material has a formulation for the declared Study."""

    if not is_finite_strain_hyperelastic(properties):
        return False
    selected_dimension = int(dimension)
    if isinstance(properties, MooneyRivlinProperties):
        if properties.plane_stress_incompressible:
            return selected_dimension == 2 and assumption == "plane_stress"
        return selected_dimension == 3
    if isinstance(properties, PlaneStressNeoHookeanProperties):
        return selected_dimension == 2 and assumption == "plane_stress"
    return selected_dimension == 3 or (
        selected_dimension == 2 and assumption == "plane_strain"
    )


def neo_hookean(
    *,
    young: float,
    poisson: float,
    density: float | None = None,
    name: str = "compressible Neo-Hookean",
) -> NeoHookeanProperties:
    """Create a compressible Neo-Hookean material."""

    return NeoHookeanProperties(
        young=young,
        poisson=poisson,
        density=density,
        name=name,
    )


def neo_hookean_plane_stress(
    *,
    young: float,
    poisson: float,
    density: float | None = None,
    name: str = "plane-stress compressible Neo-Hookean",
) -> PlaneStressNeoHookeanProperties:
    """Create a finite-strain plane-stress Neo-Hookean membrane material."""

    return PlaneStressNeoHookeanProperties(
        young=young,
        poisson=poisson,
        density=density,
        name=name,
    )


@dataclass(frozen=True)
class MixedNeoHookeanProperties:
    """Isochoric Neo-Hookean solid with an independent pressure field.

    AgentFEM uses the perturbed-Lagrangian energy

    ``psi_iso(F) - p (J - 1) - p^2 / (2 kappa)``.

    Stationarity with respect to pressure gives
    ``p = -kappa (J - 1)``.  With ``DG0`` pressure interpolation this creates
    one pressure unknown per cell and avoids encoding hybrid semantics in a
    neutral mesh topology.
    """

    young: float
    poisson: float
    density: float | None = None
    name: str = "mixed Neo-Hookean"

    def __post_init__(self) -> None:
        if not isfinite(float(self.young)) or self.young <= 0.0:
            raise ValueError("MixedNeoHookeanProperties.young must be positive.")
        if not isfinite(float(self.poisson)) or not (-1.0 < self.poisson < 0.5):
            raise ValueError(
                "MixedNeoHookeanProperties.poisson must satisfy -1 < poisson < 0.5."
            )
        if self.density is not None and (
            not isfinite(float(self.density)) or self.density <= 0.0
        ):
            raise ValueError("MixedNeoHookeanProperties.density must be positive.")

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "mixed_neo_hookean",
            "kinematics": "finite_strain",
            "young": self.young,
            "poisson": self.poisson,
            "density": self.density,
            "mu": self.mu,
            "bulk_modulus": self.bulk_modulus,
            "C10": 0.5 * self.mu,
            "D1": 2.0 / self.bulk_modulus,
            "abaqus_C10": 0.5 * self.mu,
            "abaqus_D1": 2.0 / self.bulk_modulus,
            "volumetric_potential": "quadratic_J_minus_1",
            "pressure_sign": "positive_in_compression",
            "maturity": "verified_mixed_fem_form",
        }


def mixed_neo_hookean(
    *,
    young: float | None = None,
    poisson: float | None = None,
    shear_modulus: float | None = None,
    bulk_modulus: float | None = None,
    density: float | None = None,
    name: str = "mixed Neo-Hookean",
) -> MixedNeoHookeanProperties:
    """Create a quadratic-volumetric mixed Neo-Hookean material.

    Declare either ``young`` and ``poisson`` or the physically direct
    ``shear_modulus`` and ``bulk_modulus`` pair.  Eliminating the independent
    pressure from the mixed potential recovers

    ``mu/2 * (J^(-2/3) I1 - 3) + kappa/2 * (J - 1)^2``.

    In Abaqus polynomial notation this is ``C10=mu/2`` and ``D1=2/kappa``.
    """

    engineering = young is not None or poisson is not None
    direct = shear_modulus is not None or bulk_modulus is not None
    if engineering == direct:
        raise ValueError(
            "Declare exactly one complete pair: young/poisson or "
            "shear_modulus/bulk_modulus."
        )
    if engineering:
        if young is None or poisson is None:
            raise ValueError("young and poisson must be declared together.")
        selected_young = float(young)
        selected_poisson = float(poisson)
    else:
        if shear_modulus is None or bulk_modulus is None:
            raise ValueError(
                "shear_modulus and bulk_modulus must be declared together."
            )
        mu = float(shear_modulus)
        kappa = float(bulk_modulus)
        if not isfinite(mu) or mu <= 0.0:
            raise ValueError("shear_modulus must be finite and positive.")
        if not isfinite(kappa) or kappa <= 0.0:
            raise ValueError("bulk_modulus must be finite and positive.")
        selected_young = 9.0 * kappa * mu / (3.0 * kappa + mu)
        selected_poisson = (3.0 * kappa - 2.0 * mu) / (
            2.0 * (3.0 * kappa + mu)
        )

    return MixedNeoHookeanProperties(
        young=selected_young,
        poisson=selected_poisson,
        density=density,
        name=name,
    )


def mixed_condensed_energy_value(
    deformation_gradient,
    properties: MixedNeoHookeanProperties,
) -> float:
    """Evaluate the pressure-eliminated quadratic-volumetric energy."""

    F = np.asarray(deformation_gradient, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        raise ValueError("mixed_condensed_energy_value requires one finite 3x3 F.")
    J = float(np.linalg.det(F))
    if J <= 0.0:
        raise ValueError("The deformation gradient must have positive determinant.")
    invariant = float(np.trace(F.T @ F))
    return float(
        0.5 * properties.mu * (J ** (-2.0 / 3.0) * invariant - 3.0)
        + 0.5 * properties.bulk_modulus * (J - 1.0) ** 2
    )


@dataclass(frozen=True)
class FiniteStrainKinematics:
    """Standard total-Lagrangian kinematics derived from one displacement."""

    displacement: object

    def __post_init__(self) -> None:
        function = field_api.unwrap(self.displacement)
        shape = tuple(getattr(function, "ufl_shape", ()))
        if len(shape) != 1 or shape[0] not in {2, 3}:
            raise ValueError(
                "FiniteStrainKinematics requires a 2D or 3D displacement field."
            )
        object.__setattr__(self, "displacement", function)

    @property
    def dimension(self) -> int:
        return int(self.displacement.ufl_shape[0])

    @property
    def deformation_gradient(self):
        return ufl.Identity(self.dimension) + ufl.grad(self.displacement)

    @property
    def right_cauchy_green(self):
        F = self.deformation_gradient
        return F.T * F

    @property
    def jacobian(self):
        return ufl.det(self.deformation_gradient)

    @property
    def green_lagrange_strain(self):
        return 0.5 * (
            self.right_cauchy_green - ufl.Identity(self.dimension)
        )


def kinematics(displacement) -> FiniteStrainKinematics:
    """Return the standard finite-strain kinematic measures for ``u``."""

    return FiniteStrainKinematics(displacement)


def deformation_gradient(displacement):
    """Return ``F = I + grad(u)``."""

    return kinematics(displacement).deformation_gradient


def green_lagrange_strain(displacement):
    """Return the finite-strain tensor ``E = 1/2 (F.T F - I)``."""

    return kinematics(displacement).green_lagrange_strain


def strain_energy_density_from_gradient(
    F,
    properties: NeoHookeanProperties | MooneyRivlinProperties,
):
    """Return the declared hyperelastic energy density from ``F``."""

    if isinstance(properties, MooneyRivlinProperties):
        dimension = int(F.ufl_shape[0])
        if properties.plane_stress_incompressible:
            if tuple(F.ufl_shape) != (2, 2):
                raise ValueError(
                    "Incompressible plane-stress Mooney-Rivlin requires one 2x2 F."
                )
            invariant = ufl.tr(F * F.T)
            jacobian = ufl.det(F)
            fraction = float(properties.first_invariant_fraction)
            return 0.5 * properties.mu * (
                fraction * (invariant + jacobian**-2 - 3.0)
                + (1.0 - fraction)
                * (jacobian**2 + invariant * jacobian**-2 - 3.0)
            )
        if dimension != 3:
            raise ValueError(
                "Compressible Mooney-Rivlin currently requires one 3x3 F; "
                "use mooney_rivlin_plane_stress for a two-dimensional sheet."
            )
        C = F.T * F
        J = ufl.det(F)
        invariant_1 = ufl.tr(C)
        invariant_2 = 0.5 * (invariant_1**2 - ufl.tr(C * C))
        return (
            properties.c10 * (J ** (-2.0 / 3.0) * invariant_1 - 3.0)
            + properties.c01 * (J ** (-4.0 / 3.0) * invariant_2 - 3.0)
            + 0.5 * properties.bulk_modulus * (J - 1.0) ** 2
        )

    if isinstance(properties, PlaneStressNeoHookeanProperties):
        if tuple(F.ufl_shape) != (2, 2):
            raise ValueError(
                "PlaneStressNeoHookeanProperties requires one 2x2 gradient."
            )
        thickness = plane_stress_thickness_stretch_from_gradient(F, properties)
        in_plane_jacobian = ufl.det(F)
        jacobian = in_plane_jacobian * thickness
        invariant = ufl.tr(F.T * F) + thickness**2
        return (
            0.5 * properties.mu * (invariant - 3.0)
            - properties.mu * ufl.ln(jacobian)
            + 0.5 * properties.lambda_ * ufl.ln(jacobian) ** 2
        )
    dimension = F.ufl_shape[0]
    C = F.T * F
    J = ufl.det(F)
    return (
        0.5 * properties.mu * (ufl.tr(C) - dimension)
        - properties.mu * ufl.ln(J)
        + 0.5 * properties.lambda_ * ufl.ln(J) ** 2
    )


def strain_energy_density(
    displacement, properties: NeoHookeanProperties | MooneyRivlinProperties
):
    """Return the declared hyperelastic energy density ``psi(u)``."""

    return strain_energy_density_from_gradient(
        deformation_gradient(displacement),
        properties,
    )


def mixed_strain_energy_density(
    displacement,
    pressure,
    properties: MixedNeoHookeanProperties,
):
    """Return the mixed isochoric/volumetric energy density."""

    F = deformation_gradient(displacement)
    dimension = int(F.ufl_shape[0])
    J = ufl.det(F)
    isochoric_invariant = J ** (-2.0 / dimension) * ufl.tr(F.T * F)
    isochoric = 0.5 * properties.mu * (isochoric_invariant - dimension)
    return (
        isochoric
        - pressure * (J - 1.0)
        - 0.5 * pressure**2 / properties.bulk_modulus
    )


def mixed_first_piola(
    displacement,
    pressure,
    properties: MixedNeoHookeanProperties,
):
    """Return first Piola stress for the mixed Neo-Hookean potential."""

    F = deformation_gradient(displacement)
    dimension = int(F.ufl_shape[0])
    J = ufl.det(F)
    invariant = ufl.tr(F.T * F)
    inverse_transpose = ufl.inv(F).T
    isochoric = properties.mu * J ** (-2.0 / dimension) * (
        F - invariant / dimension * inverse_transpose
    )
    return isochoric - pressure * J * inverse_transpose


def mixed_cauchy_stress(
    displacement,
    pressure,
    properties: MixedNeoHookeanProperties,
):
    """Return Cauchy stress for the mixed Neo-Hookean potential."""

    F = deformation_gradient(displacement)
    return mixed_first_piola(displacement, pressure, properties) * F.T / ufl.det(F)


def first_piola_from_gradient(
    F, properties: NeoHookeanProperties | MooneyRivlinProperties
):
    """Return first Piola stress ``P = d psi / d F``."""

    if isinstance(properties, MooneyRivlinProperties):
        variable = ufl.variable(F)
        return ufl.diff(
            strain_energy_density_from_gradient(variable, properties), variable
        )

    if isinstance(properties, PlaneStressNeoHookeanProperties):
        variable = ufl.variable(F)
        return ufl.diff(
            strain_energy_density_from_gradient(variable, properties),
            variable,
        )

    inverse_transpose = ufl.inv(F).T
    J = ufl.det(F)
    return (
        properties.mu * (F - inverse_transpose)
        + properties.lambda_ * ufl.ln(J) * inverse_transpose
    )


def first_piola(
    displacement, properties: NeoHookeanProperties | MooneyRivlinProperties
):
    """Return the first Piola stress for a displacement field."""

    return first_piola_from_gradient(
        deformation_gradient(displacement),
        properties,
    )


def cauchy_stress(
    displacement, properties: NeoHookeanProperties | MooneyRivlinProperties
):
    """Return the Cauchy stress ``sigma = J^-1 P F^T``."""

    F = deformation_gradient(displacement)
    J = ufl.det(F)
    if isinstance(properties, PlaneStressNeoHookeanProperties):
        J *= plane_stress_thickness_stretch_from_gradient(F, properties)
    elif (
        isinstance(properties, MooneyRivlinProperties)
        and properties.plane_stress_incompressible
    ):
        J = ufl.as_ufl(1.0)
    return (1.0 / J) * first_piola_from_gradient(F, properties) * F.T


def mooney_rivlin_energy_value(
    deformation_gradient,
    properties: MooneyRivlinProperties,
) -> float:
    """Evaluate either supported Mooney-Rivlin energy numerically."""

    if not isinstance(properties, MooneyRivlinProperties):
        raise TypeError("mooney_rivlin_energy_value requires MooneyRivlinProperties.")
    F = np.asarray(deformation_gradient, dtype=float)
    if not np.all(np.isfinite(F)) or float(np.linalg.det(F)) <= 0.0:
        raise ValueError("Mooney-Rivlin deformation gradient must be finite with J>0.")
    if properties.plane_stress_incompressible:
        if F.shape != (2, 2):
            raise ValueError("Plane-stress Mooney-Rivlin requires one 2x2 F.")
        invariant = float(np.trace(F @ F.T))
        jacobian = float(np.linalg.det(F))
        fraction = float(properties.first_invariant_fraction)
        return float(
            0.5
            * properties.mu
            * (
                fraction * (invariant + jacobian**-2 - 3.0)
                + (1.0 - fraction)
                * (jacobian**2 + invariant * jacobian**-2 - 3.0)
            )
        )
    if F.shape != (3, 3):
        raise ValueError("Compressible Mooney-Rivlin requires one 3x3 F.")
    C = F.T @ F
    jacobian = float(np.linalg.det(F))
    invariant_1 = float(np.trace(C))
    invariant_2 = 0.5 * (invariant_1**2 - float(np.trace(C @ C)))
    return float(
        properties.c10 * (jacobian ** (-2.0 / 3.0) * invariant_1 - 3.0)
        + properties.c01 * (jacobian ** (-4.0 / 3.0) * invariant_2 - 3.0)
        + 0.5 * properties.bulk_modulus * (jacobian - 1.0) ** 2
    )


def mooney_rivlin_first_piola_value(
    deformation_gradient,
    properties: MooneyRivlinProperties,
) -> np.ndarray:
    """Evaluate first Piola stress for numerical oracles and wave analysis."""

    F = np.asarray(deformation_gradient, dtype=float)
    if properties.plane_stress_incompressible:
        if F.shape != (2, 2) or float(np.linalg.det(F)) <= 0.0:
            raise ValueError("Plane-stress Mooney-Rivlin requires a positive-J 2x2 F.")
        scale = max(1.0, float(np.linalg.norm(F)))
        step = np.cbrt(np.finfo(float).eps) * scale
        result = np.empty_like(F)
        for i in range(2):
            for j in range(2):
                perturbation = np.zeros_like(F)
                perturbation[i, j] = step
                result[i, j] = (
                    mooney_rivlin_energy_value(F + perturbation, properties)
                    - mooney_rivlin_energy_value(F - perturbation, properties)
                ) / (2.0 * step)
        return result
    if F.shape != (3, 3) or float(np.linalg.det(F)) <= 0.0:
        raise ValueError("Compressible Mooney-Rivlin requires a positive-J 3x3 F.")
    J = float(np.linalg.det(F))
    inverse_transpose = np.linalg.inv(F).T
    C = F.T @ F
    invariant_1 = float(np.trace(C))
    invariant_2 = 0.5 * (invariant_1**2 - float(np.trace(C @ C)))
    return (
        2.0
        * properties.c10
        * J ** (-2.0 / 3.0)
        * (F - invariant_1 / 3.0 * inverse_transpose)
        + 2.0
        * properties.c01
        * J ** (-4.0 / 3.0)
        * (
            invariant_1 * F
            - F @ C
            - 2.0 * invariant_2 / 3.0 * inverse_transpose
        )
        + properties.bulk_modulus * J * (J - 1.0) * inverse_transpose
    )


def plane_stress_thickness_stretch_from_gradient(
    F,
    properties: PlaneStressNeoHookeanProperties,
    *,
    iterations: int = 2,
):
    """Return the local thickness stretch satisfying ``P33 = 0``.

    Newton iterations are embedded in the UFL expression.  The initial value
    is the infinitesimal plane-stress contraction continued multiplicatively.
    Two iterations give tight closure on the verified deformation range
    while keeping automatic differentiation portable across FFCx versions. The
    independent numerical material-point oracle iterates to tolerance.
    """

    if not isinstance(properties, PlaneStressNeoHookeanProperties):
        raise TypeError(
            "plane_stress_thickness_stretch_from_gradient requires "
            "PlaneStressNeoHookeanProperties."
        )
    if tuple(F.ufl_shape) != (2, 2):
        raise ValueError("Finite-strain plane stress requires one 2x2 gradient.")
    count = int(iterations)
    if count <= 0:
        raise ValueError("Plane-stress local iterations must be positive.")
    in_plane_jacobian = ufl.det(F)
    exponent = -properties.poisson / (1.0 - properties.poisson)
    thickness = in_plane_jacobian**exponent
    for _ in range(count):
        residual = (
            properties.mu * (thickness**2 - 1.0)
            + properties.lambda_ * ufl.ln(in_plane_jacobian * thickness)
        )
        derivative = (
            2.0 * properties.mu * thickness
            + properties.lambda_ / thickness
        )
        thickness = thickness - residual / derivative
    return thickness


def plane_stress_out_of_plane_first_piola_from_gradient(
    F,
    properties: PlaneStressNeoHookeanProperties,
):
    """Return the condensed ``P33`` residual for diagnostics and tests."""

    thickness = plane_stress_thickness_stretch_from_gradient(F, properties)
    jacobian = ufl.det(F) * thickness
    return (
        properties.mu * (thickness - 1.0 / thickness)
        + properties.lambda_ * ufl.ln(jacobian) / thickness
    )


def plane_stress_thickness_stretch_value(
    deformation_gradient,
    properties: PlaneStressNeoHookeanProperties,
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 30,
) -> float:
    """Solve the local ``P33=0`` condition for one numerical 2x2 ``F``."""

    if not isinstance(properties, PlaneStressNeoHookeanProperties):
        raise TypeError(
            "plane_stress_thickness_stretch_value requires "
            "PlaneStressNeoHookeanProperties."
        )
    F = np.asarray(deformation_gradient, dtype=float)
    if F.shape != (2, 2) or not np.all(np.isfinite(F)):
        raise ValueError("Plane-stress deformation_gradient must be finite 2x2.")
    in_plane_jacobian = float(np.linalg.det(F))
    if in_plane_jacobian <= 0.0:
        raise ValueError("Plane-stress condensation requires det(F2) > 0.")
    selected_tolerance = float(tolerance)
    if not isfinite(selected_tolerance) or selected_tolerance <= 0.0:
        raise ValueError("Plane-stress tolerance must be finite and positive.")
    count = int(maximum_iterations)
    if count <= 0:
        raise ValueError("maximum_iterations must be positive.")
    exponent = -properties.poisson / (1.0 - properties.poisson)
    thickness = in_plane_jacobian**exponent
    scale = max(abs(properties.mu), abs(properties.lambda_), 1.0)
    for _ in range(count):
        residual = (
            properties.mu * (thickness**2 - 1.0)
            + properties.lambda_ * log(in_plane_jacobian * thickness)
        )
        if abs(residual) <= selected_tolerance * scale:
            return float(thickness)
        derivative = (
            2.0 * properties.mu * thickness
            + properties.lambda_ / thickness
        )
        candidate = thickness - residual / derivative
        if not isfinite(candidate) or candidate <= 0.0:
            candidate = 0.5 * thickness
        thickness = candidate
    raise RuntimeError(
        "Plane-stress thickness condensation did not converge within "
        f"{count} iterations."
    )


def plane_stress_first_piola_value(
    deformation_gradient,
    properties: PlaneStressNeoHookeanProperties,
) -> np.ndarray:
    """Return the condensed numerical in-plane first Piola stress."""

    F = np.asarray(deformation_gradient, dtype=float)
    thickness = plane_stress_thickness_stretch_value(F, properties)
    inverse_transpose = np.linalg.inv(F).T
    jacobian = float(np.linalg.det(F)) * thickness
    return (
        properties.mu * (F - inverse_transpose)
        + properties.lambda_ * np.log(jacobian) * inverse_transpose
    )


def plane_stress_uniaxial_deformation_gradient(
    axial_stretch: float,
    properties: PlaneStressNeoHookeanProperties | MooneyRivlinProperties,
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 30,
) -> np.ndarray:
    """Return homogeneous uniaxial ``F2`` with traction-free lateral faces.

    Isotropy makes the in-plane lateral and condensed thickness stretches
    equal.  The returned gradient therefore satisfies both ``P11=0`` and the
    local ``P33=0`` plane-stress condition.
    """

    if not is_plane_stress_hyperelastic(properties):
        raise TypeError(
            "plane_stress_uniaxial_deformation_gradient requires "
            "a supported plane-stress hyperelastic material."
        )
    axial = float(axial_stretch)
    if not isfinite(axial) or axial <= 0.0:
        raise ValueError("axial_stretch must be finite and positive.")
    if isinstance(properties, MooneyRivlinProperties):
        # For an isotropic incompressible sheet with traction-free width and
        # thickness, symmetry and J3=1 give lambda_width=lambda_thickness.
        return np.diag((axial**-0.5, axial))
    selected_tolerance = float(tolerance)
    lateral = axial ** (-properties.poisson)
    scale = max(abs(properties.mu), abs(properties.lambda_), 1.0)
    for _ in range(int(maximum_iterations)):
        residual = (
            properties.mu * (lateral**2 - 1.0)
            + properties.lambda_ * log(lateral**2 * axial)
        )
        if abs(residual) <= selected_tolerance * scale:
            return np.diag((float(lateral), axial))
        derivative = (
            2.0 * properties.mu * lateral
            + 2.0 * properties.lambda_ / lateral
        )
        candidate = lateral - residual / derivative
        lateral = candidate if isfinite(candidate) and candidate > 0.0 else 0.5 * lateral
    raise RuntimeError("Uniaxial plane-stress lateral contraction did not converge.")


def internal_virtual_work(
    displacement,
    test_function,
    properties: NeoHookeanProperties | MooneyRivlinProperties,
    *,
    measure=ufl.dx,
):
    """Return ``integral P : grad(v) dV``."""

    displacement = field_api.unwrap(displacement)
    return ufl.inner(
        first_piola(displacement, properties),
        ufl.grad(test_function),
    ) * measure


def residual(
    displacement,
    test_function,
    properties: NeoHookeanProperties | MooneyRivlinProperties,
    *,
    body_force=None,
    traction=None,
    measure=ufl.dx,
    boundary_measure=None,
):
    """Build total Lagrangian residual ``internal - external``.

    ``traction`` is interpreted as nominal traction per reference area.
    """

    value = internal_virtual_work(
        displacement,
        test_function,
        properties,
        measure=measure,
    )
    if body_force is not None:
        value -= ufl.inner(body_force, test_function) * measure
    if traction is not None:
        if boundary_measure is None:
            raise ValueError("hyperelastic residual traction requires boundary_measure.")
        value -= ufl.inner(traction, test_function) * boundary_measure
    return value


def tangent(residual_form, displacement, trial_function=None):
    """Differentiate a residual form with respect to displacement."""

    displacement = field_api.unwrap(displacement)
    if trial_function is None:
        trial_function = ufl.TrialFunction(displacement.function_space)
    return ufl.derivative(residual_form, displacement, trial_function)


def principal_nominal_stress(
    stretches,
    properties: NeoHookeanProperties | MooneyRivlinProperties,
) -> np.ndarray:
    """Evaluate diagonal first-Piola stresses for principal stretches.

    This numerical helper is useful for material tests and analytical
    verification.  All stretches must be finite and positive.
    """

    selected = np.asarray(stretches, dtype=float)
    if selected.ndim != 1 or selected.size not in {2, 3}:
        raise ValueError("stretches must be a vector of length two or three.")
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("principal stretches must be finite and positive.")
    if isinstance(properties, MooneyRivlinProperties):
        return np.diag(
            mooney_rivlin_first_piola_value(np.diag(selected), properties)
        )
    J = float(np.prod(selected))
    return (
        properties.mu * (selected - 1.0 / selected)
        + properties.lambda_ * log(J) / selected
    )


def principal_energy_density(
    stretches,
    properties: NeoHookeanProperties | MooneyRivlinProperties,
) -> float:
    """Evaluate the Neo-Hookean energy for principal stretches."""

    selected = np.asarray(stretches, dtype=float)
    if selected.ndim != 1 or selected.size not in {2, 3}:
        raise ValueError("stretches must be a vector of length two or three.")
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("principal stretches must be finite and positive.")
    if isinstance(properties, MooneyRivlinProperties):
        return mooney_rivlin_energy_value(np.diag(selected), properties)
    J = float(np.prod(selected))
    return float(
        0.5 * properties.mu * (np.dot(selected, selected) - selected.size)
        - properties.mu * log(J)
        + 0.5 * properties.lambda_ * log(J) ** 2
    )
