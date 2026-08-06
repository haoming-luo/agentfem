"""Queryable maturity catalog for constitutive capabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstitutiveCapability:
    """What a material capability can truthfully do in this release."""

    name: str
    model: str
    maturity: str
    available_scope: str
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model,
            "maturity": self.maturity,
            "available_scope": self.available_scope,
            "limitations": self.limitations,
        }


_CAPABILITIES = {
    "linear_elasticity": ConstitutiveCapability(
        name="linear_elasticity",
        model="isotropic and selected 2D anisotropic small-strain elasticity",
        maturity="fem_integrated",
        available_scope=(
            "static, sequential thermoelasticity, and implicit/explicit dynamics"
        ),
    ),
    "neo_hookean": ConstitutiveCapability(
        name="neo_hookean",
        model="compressible finite-strain Neo-Hookean",
        maturity="fem_integrated",
        available_scope=(
            "nonlinear static in 3D and 2D plane strain; automatic or fixed "
            "load increments, Newton cutback/rollback, finite-J acceptance, "
            "and accepted-increment histories for ordinary Dirichlet and "
            "natural loading; affine periodic-cell load paths"
        ),
        limitations=(
            "one material contribution in the model convenience step",
            "2D plane stress thickness-stretch solve is not implemented",
        ),
    ),
    "j2_plasticity": ConstitutiveCapability(
        name="j2_plasticity",
        model="small-strain J2 plasticity with linear isotropic hardening",
        maturity="fem_integrated",
        available_scope=(
            "3D small-strain global Newton path with shared quadrature "
            "transaction, analytical consistent tangent, cyclic tabular "
            "amplitude, physical-increment cutback, work/energy history, "
            "and serial restart"
        ),
        limitations=(
            "3D only; plane stress needs a constrained local return map",
            "the first global provider and its checkpoint are serial-only",
            "no multi-region or kinematic-hardening global driver",
            "finite-strain plasticity is not implemented",
        ),
    ),
    "thermoelasticity": ConstitutiveCapability(
        name="thermoelasticity",
        model="isotropic linear thermoelasticity",
        maturity="fem_integrated",
        available_scope=(
            "steady and implicit transient heat transfer with regional "
            "multi-material conductivity/capacity, and sequential thermal-"
            "stress analysis in 2D/3D"
        ),
        limitations=(
            "temperature-dependent property tables are not implemented",
            "fully coupled monolithic temperature-displacement is not implemented",
        ),
    ),
    "power_law_creep": ConstitutiveCapability(
        name="power_law_creep",
        model="Mises time-hardening power-law creep",
        maturity="fem_integrated_foundation",
        available_scope=(
            "material-point constant-stress and relaxation checks plus a 3D "
            "isothermal small-strain global step with backward Euler, shared "
            "quadrature transaction, analytical tangent, automatic physical-"
            "time cutback, standard creep fields, dissipation, and serial restart"
        ),
        limitations=(
            "the first global provider is serial, 3D, isothermal, and single-material",
            "Arrhenius temperature fields and Sinh/K-R laws remain local consumers",
            "no external component benchmark or damage regularization",
        ),
    ),
    "creep_damage": ConstitutiveCapability(
        name="creep_damage",
        model="Kachanov-Rabotnov damage coupling and hyperbolic-sine creep",
        maturity="material_point_verified",
        available_scope=(
            "exact constant-stress K-R strain/damage update, rupture-time "
            "screening, associative Mises tensor flow, and Sinh stress response"
        ),
        limitations=(
            "von Mises is the only implemented multiaxial damage-stress measure",
            "no global quadrature creep step or mesh-regularized damage",
            "parameters require traceable material-specific calibration",
        ),
    ),
    "modified_theta_projection": ConstitutiveCapability(
        name="modified_theta_projection",
        model="three-parameter modified theta creep-curve projection",
        maturity="curve_projection_verified",
        available_scope=(
            "deterministic nonnegative curve fitting, strain/rate projection, "
            "and time-to-strain assessment"
        ),
        limitations=(
            "curve assessment only; not a global finite-element stress update",
            "stress and temperature dependence across tests must be calibrated externally",
        ),
    ),
    "stress_life_fatigue": ConstitutiveCapability(
        name="stress_life_fatigue",
        model="S-N/Basquin fatigue with rainflow and Miner damage",
        maturity="postprocessor",
        available_scope="scalar stress histories and block loading",
        limitations=(
            "uniaxial equivalent histories only",
            "no critical-plane multiaxial fatigue",
            "linear Goodman is the only mean-stress correction",
        ),
    ),
    "abaqus_user_material_bridge": ConstitutiveCapability(
        name="abaqus_user_material_bridge",
        model="solver-neutral material-point protocol and UMAT/UHYPER adapter specification",
        maturity="interface_contract",
        available_scope="validated data contracts and migration architecture",
        limitations=(
            "no compiled Abaqus subroutine adapter",
            "no quadrature-field state driver",
            "arbitrary UMAT/UHYPER source is not executable",
        ),
    ),
}


def capabilities() -> tuple[ConstitutiveCapability, ...]:
    """Return all constitutive capabilities in stable name order."""

    return tuple(_CAPABILITIES[name] for name in sorted(_CAPABILITIES))


def capability(name: str) -> ConstitutiveCapability:
    """Return one capability or raise with the available names."""

    key = str(name).lower().replace("-", "_")
    try:
        return _CAPABILITIES[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown constitutive capability {name!r}; "
            f"available={tuple(sorted(_CAPABILITIES))}."
        ) from exc
