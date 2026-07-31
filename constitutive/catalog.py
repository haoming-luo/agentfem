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
        available_scope="static, transient heat building blocks, and explicit dynamics",
    ),
    "neo_hookean": ConstitutiveCapability(
        name="neo_hookean",
        model="compressible finite-strain Neo-Hookean",
        maturity="fem_integrated",
        available_scope="nonlinear static, 3D and 2D plane strain",
        limitations=(
            "one material contribution in the model convenience step",
            "2D plane stress thickness-stretch solve is not implemented",
        ),
    ),
    "j2_plasticity": ConstitutiveCapability(
        name="j2_plasticity",
        model="small-strain J2 plasticity with linear isotropic hardening",
        maturity="material_point_verified",
        available_scope="3D tensor and exact uniaxial radial-return updates",
        limitations=(
            "no quadrature-field state driver",
            "no consistent-tangent FEM solve",
        ),
    ),
    "power_law_creep": ConstitutiveCapability(
        name="power_law_creep",
        model="Mises time-hardening power-law creep",
        maturity="material_point_verified",
        available_scope="constant-stress, relaxation, and tensor increments",
        limitations=(
            "no adaptive global time-step driver",
            "no quadrature-field state storage",
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
