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
            "amplitude, regional material dispatch, physical-increment "
            "cutback, work/energy history, structurally benchmarked "
            "distributed Newton, and portable full-Step restart across MPI "
            "rank counts"
        ),
        limitations=(
            "3D only; plane stress needs a constrained local return map",
            "no kinematic-hardening global driver",
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
        model="isothermal and Arrhenius Mises time-hardening power-law creep",
        maturity="fem_integrated_foundation",
        available_scope=(
            "material-point constant-stress and relaxation checks plus a 3D "
            "small-strain global step with backward Euler, shared quadrature "
            "transaction, analytical tangent, automatic physical-time cutback, "
            "regional materials, standard creep fields, dissipation, portable "
            "full-Step restart, MPI-portable quadrature state, and a scalar or "
            "finite-element temperature field for normalized Arrhenius rates"
        ),
        limitations=(
            "the first global Newton provider is 3D; its MPI route is experimental",
            "transient thermal-history transfer is not automated",
            "Sinh and K-R laws remain local consumers",
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
    "cyclic_cohesive_fatigue": ConstitutiveCapability(
        name="cyclic_cohesive_fatigue",
        model=(
            "replaceable Mode-I opening-range or mixed-mode cohesive-energy "
            "fatigue damage layered on a bilinear envelope"
        ),
        maturity="experimental_mixed_mode_global_lifecycle",
        available_scope=(
            "independent cycle coordinate, sine/triangle/tabular force cycles, "
            "damage/front-limited cycle-jump decisions and ledger, analytical "
            "constant-extrema blocks, commit/rollback/restart, all-field "
            "physical-facet checkpoint, named 2D/3D cohesive interfaces, atomic "
            "multi-interface mesh splitting, global peak/valley/post-damage "
            "equilibrium lifecycle, automatic feedback cutback, durable cycle "
            "restart, native serial/MPI hyperelastic bulk-plus-cohesive Newton "
            "equilibrium with algorithmic tangent and strong-constraint reaction, "
            "3D failed-area/front/COD observations, complete local jump extrema, "
            "GI/GII cohesive-energy ranges, BK/power interaction, material-aware "
            "cyclic fields, ordered non-proportional jump paths, segment-resolved "
            "driving, and cross-rank-count physical-facet state portability"
        ),
        limitations=(
            "reference-point work and complete monotonic/fatigue energy closure remain",
            "peak/valley mixed-mode driving is limited to proportional cycles; non-proportional cycles require an explicitly ordered closed path",
            "reference fatigue evolution requires material/interface calibration",
            "local cohesive GI/GII drivers are not structural J-integral or VCCT energy-release rates",
            "no cylinder benchmark, cross-partition bulk restart, mixed-mode external validation, CT validation, or experimental prediction yet",
        ),
    ),
    "mixed_mode_cohesive_interface": ConstitutiveCapability(
        name="mixed_mode_cohesive_interface",
        model=(
            "quadratic nominal-traction initiation with bilinear energy "
            "evolution and BK or power-law mode interaction"
        ),
        maturity="experimental_global_facet_consumer",
        available_scope=(
            "2D/3D fixed paths, full vector jump and traction, intact/degraded "
            "tangential transfer, compression penalty, optional regularized "
            "Coulomb resistance, analytical tangent, standard interface fields, "
            "serial/MPI assembly, physical-facet restart, rigid-mode preflight, "
            "Mode-I deviation audit, spherical arc-length continuation, and an "
            "experimental proportional or ordered-path mixed-mode cyclic consumer"
        ),
        limitations=(
            "mixed-mode damage freezes mode mix at initiation and is intended for proportional or mildly changing paths",
            "friction is a smooth penalty regularization rather than a general contact active-set solver",
            "ordered non-proportional cycles are supported on fixed paths; free-path crack growth is not implemented",
            "no external mixed-mode structural benchmark has yet promoted this experimental law",
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
