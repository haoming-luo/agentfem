"""Named material definitions assembled from independent scientific behaviors.

The Study chooses the governing problem, not the material.  A material keeps
its identity and source while exposing one executable behavior per physics
role.  ``Model.material`` resolves that role and registers the existing
constitutive object, so the new public language does not fork the numerical
kernel or invalidate the concise 0.2.x constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class MaterialBehavior:
    """One named behavior carried by a physical material definition."""

    role: str
    model: object

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower().replace("-", "_")
        if not role:
            raise ValueError("MaterialBehavior.role must be non-empty.")
        if self.model is None:
            raise ValueError(f"Material behavior {role!r} requires a model object.")
        object.__setattr__(self, "role", role)

    def summary(self) -> dict[str, object]:
        value = self.model
        details = value.as_dict() if hasattr(value, "as_dict") else {}
        return {
            "role": self.role,
            "model": type(value).__name__,
            "details": details,
        }


@dataclass(frozen=True)
class MaterialCompatibility:
    """Pre-solve explanation of one material/Study pairing."""

    material: str
    physics: str
    analysis: str
    selected_role: str | None
    compatible: bool
    missing: tuple[str, ...] = ()
    message: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "material": self.material,
            "physics": self.physics,
            "analysis": self.analysis,
            "selected_role": self.selected_role,
            "compatible": self.compatible,
            "missing": list(self.missing),
            "message": self.message,
        }


@dataclass(frozen=True)
class MaterialDefinition:
    """Physical material identity plus independently reusable behaviors.

    ``mechanical`` and ``thermal`` are roles, not material families.  Isotropy,
    plasticity, hyperelasticity, creep, and similar choices remain properties
    of the supplied constitutive objects rather than branches in the material
    library hierarchy.
    """

    name: str
    behaviors: tuple[MaterialBehavior, ...]
    source: str = "user_defined"
    reference_only: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("MaterialDefinition.name must be non-empty.")
        behaviors = tuple(self.behaviors)
        if not behaviors:
            raise ValueError("A material definition requires at least one behavior.")
        roles = [item.role for item in behaviors]
        if len(set(roles)) != len(roles):
            raise ValueError(
                "A material definition may expose only one active model per role; "
                f"received duplicate roles {roles!r}."
            )
        source = str(self.source).strip()
        if not source:
            raise ValueError("MaterialDefinition.source must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "behaviors", behaviors)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.behaviors)

    def behavior(self, role: str) -> object:
        selected = str(role).strip().lower().replace("-", "_")
        for item in self.behaviors:
            if item.role == selected:
                return item.model
        raise KeyError(
            f"Material {self.name!r} has no {selected!r} behavior. "
            f"Available roles: {self.roles}."
        )

    def compatibility(self, study) -> MaterialCompatibility:
        physics = str(getattr(study, "physics", "unknown"))
        analysis = str(getattr(study, "analysis", "unknown"))
        role = _role_for_physics(physics)
        selected = _select_behavior(self, role)
        if selected is None:
            return MaterialCompatibility(
                material=self.name,
                physics=physics,
                analysis=analysis,
                selected_role=role,
                compatible=False,
                missing=(role,),
                message=(
                    f"Material {self.name!r} has no behavior for {physics!r}. "
                    f"Available roles: {self.roles}."
                ),
            )
        missing = _missing_capabilities(selected.model, study)
        return MaterialCompatibility(
            material=self.name,
            physics=physics,
            analysis=analysis,
            selected_role=selected.role,
            compatible=not missing,
            missing=missing,
            message=(
                "compatible"
                if not missing
                else f"Missing required material capabilities: {', '.join(missing)}."
            ),
        )

    def resolve_for(self, study) -> object:
        report = self.compatibility(study)
        if not report.compatible:
            raise ValueError(report.message)
        return self.behavior(str(report.selected_role))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "material_definition",
            "name": self.name,
            "source": self.source,
            "reference_only": bool(self.reference_only),
            "behaviors": [item.summary() for item in self.behaviors],
            "metadata": dict(self.metadata),
        }


def define(
    name: str,
    behavior=None,
    *,
    mechanical=None,
    thermal=None,
    behaviors: Mapping[str, object] | None = None,
    source: str = "user_defined",
    reference_only: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> MaterialDefinition:
    """Define a named material without coupling it to one Study.

    The positional ``behavior`` is the concise mechanical route.  Explicit
    role keywords keep thermo-mechanical and future multiphysics definitions
    readable without introducing a string mini-language.
    """

    selected: dict[str, object] = dict(behaviors or {})
    if behavior is not None:
        if mechanical is not None:
            raise ValueError("Pass positional behavior or mechanical=..., not both.")
        mechanical = behavior
    for role, model in (("mechanical", mechanical), ("thermal", thermal)):
        if model is None:
            continue
        if role in selected:
            raise ValueError(f"Material behavior role {role!r} was supplied twice.")
        selected[role] = model
    return MaterialDefinition(
        name=name,
        behaviors=tuple(
            MaterialBehavior(role, model) for role, model in selected.items()
        ),
        source=source,
        reference_only=reference_only,
        metadata={} if metadata is None else metadata,
    )


def _role_for_physics(physics: str) -> str:
    if physics == "heat_transfer":
        return "thermal"
    if physics == "solid_mechanics":
        return "mechanical"
    return physics


def _select_behavior(
    material: MaterialDefinition,
    preferred_role: str,
) -> MaterialBehavior | None:
    for item in material.behaviors:
        if item.role == preferred_role:
            return item
    if len(material.behaviors) == 1:
        candidate = material.behaviors[0]
        if preferred_role == "thermal" and hasattr(candidate.model, "conductivity"):
            return candidate
        if preferred_role == "mechanical" and not hasattr(candidate.model, "conductivity"):
            return candidate
    return None


def _missing_capabilities(model: object, study) -> tuple[str, ...]:
    missing: list[str] = []
    physics = getattr(study, "physics", None)
    analysis = getattr(study, "analysis", None)
    if physics == "heat_transfer":
        if not hasattr(model, "conductivity"):
            missing.append("conductivity")
        if analysis == "first_order_transient" and not any(
            hasattr(model, name)
            for name in ("volumetric_heat_capacity", "heat_capacity", "specific_heat")
        ):
            missing.append("heat_capacity")
    if physics == "solid_mechanics" and analysis == "second_order_dynamics":
        density = getattr(model, "density", None)
        if density is None:
            missing.append("density")
    return tuple(missing)
