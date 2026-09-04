"""Constraint containers for standard finite-element workflows.

Strong constraints such as Dirichlet data and periodic/MPC relations belong
here. Natural boundary data such as Neumann fluxes and tractions are weak-form
loads, so they belong in ``agentfem.loads``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from .. import amplitudes
from ..ir.values import describe_value
from ..kernel import constants, dofs
from . import boundary, mpc
from .affine import (
    AbaqusPeriodicConstraint,
    AffineReduction,
    DeformationGradientPath,
    DistributedAffineReduction,
    abaqus_periodic_cell,
    deformation_gradient_path,
)
from .mpc import RectangularPeriodicMPC, rectangular_periodic_mpc


@dataclass(frozen=True)
class ConstraintCapabilities:
    """Solver-facing capability contract for one kinematic constraint.

    Physical meaning and numerical enforcement stay separate: a periodic
    relation may be enforced by explicit nodal projection or by exact affine
    elimination/MPC.  Model validation, agents, and future GUIs consume this
    same contract before any form is assembled.
    """

    kind: str
    enforcement: str
    analyses: tuple[str, ...] = ()
    procedures: tuple[str, ...] = ()
    strict: bool = True
    supports_parallel: bool = True
    reaction_evidence: str = "provider_defined"
    work_evidence: str = "provider_defined"

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "enforcement": self.enforcement,
            "analyses": self.analyses,
            "procedures": self.procedures,
            "strict": self.strict,
            "supports_parallel": self.supports_parallel,
            "reaction_evidence": self.reaction_evidence,
            "work_evidence": self.work_evidence,
        }


@dataclass(frozen=True)
class ConstraintDualEvidence:
    """Provider-owned force and optional work-conjugate coordinate.

    MPC, weak and contact reactions cannot be reconstructed from the strong-
    Dirichlet residual.  Their active provider must publish this record after
    convergence.  The record carries values and provenance; it never changes
    the constraint or solver state.
    """

    constraint_name: str
    role: str
    force: np.ndarray
    coordinate: np.ndarray | None = None
    resultant: np.ndarray | None = None
    source: str = "provider_dual"
    complete: bool = True

    def __post_init__(self) -> None:
        name = str(self.constraint_name).strip()
        role = str(self.role).strip().lower().replace("-", "_")
        allowed = {"mpc_constraint", "weak_constraint", "contact_constraint"}
        force = np.asarray(self.force, dtype=float).reshape(-1)
        coordinate = (
            None
            if self.coordinate is None
            else np.asarray(self.coordinate, dtype=float).reshape(-1)
        )
        resultant = (
            None
            if self.resultant is None
            else np.asarray(self.resultant, dtype=float).reshape(-1)
        )
        if not name or role not in allowed:
            raise ValueError("Constraint dual evidence needs a name and dual role.")
        if force.size == 0 or not np.all(np.isfinite(force)):
            raise ValueError("Constraint dual force must contain finite values.")
        if coordinate is not None and (
            coordinate.shape != force.shape or not np.all(np.isfinite(coordinate))
        ):
            raise ValueError(
                "Constraint dual coordinate must be finite and match the force."
            )
        if resultant is not None and (
            resultant.size == 0 or not np.all(np.isfinite(resultant))
        ):
            raise ValueError("Constraint dual resultant must contain finite values.")
        if not str(self.source).strip():
            raise ValueError("Constraint dual evidence must identify its provider.")
        object.__setattr__(self, "constraint_name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "force", force.copy())
        object.__setattr__(
            self,
            "coordinate",
            None if coordinate is None else coordinate.copy(),
        )
        object.__setattr__(
            self,
            "resultant",
            None if resultant is None else resultant.copy(),
        )

    @property
    def force_complete(self) -> bool:
        """Whether a physical-space resultant closes global force balance."""

        return bool(self.complete and self.resultant is not None)

    @property
    def work_complete(self) -> bool:
        return bool(self.complete and self.coordinate is not None)

    def work_sample(self):
        """Return the shared work sample, failing if no coordinate was supplied."""

        if self.coordinate is None:
            raise RuntimeError(
                f"Constraint dual {self.constraint_name!r} has no work coordinate."
            )
        from .._work_energy import GeneralizedWorkSample

        return GeneralizedWorkSample(
            name=self.constraint_name,
            role=self.role,
            force=self.force,
            displacement=self.coordinate,
        )

    def summary(self) -> dict[str, object]:
        return {
            "constraint_name": self.constraint_name,
            "role": self.role,
            "force": self.force.tolist(),
            "coordinate": (
                None if self.coordinate is None else self.coordinate.tolist()
            ),
            "resultant": (
                None if self.resultant is None else self.resultant.tolist()
            ),
            "source": self.source,
            "complete": bool(self.complete),
            "force_complete": self.force_complete,
            "work_complete": self.work_complete,
        }


def constraint_dual(
    constraint,
    *,
    force,
    coordinate=None,
    resultant=None,
    role="mpc_constraint",
    source="provider_dual",
    complete=True,
) -> ConstraintDualEvidence:
    """Create provider evidence tied to one named constraint asset."""

    return ConstraintDualEvidence(
        constraint_name=str(getattr(constraint, "name", constraint)),
        role=role,
        force=force,
        coordinate=coordinate,
        resultant=resultant,
        source=source,
        complete=complete,
    )


def collect_provider_duals(constraints, problem, *, extra=()) -> tuple[ConstraintDualEvidence, ...]:
    """Collect converged dual evidence from active constraint providers.

    A constraint that owns reactions outside the strong-Dirichlet residual may
    implement ``dual_evidence(problem)`` and return one
    :class:`ConstraintDualEvidence` record, an iterable of records, or ``None``.
    AgentFEM only transports and validates those provider-owned values; it does
    not reconstruct an MPC multiplier, weak-boundary traction, or contact force
    from incomplete public state.

    ``extra`` retains the internal callback bridge used by older Step
    providers while they migrate to the constraint-owned protocol.
    """

    records: list[ConstraintDualEvidence] = []
    assets = _flatten_constraint_assets(constraints)
    declared = {
        str(getattr(item, "name", type(item).__name__)) for item in assets
    }
    for item in assets:
        provider = getattr(item, "dual_evidence", None)
        if provider is None:
            continue
        if not callable(provider):
            raise TypeError(
                f"Constraint {getattr(item, 'name', type(item).__name__)!r} "
                "exposes a non-callable dual_evidence provider."
            )
        supplied = provider(problem)
        if supplied is None:
            continue
        selected = (
            (supplied,)
            if isinstance(supplied, ConstraintDualEvidence)
            else tuple(supplied)
        )
        if any(not isinstance(value, ConstraintDualEvidence) for value in selected):
            raise TypeError(
                "Constraint dual_evidence providers must return "
                "ConstraintDualEvidence records."
            )
        records.extend(selected)
    records.extend(tuple(extra))
    if any(not isinstance(item, ConstraintDualEvidence) for item in records):
        raise TypeError("extra provider duals must be ConstraintDualEvidence records.")
    unexpected = tuple(
        sorted({item.constraint_name for item in records}.difference(declared))
    )
    if unexpected:
        raise ValueError(
            "Provider dual evidence does not match declared constraints: "
            f"{unexpected!r}."
        )
    names = tuple(item.constraint_name for item in records)
    if len(set(names)) != len(names):
        raise ValueError("Provider dual constraint names must be unique.")
    expected_roles = {}
    for item in assets:
        capability = constraint_capabilities(item)
        if capability is None:
            continue
        role = {
            "periodic_constraint": "mpc_constraint",
            "mpc_constraint": "mpc_constraint",
            "weak_constraint": "weak_constraint",
            "contact_constraint": "contact_constraint",
        }.get(capability.kind)
        if role is not None:
            expected_roles[str(getattr(item, "name", type(item).__name__))] = role
    incompatible = tuple(
        sorted(
            (item.constraint_name, item.role, expected_roles[item.constraint_name])
            for item in records
            if item.constraint_name in expected_roles
            and item.role != expected_roles[item.constraint_name]
        )
    )
    if incompatible:
        raise ValueError(
            "Provider dual roles do not match declared constraint capabilities: "
            f"{incompatible!r}."
        )
    return tuple(records)


@dataclass(frozen=True)
class DirichletConstraint:
    """Strong Dirichlet constraint and its optional mutable value object."""

    bc: object
    value: object | None = None
    name: str = "dirichlet"
    location: object | None = None

    @classmethod
    def component(
        cls,
        V,
        component: int,
        marker=None,
        value=0.0,
        *,
        location=None,
        name: str = "dirichlet",
    ):
        """Create a component-wise Dirichlet constraint on a vector space."""

        constant, bc = boundary.component_dirichlet_bc(
            V,
            component,
            marker,
            value=value,
            location=location,
        )
        selected_location = location
        return cls(bc=bc, value=constant, name=name, location=selected_location)

    @classmethod
    def scalar(cls, V, marker=None, value=0.0, *, location=None, name: str = "dirichlet"):
        """Create a scalar Dirichlet constraint."""

        constant, bc = boundary.scalar_dirichlet_bc(V, marker, value=value, location=location)
        return cls(bc=bc, value=constant, name=name, location=location)

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            "name": self.name,
            "kind": "dirichlet_constraint",
            "location": getattr(self.location, "name", None),
            "value": describe_value(self.value),
        }


@dataclass(frozen=True)
class TimeDependentDirichlet:
    """Dirichlet constraint driven by an amplitude."""

    constant: object
    bc: object
    amplitude: amplitudes.Amplitude
    name: str = "time_dependent_dirichlet"
    location: object | None = None

    def update(self, time: float) -> float:
        """Evaluate the amplitude and update the backing constant."""

        value = self.amplitude(time)
        self.constant.value = constants.scalar_value(value)
        return value

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            "name": self.name,
            "kind": "time_dependent_dirichlet",
            "location": getattr(self.location, "name", None),
            "amplitude": self.amplitude.summary(),
        }


@dataclass(frozen=True)
class RemoteDisplacementConstraint:
    """Rigid boundary motion prescribed about a named reference point."""

    bc: object
    value: object
    reference_values: np.ndarray
    reference_point: object
    translation: tuple[float, ...]
    rotation: object
    name: str = "remote_displacement"
    location: object | None = None
    coordinate_system: str | None = None

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "remote_displacement_constraint",
            "location": getattr(self.location, "name", None),
            "reference_point": getattr(
                self.reference_point, "name", "reference_point"
            ),
            "translation": self.translation,
            "rotation": self.rotation,
            "coordinate_system": self.coordinate_system,
            "kinematics": "prescribed rigid boundary motion",
        }


@dataclass
class PrescribedValuePath:
    """Update ordinary strong boundary values along a normalized step path.

    Constant Dirichlet data are interpreted as end-of-step values and scaled
    by ``0 <= factor <= 1``.  A :class:`TimeDependentDirichlet` instead
    evaluates its own amplitude at the normalized factor.  The object keeps
    this policy visible and reusable by nonlinear procedures.
    """

    constants: tuple[tuple[object, object], ...] = ()
    fields: tuple[tuple[object, np.ndarray], ...] = ()
    amplitudes: tuple[TimeDependentDirichlet, ...] = ()

    def update(self, factor: float) -> None:
        selected = float(factor)
        if not 0.0 <= selected <= 1.0 + 1.0e-12:
            raise ValueError("PrescribedValuePath factor must lie in [0, 1].")
        for constant, reference in self.constants:
            constant.value = selected * reference
        for function, reference in self.fields:
            function.x.array[:] = selected * reference
            function.x.scatter_forward()
        for constraint in self.amplitudes:
            constraint.update(selected)

    def snapshot_runtime_state(self) -> dict[str, tuple[np.ndarray, ...]]:
        """Capture backing values so a failed nonlinear attempt is atomic."""

        return {
            "constants": tuple(
                np.asarray(constant.value).copy()
                for constant, _reference in self.constants
            ),
            "fields": tuple(
                np.asarray(function.x.array).copy()
                for function, _reference in self.fields
            ),
            "amplitudes": tuple(
                np.asarray(constraint.constant.value).copy()
                for constraint in self.amplitudes
            ),
        }

    def restore_runtime_state(
        self,
        state: dict[str, tuple[np.ndarray, ...]],
    ) -> None:
        """Restore a snapshot without re-evaluating any user amplitude."""

        expected = {
            "constants": len(self.constants),
            "fields": len(self.fields),
            "amplitudes": len(self.amplitudes),
        }
        actual = {name: len(tuple(state.get(name, ()))) for name in expected}
        if actual != expected:
            raise ValueError(
                "Prescribed-value runtime state differs from the declared path."
            )
        for (constant, _reference), value in zip(
            self.constants,
            state["constants"],
        ):
            constant.value = value
        for (function, _reference), value in zip(
            self.fields,
            state["fields"],
        ):
            function.x.array[:] = value
            function.x.scatter_forward()
        for constraint, value in zip(self.amplitudes, state["amplitudes"]):
            constraint.constant.value = value

    def summary(self) -> dict[str, object]:
        return {
            "kind": "prescribed_value_path",
            "constant_values": len(self.constants),
            "field_values": len(self.fields),
            "amplitude_values": len(self.amplitudes),
        }


def prescribed_value_path(constraints) -> PrescribedValuePath:
    """Create a normalized load-factor driver from registered constraints."""

    constants = []
    histories = []
    fields = []
    for item in _flatten_dirichlet(constraints):
        if isinstance(item, TimeDependentDirichlet):
            histories.append(item)
            continue
        if isinstance(item, RemoteDisplacementConstraint):
            fields.append((item.value, item.reference_values.copy()))
            continue
        value = getattr(item, "value", None)
        if value is None or not hasattr(value, "value"):
            continue
        reference = np.asarray(value.value).copy()
        constants.append((value, reference))
    return PrescribedValuePath(tuple(constants), tuple(fields), tuple(histories))


def dirichlet_constraints(constraints) -> tuple[object, ...]:
    """Return concrete Dirichlet assets from nested model constraint sets."""

    return _flatten_dirichlet(constraints)


def constraint_assets(constraints) -> tuple[object, ...]:
    """Return every concrete asset from nested constraint containers.

    Solver providers must inspect this complete view before lowering.  Using
    :func:`dirichlet_constraints` for capability selection would intentionally
    omit periodic/MPC assets and could therefore turn an unsupported mixed
    constraint set into an apparently ordinary strong-boundary problem.
    """

    return _flatten_constraint_assets(constraints)


def _flatten_dirichlet(items) -> tuple[object, ...]:
    if items is None:
        return ()
    if isinstance(items, (list, tuple)):
        result = []
        for item in items:
            result.extend(_flatten_dirichlet(item))
        return tuple(result)
    if hasattr(items, "dirichlet"):
        return _flatten_dirichlet(items.dirichlet)
    return (items,)


def scalar_dirichlet(
    V,
    marker=None,
    value=0.0,
    *,
    location=None,
    on=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Semantic wrapper for scalar essential boundary data."""

    selected_location = _select_location(location=location, on=on)
    return DirichletConstraint.scalar(V, marker, value=value, location=selected_location, name=name)


def component_dirichlet(
    V,
    component: int,
    marker=None,
    value=0.0,
    *,
    location=None,
    on=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Semantic wrapper for vector-component essential boundary data."""

    selected_location = _select_location(location=location, on=on)
    return DirichletConstraint.component(
        V,
        component,
        marker,
        value=value,
        location=selected_location,
        name=name,
    )


def axisymmetric_plane_strain(
    displacement,
    *,
    value: float = 0.0,
    name: str = "axisymmetric_plane_strain",
) -> DirichletConstraint:
    """Constrain ``u_z`` everywhere in an ``(r, z)`` meridian model.

    This is the long-cylinder plane-strain specialization of an axisymmetric
    solid, not a generic requirement of axisymmetric analysis.  It includes
    interior high-order displacement dofs and is therefore stronger and less
    error-prone than fixing only the two end boundaries.
    """

    return component_dirichlet(
        displacement,
        1,
        marker=lambda x: np.ones(x.shape[1], dtype=bool),
        value=value,
        name=name,
    )


def axisymmetric_axis(
    displacement,
    *,
    location=None,
    on=None,
    value: float = 0.0,
    name: str = "axisymmetric_axis",
) -> DirichletConstraint:
    """Enforce radial regularity ``u_r=0`` on the revolution axis."""

    return component_dirichlet(
        displacement,
        0,
        marker=lambda x: np.isclose(x[0], 0.0),
        location=location,
        on=on,
        value=value,
        name=name,
    )


def dirichlet(
    V,
    marker=None,
    value=0.0,
    *,
    component: int | None = None,
    location=None,
    on=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Create scalar or component-wise Dirichlet data from one entry point."""

    selected_location = _select_location(location=location, on=on)
    if component is None:
        return scalar_dirichlet(V, marker, value=value, location=selected_location, name=name)
    return component_dirichlet(
        V,
        component,
        marker,
        value=value,
        location=selected_location,
        name=name,
    )


def time_dependent_component_dirichlet(
    target,
    component: int,
    marker=None,
    value=None,
    *,
    amplitude=None,
    location=None,
    on=None,
    name: str = "time_dependent_dirichlet",
) -> TimeDependentDirichlet:
    """Create a component-wise Dirichlet constraint driven by an amplitude."""

    selected_location = _select_location(location=location, on=on)
    selected_amplitude = amplitude if amplitude is not None else value
    if selected_amplitude is None:
        raise ValueError("time_dependent_component_dirichlet requires value= or amplitude=.")
    history = amplitudes.as_amplitude(selected_amplitude, name=name)
    constant, bc = boundary.component_dirichlet_bc(
        target,
        component,
        marker,
        value=0.0,
        location=selected_location,
    )
    return TimeDependentDirichlet(
        constant=constant,
        bc=bc,
        amplitude=history,
        name=name,
        location=selected_location,
    )


def time_dependent_scalar_dirichlet(
    target,
    marker=None,
    value=None,
    *,
    amplitude=None,
    location=None,
    on=None,
    name: str = "time_dependent_dirichlet",
) -> TimeDependentDirichlet:
    """Create a scalar Dirichlet constraint driven by an amplitude."""

    selected_location = _select_location(location=location, on=on)
    selected_amplitude = amplitude if amplitude is not None else value
    if selected_amplitude is None:
        raise ValueError("time_dependent_scalar_dirichlet requires value= or amplitude=.")
    history = amplitudes.as_amplitude(selected_amplitude, name=name)
    constant, bc = boundary.scalar_dirichlet_bc(
        target,
        marker,
        value=0.0,
        location=selected_location,
    )
    return TimeDependentDirichlet(
        constant=constant,
        bc=bc,
        amplitude=history,
        name=name,
        location=selected_location,
    )


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    boundary.apply_dirichlet_bcs(function, bcs)


def fixed(
    target,
    *,
    location=None,
    on=None,
    value=0.0,
    components: int | tuple[int, ...] | list[int] | None = None,
    name: str | None = None,
) -> "ConstraintSet":
    """Create fixed-value Dirichlet constraints for an application field.

    For scalar fields, ``components=None`` creates one scalar Dirichlet
    condition. For vector fields, ``components=None`` fixes every component.
    Provide one or more component indices when only selected dofs should be
    constrained.
    """

    selected_location = _select_location(location=location, on=on)
    if selected_location is None:
        raise ValueError("fixed requires a geometric location. Pass on=... or location=....")
    label = name or f"fixed_{getattr(selected_location, 'name', 'location')}"
    if components is None:
        components = _all_components_or_none(target)

    if components is None:
        return ConstraintSet(
            dirichlet=[
                scalar_dirichlet(target, location=selected_location, value=value, name=label),
            ]
        )

    component_ids = _component_ids(components, target=target)
    component_values = _component_values(value, len(component_ids))
    return ConstraintSet(
        dirichlet=[
            component_dirichlet(
                target,
                component,
                location=selected_location,
                value=component_value,
                name=f"{label}_component_{component}",
            )
            for component, component_value in zip(component_ids, component_values)
        ]
    )


def fixed_component(
    target,
    component: int,
    *,
    location=None,
    on=None,
    value=0.0,
    name: str | None = None,
):
    """Create a fixed-value constraint for one vector component."""

    return fixed(target, location=location, on=on, value=value, components=component, name=name)


def symmetry(
    target,
    *,
    on=None,
    location=None,
    normal_axis: int | str,
    value=0.0,
    name: str | None = None,
) -> "ConstraintSet":
    """Apply an axis-aligned solid-mechanics symmetry condition.

    For displacement-only solid elements, symmetry fixes the displacement
    component normal to the plane. Arbitrary inclined symmetry planes require
    a linear multi-point constraint and are deliberately not approximated here.
    """

    component = _axis_component(normal_axis)
    available = _all_components_or_none(target)
    if available is None or component not in available:
        raise ValueError(
            f"normal_axis={normal_axis!r} selects component {component}, "
            f"but the target provides components={available!r}."
        )
    selected_location = _select_location(location=location, on=on)
    label = name or f"symmetry_{'xyz'[component]}"
    return fixed(
        target,
        location=selected_location,
        value=value,
        components=component,
        name=label,
    )


def roller(
    target,
    *,
    on=None,
    location=None,
    normal_axis: int | str,
    value=0.0,
    name: str | None = None,
) -> "ConstraintSet":
    """Alias for an axis-aligned frictionless roller/support condition."""

    return symmetry(
        target,
        on=on,
        location=location,
        normal_axis=normal_axis,
        value=value,
        name=name or f"roller_{normal_axis}",
    )


def fixed_all(target, *, location=None, on=None, value=0.0, name: str | None = None):
    """Create a scalar/all-dof fixed-value constraint."""

    return fixed(target, location=location, on=on, value=value, components=None, name=name)


def prescribed(
    target,
    *,
    on=None,
    location=None,
    value=0.0,
    component=None,
    components=None,
    name: str | None = None,
):
    """Create prescribed scalar or vector-component values.

    This engineering spelling is equivalent to ``fixed`` but reads naturally
    for non-zero displacement and temperature boundary data.
    """

    if component is not None:
        if components is not None:
            raise ValueError("Pass either component=... or components=..., not both.")
        components = component
    return fixed(
        target,
        on=on,
        location=location,
        value=value,
        components=components,
        name=name or "prescribed",
    )


def clamped(target, *, on=None, location=None, value=0.0, name: str | None = None):
    """Fix every displacement component on a support boundary."""

    available = _all_components_or_none(target)
    if available is None:
        raise ValueError("clamped requires a vector displacement-like field.")
    return fixed(
        target,
        on=on,
        location=location,
        value=value,
        components=available,
        name=name or "clamped",
    )


def prescribed_temperature(
    target,
    value,
    *,
    on=None,
    location=None,
    name: str | None = None,
):
    """Prescribe temperature on a named boundary."""

    if _all_components_or_none(target) is not None:
        raise ValueError("prescribed_temperature requires a scalar field.")
    return fixed(
        target,
        on=on,
        location=location,
        value=value,
        components=None,
        name=name or "prescribed_temperature",
    )


def remote_displacement(
    target,
    *,
    reference_point,
    on=None,
    location=None,
    translation=None,
    rotation=None,
    system=None,
    name: str = "remote_displacement",
) -> RemoteDisplacementConstraint:
    """Prescribe rigid translation/rotation of a solid boundary.

    This is a known-motion kinematic coupling.  It does not introduce an
    unknown reference-point degree of freedom and therefore remains a standard
    strong boundary condition with deterministic load-path scaling.
    """

    selected = _select_location(location=location, on=on)
    if selected is None:
        raise ValueError("remote_displacement requires on=... or location=....")
    value = getattr(target, "value", target)
    space = value.function_space if hasattr(value, "function_space") else target
    shape = tuple(getattr(value, "ufl_shape", ()))
    if len(shape) != 1 or shape[0] not in {2, 3}:
        raise ValueError("remote_displacement requires a 2D or 3D vector space.")
    dimension = int(shape[0])
    point = np.asarray(
        getattr(reference_point, "coordinates", reference_point), dtype=float
    ).reshape(-1)
    if point.size != dimension or not np.all(np.isfinite(point)):
        raise ValueError(f"reference_point must have {dimension} finite components.")
    translation_values = np.zeros(dimension) if translation is None else np.asarray(
        system.vector_to_global(translation) if system is not None else translation,
        dtype=float,
    ).reshape(-1)
    if translation_values.size != dimension or not np.all(np.isfinite(translation_values)):
        raise ValueError(f"translation must have {dimension} finite components.")
    if dimension == 2:
        rotation_value = float(0.0 if rotation is None else np.asarray(rotation).reshape(-1)[0])
    else:
        rotation_value = np.zeros(3) if rotation is None else np.asarray(
            system.vector_to_global(rotation) if system is not None else rotation,
            dtype=float,
        ).reshape(-1)
        if rotation_value.size != 3 or not np.all(np.isfinite(rotation_value)):
            raise ValueError("3D rotation must have three finite components.")

    function = fem.Function(space, name=name)

    def rigid_motion(x):
        arm = x[:dimension] - point[:, None]
        if dimension == 2:
            rotational = rotation_value * np.vstack((-arm[1], arm[0]))
        else:
            rotational = np.cross(
                np.broadcast_to(rotation_value, (x.shape[1], 3)),
                arm.T,
            ).T
        return translation_values[:, None] + rotational

    function.interpolate(rigid_motion)
    function.x.scatter_forward()
    selected_dofs = dofs.locate_dofs(space, selected)
    bc = fem.dirichletbc(function, selected_dofs)
    stored_rotation = (
        float(rotation_value)
        if dimension == 2
        else tuple(float(value) for value in rotation_value)
    )
    return RemoteDisplacementConstraint(
        bc=bc,
        value=function,
        reference_values=function.x.array.copy(),
        reference_point=reference_point,
        translation=tuple(float(value) for value in translation_values),
        rotation=stored_rotation,
        name=name,
        location=selected,
        coordinate_system=getattr(system, "name", None),
    )


def _all_components_or_none(target) -> tuple[int, ...] | None:
    """Return vector component ids, or ``None`` for a scalar target."""

    value = getattr(target, "value", target)
    shape = getattr(value, "ufl_shape", ())
    if len(shape) == 0:
        return None
    if len(shape) != 1:
        raise ValueError(
            "Automatic fixed constraints only support scalar or vector fields. "
            "Pass components explicitly for tensor-valued targets."
        )
    return tuple(range(int(shape[0])))


def _component_ids(components, *, target) -> tuple[int, ...]:
    """Normalize integer or x/y/z component names for a vector target."""

    selected = (
        (components,)
        if isinstance(components, (Integral, str))
        else tuple(components)
    )
    normalized = tuple(
        _axis_component(item) if isinstance(item, str) else int(item)
        for item in selected
    )
    available = _all_components_or_none(target)
    if available is None:
        raise ValueError("Scalar targets do not accept component selection.")
    invalid = tuple(item for item in normalized if item not in available)
    if invalid:
        raise ValueError(
            f"components={invalid!r} are unavailable; target components are "
            f"{available!r}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("components must not contain duplicates.")
    return normalized


def _component_values(value, count: int) -> tuple:
    """Expand a scalar or component-wise value into one value per component."""

    if _is_scalar_value(value):
        return tuple(value for _ in range(count))
    try:
        values = tuple(value)
    except TypeError:
        return tuple(value for _ in range(count))
    if len(values) != count:
        raise ValueError(
            f"Expected {count} component values for fixed constraint, got {len(values)}."
        )
    return values


def _is_scalar_value(value) -> bool:
    return isinstance(value, (str, bytes, Real)) or not hasattr(value, "__len__")


def _select_location(*, location=None, on=None):
    if location is not None and on is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    return location if location is not None else on


def _axis_component(axis: int | str) -> int:
    if isinstance(axis, str):
        normalized = axis.lower().strip()
        names = {"x": 0, "y": 1, "z": 2}
        if normalized not in names:
            raise ValueError("normal_axis must be x, y, z, 0, 1, or 2.")
        return names[normalized]
    selected = int(axis)
    if selected not in {0, 1, 2}:
        raise ValueError("normal_axis must be x, y, z, 0, 1, or 2.")
    return selected


def _space(target):
    if hasattr(target, "space"):
        return target.space
    if hasattr(target, "function_space"):
        return target.function_space
    if hasattr(target, "value") and hasattr(target.value, "function_space"):
        return target.value.function_space
    return target


def _region_marker(location):
    marker = getattr(location, "marker", location)
    if marker is None:
        raise ValueError("Periodic constraints require master/slave markers or regions.")
    return marker


def _axis_id(axis: str | int, gdim: int) -> int:
    if isinstance(axis, str):
        names = {"x": 0, "y": 1, "z": 2}
        key = axis.lower()
        if key not in names:
            raise ValueError("match_axis must be 'x', 'y', 'z', or an integer.")
        axis_id = names[key]
    else:
        axis_id = int(axis)
    if axis_id < 0 or axis_id >= int(gdim):
        raise ValueError(f"match_axis {axis!r} is outside geometric dimension {gdim}.")
    return axis_id


@dataclass(frozen=True)
class PeriodicProjectionConstraint:
    """Projection-style periodic constraint for explicit field updates.

    This method enforces equality by averaging paired dof values. It is useful
    for serial explicit dynamics workflows, but it is not a strict MPC
    constraint and does not currently support distributed meshes.
    """

    pairs: tuple[tuple[np.ndarray, np.ndarray], ...]
    pair_count: int
    name: str = "periodic_projection"
    master: object | None = None
    slave: object | None = None
    match_axis: str | int = 0
    supports_parallel: bool = False
    tolerance: float = 1.0e-12
    maximum_coordinate_mismatch: float = 0.0

    @property
    def capabilities(self) -> ConstraintCapabilities:
        return ConstraintCapabilities(
            kind="periodic_constraint",
            enforcement="nodal_pair_projection",
            analyses=("second_order_dynamics",),
            procedures=("central_difference",),
            strict=False,
            supports_parallel=False,
            reaction_evidence="unavailable",
            work_evidence="unavailable",
        )

    def apply(self, function) -> None:
        """Apply periodic equality by averaging paired dof values."""

        from .. import fields

        function = fields.unwrap(function)
        values = function.x.array
        for slave_dofs, master_dofs in self.pairs:
            averaged = 0.5 * (values[slave_dofs] + values[master_dofs])
            values[slave_dofs] = averaged
            values[master_dofs] = averaged
        function.x.scatter_forward()

    def __call__(self, function) -> None:
        """Callable alias for use by time integrators."""

        self.apply(function)

    def mismatch(self, function) -> float:
        """Return the max absolute paired-dof mismatch."""

        from .. import fields

        function = fields.unwrap(function)
        values = function.x.array
        local = 0.0
        for slave_dofs, master_dofs in self.pairs:
            if len(slave_dofs) > 0:
                local = max(local, float(np.max(np.abs(values[slave_dofs] - values[master_dofs]))))
        return function.function_space.mesh.comm.allreduce(local, op=MPI.MAX)

    def summary(self) -> dict[str, object]:
        """Return method and limitation details for logs or agents."""

        return {
            "name": self.name,
            "kind": "periodic_constraint",
            "method": "projection",
            "enforcement": "nodal_pair_averaging",
            "pair_count": self.pair_count,
            "master": getattr(self.master, "name", None),
            "slave": getattr(self.slave, "name", None),
            "match_axis": self.match_axis,
            "tolerance": self.tolerance,
            "maximum_coordinate_mismatch": self.maximum_coordinate_mismatch,
            "unmatched_dofs": 0,
            "strict": False,
            "supports_parallel": self.supports_parallel,
            "capabilities": self.capabilities.summary(),
        }

    def diagnostics(self, function=None) -> dict[str, object]:
        """Return construction and optional live field mismatch evidence."""

        values = {
            "constraint": self.name,
            "method": "projection",
            "pair_count": self.pair_count,
            "unmatched_dofs": 0,
            "maximum_coordinate_mismatch": self.maximum_coordinate_mismatch,
            "strict": False,
            "supports_parallel": False,
        }
        if function is not None:
            values["maximum_field_mismatch"] = self.mismatch(function)
        return values


def periodic(
    target,
    *,
    master,
    slave,
    match_axis: str | int = 0,
    method: str = "projection",
    tolerance: float = 1.0e-12,
    name: str = "periodic",
):
    """Create a periodic constraint with an explicit method choice."""

    normalized = method.lower().replace("-", "_")
    if normalized in {"projection", "nodal_projection"}:
        return periodic_projection(
            target,
            master=master,
            slave=slave,
            match_axis=match_axis,
            tolerance=tolerance,
            name=name,
        )
    if normalized in {"mpc", "multi_point_constraint"}:
        raise NotImplementedError(
            "Automatic exact-MPC construction from arbitrary master/slave "
            "markers is not yet available. For rectangular domains, use "
            "constraints.rectangular_periodic_mpc(...); linear-static and "
            "steady-heat model.step() providers lower that asset through the "
            "shared exact-MPC solver. Use method='projection' only for serial "
            "explicit central-difference workflows."
        )
    raise ValueError(f"Unknown periodic constraint method: {method!r}.")


def periodic_projection(
    target,
    *,
    master,
    slave,
    match_axis: str | int = 0,
    tolerance: float = 1.0e-12,
    name: str = "periodic_projection",
) -> PeriodicProjectionConstraint:
    """Create component-wise dof pairs for projection-style periodicity."""

    V = _space(target)
    domain = V.mesh
    if domain.comm.size > 1:
        raise RuntimeError(
            "Projection-style periodic constraints are serial-only in this release. "
            "Use method='mpc' when a parallel implementation is added."
        )
    master_marker = _region_marker(master)
    slave_marker = _region_marker(slave)
    axis_id = _axis_id(match_axis, domain.geometry.dim)
    pairs = []
    pair_count = 0
    maximum_coordinate_mismatch = 0.0
    components = range(V.num_sub_spaces) if getattr(V, "num_sub_spaces", 0) else (None,)
    for component in components:
        if component is None:
            coords = V.tabulate_dof_coordinates()
            slave_parent = fem.locate_dofs_geometrical(V, slave_marker)
            master_parent = fem.locate_dofs_geometrical(V, master_marker)
            slave_child = slave_parent
            master_child = master_parent
        else:
            Vc, _ = V.sub(component).collapse()
            coords = Vc.tabulate_dof_coordinates()
            slave_parent, slave_child = fem.locate_dofs_geometrical(
                (V.sub(component), Vc), slave_marker
            )
            master_parent, master_child = fem.locate_dofs_geometrical(
                (V.sub(component), Vc), master_marker
            )
        if len(slave_child) != len(master_child):
            raise RuntimeError(
                "Periodic projection requires matching slave/master dofs "
                f"for component {component}."
            )

        slave_order = np.argsort(coords[slave_child, axis_id])
        master_order = np.argsort(coords[master_child, axis_id])
        slave_parent = np.asarray(slave_parent[slave_order], dtype=np.int32)
        master_parent = np.asarray(master_parent[master_order], dtype=np.int32)
        slave_coords = coords[slave_child[slave_order], axis_id]
        master_coords = coords[master_child[master_order], axis_id]
        mismatch = (
            0.0
            if len(slave_coords) == 0
            else float(np.max(np.abs(slave_coords - master_coords)))
        )
        maximum_coordinate_mismatch = max(maximum_coordinate_mismatch, mismatch)
        if not np.allclose(slave_coords, master_coords, atol=tolerance, rtol=0.0):
            raise RuntimeError(
                "Periodic projection dofs are not aligned on match_axis "
                f"{match_axis!r}: max mismatch={mismatch:.3e}."
            )
        pairs.append((slave_parent, master_parent))
        pair_count += len(slave_parent)
    return PeriodicProjectionConstraint(
        pairs=tuple(pairs),
        pair_count=pair_count,
        name=name,
        master=master,
        slave=slave,
        match_axis=match_axis,
        tolerance=float(tolerance),
        maximum_coordinate_mismatch=maximum_coordinate_mismatch,
    )


def constraint_capabilities(constraint) -> ConstraintCapabilities | None:
    """Return the public capability contract of a known constraint asset."""

    selected = getattr(constraint, "capabilities", None)
    if callable(selected):
        selected = selected()
    if isinstance(selected, ConstraintCapabilities):
        return selected
    if isinstance(
        constraint,
        (DirichletConstraint, TimeDependentDirichlet, RemoteDisplacementConstraint),
    ) or hasattr(constraint, "dof_indices"):
        return ConstraintCapabilities(
            kind="dirichlet_constraint",
            enforcement="strong_elimination",
            reaction_evidence="unconstrained_residual",
            work_evidence=(
                "proportional_prescribed_path"
                if getattr(constraint, "value", None) is not None
                or getattr(constraint, "constant", None) is not None
                else "uninspectable_prescribed_value"
            ),
        )
    if isinstance(constraint, RectangularPeriodicMPC):
        return ConstraintCapabilities(
            kind="periodic_constraint",
            enforcement="exact_multi_point_constraint",
            analyses=("linear_static", "first_order_transient"),
            strict=True,
            supports_parallel=True,
            reaction_evidence="provider_dual_required",
            work_evidence="provider_dual_path_required",
        )
    if isinstance(constraint, AbaqusPeriodicConstraint):
        return ConstraintCapabilities(
            kind="periodic_constraint",
            enforcement="exact_affine_elimination",
            strict=True,
            supports_parallel=bool(constraint.summary()["supports_parallel"]),
            reaction_evidence="provider_dual_required",
            work_evidence="provider_dual_path_required",
        )
    if isinstance(constraint, PeriodicConstraintSpec):
        return ConstraintCapabilities(
            kind="periodic_constraint",
            enforcement="unresolved_periodic_specification",
            strict=False,
            supports_parallel=False,
            reaction_evidence="unavailable",
            work_evidence="unavailable",
        )
    return None


def constraint_balance_contract(constraints, *, provider_duals=()) -> dict[str, object]:
    """Describe whether strong-reaction force/work diagnostics are complete.

    Strong Dirichlet elimination exposes reactions through the unconstrained
    residual.  MPC, weak, contact, projection, and unknown constraint assets
    require their own dual variables; this contract prevents those forces from
    being silently omitted from a global balance reported as complete.
    """

    dual_records = tuple(provider_duals)
    if any(not isinstance(item, ConstraintDualEvidence) for item in dual_records):
        raise TypeError("provider_duals must contain ConstraintDualEvidence records.")
    duals = {item.constraint_name: item for item in dual_records}
    if len(duals) != len(dual_records):
        raise ValueError("Provider dual constraint names must be unique.")
    records = []
    force_gaps = []
    work_gaps = []
    for item in _flatten_constraint_assets(constraints):
        capability = constraint_capabilities(item)
        name = str(getattr(item, "name", type(item).__name__))
        if capability is None:
            summary = {
                "name": name,
                "kind": type(item).__name__,
                "enforcement": "unknown",
                "reaction_evidence": "unavailable",
                "work_evidence": "unavailable",
            }
        else:
            summary = {"name": name, **capability.summary()}
        dual = duals.get(name)
        if dual is not None:
            summary["provider_dual"] = dual.summary()
        records.append(summary)
        force_available = summary["reaction_evidence"] == "unconstrained_residual"
        work_available = summary["work_evidence"] == "proportional_prescribed_path"
        if summary["reaction_evidence"] == "provider_dual_required":
            force_available = bool(dual is not None and dual.force_complete)
        if summary["work_evidence"] == "provider_dual_path_required":
            work_available = bool(dual is not None and dual.work_complete)
        if not force_available:
            force_gaps.append(name)
        if not work_available:
            work_gaps.append(name)
    names = {record["name"] for record in records}
    if len(names) != len(records):
        raise ValueError(
            "Constraint names must be unique when assembling balance evidence."
        )
    unexpected = tuple(sorted(set(duals) - names))
    if unexpected:
        raise ValueError(
            "Provider dual evidence does not match declared constraints: "
            f"{unexpected!r}."
        )
    return {
        "kind": "constraint_balance_contract",
        "reaction_scope": (
            "all declared constraints"
            if not force_gaps
            else "available constraint channels"
        ),
        "work_scope": (
            "all declared constraints"
            if not work_gaps
            else "available constraint channels"
        ),
        "force_balance_available": not force_gaps,
        "work_balance_available": not work_gaps,
        "force_balance_gaps": tuple(force_gaps),
        "work_balance_gaps": tuple(work_gaps),
        "constraints": tuple(records),
        "provider_duals": tuple(item.summary() for item in dual_records),
        "unexpected_provider_duals": unexpected,
    }


def validate_solver_compatibility(
    *,
    constraints,
    analysis: str,
    procedure: str | None = None,
    comm_size: int = 1,
):
    """Validate constraint/procedure compatibility before assembly or solve."""

    from ..validation import ValidationReport, issue

    normalized_analysis = str(analysis).lower().replace("-", "_").strip()
    normalized_procedure = (
        None
        if procedure is None
        else str(procedure).lower().replace("-", "_").strip()
    )
    aliases = {
        "explicit": "central_difference",
        "implicit": "newmark",
        "generalizedalpha": "generalized_alpha",
    }
    normalized_procedure = aliases.get(normalized_procedure, normalized_procedure)
    issues = []
    for index, constraint in enumerate(_flatten_constraint_assets(constraints)):
        capability = constraint_capabilities(constraint)
        if capability is None:
            continue
        path = f"model.constraints[{index}]"
        if capability.analyses and normalized_analysis not in capability.analyses:
            issues.append(
                issue(
                    "AFM-CONSTRAINT-ANALYSIS-001",
                    path,
                    (
                        f"{type(constraint).__name__} cannot enforce "
                        f"analysis={normalized_analysis!r}."
                    ),
                    hint=(
                        "Choose a compatible enforcement backend; periodic nodal "
                        "projection is intended for explicit central difference."
                    ),
                    constraint=capability.summary(),
                )
            )
        elif (
            capability.procedures
            and normalized_procedure is not None
            and normalized_procedure not in capability.procedures
        ):
            issues.append(
                issue(
                    "AFM-CONSTRAINT-PROCEDURE-001",
                    path,
                    (
                        f"{type(constraint).__name__} with "
                        f"enforcement={capability.enforcement!r} does not support "
                        f"procedure={normalized_procedure!r}."
                    ),
                    hint=(
                        "Use central_difference with periodic projection, or an "
                        "exact affine/MPC constraint supported by the implicit solver."
                    ),
                    constraint=capability.summary(),
                )
            )
        if int(comm_size) > 1 and not capability.supports_parallel:
            issues.append(
                issue(
                    "AFM-CONSTRAINT-PARALLEL-001",
                    path,
                    f"{type(constraint).__name__} is serial-only.",
                    hint="Use a verified distributed affine/MPC backend or run serially.",
                    mpi_ranks=int(comm_size),
                    constraint=capability.summary(),
                )
            )
        if (
            isinstance(constraint, TimeDependentDirichlet)
            and normalized_analysis == "second_order_dynamics"
            and normalized_procedure in {"newmark", "generalized_alpha"}
        ):
            issues.append(
                issue(
                    "AFM-CONSTRAINT-002",
                    path,
                    "Implicit structural dynamics requires consistent prescribed displacement, velocity, and acceleration histories.",
                    hint="Use central difference or an expert consistent-support formulation.",
                )
            )
    return ValidationReport.from_issues(issues, scope="constraint_compatibility")


def _flatten_constraint_assets(items) -> tuple[object, ...]:
    if items is None:
        return ()
    if isinstance(items, (list, tuple)):
        return tuple(
            selected
            for item in items
            for selected in _flatten_constraint_assets(item)
        )
    if isinstance(items, ConstraintSet):
        return _flatten_constraint_assets((*items.dirichlet, *items.periodic))
    return (items,)


@dataclass(frozen=True)
class PeriodicConstraintSpec:
    """Geometric description of a periodic constraint.

    This is a lightweight specification. Concrete implementations may use
    explicit nodal projection, MPC, or another backend.
    """

    slave_marker: object
    master_marker: object
    map_slave_to_master: object
    name: str = "periodic"

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {"name": self.name, "kind": "periodic_constraint"}


@dataclass
class ConstraintSet:
    """Collection of constraints used by assembly or field updates."""

    dirichlet: list[DirichletConstraint] = field(default_factory=list)
    periodic: list[PeriodicConstraintSpec] = field(default_factory=list)

    @property
    def bcs(self) -> list:
        """Return DOLFINx Dirichlet BC objects."""

        return [constraint.bc for constraint in self.dirichlet]

    def add_dirichlet(self, constraint: DirichletConstraint) -> None:
        self.dirichlet.append(constraint)

    def add_periodic(self, constraint: PeriodicConstraintSpec) -> None:
        self.periodic.append(constraint)

    def extend(self, other: "ConstraintSet") -> None:
        """Append constraints from another set."""

        self.dirichlet.extend(other.dirichlet)
        self.periodic.extend(other.periodic)

    def summary(self) -> dict[str, object]:
        """Return compact descriptions of all constraints."""

        return {
            "dirichlet": tuple(constraint.summary() for constraint in self.dirichlet),
            "periodic": tuple(constraint.summary() for constraint in self.periodic),
        }
