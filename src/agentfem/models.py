"""Lightweight model registries for AgentFEM workflows.

The model layer records mesh, fields, amplitudes, materials, constraints,
loads, and boundary models. It is an audit and validation object, not a solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from . import constraints as constraint_api
from . import loads as load_api
from .materials.definitions import MaterialDefinition
from .materials.properties import constant_volumetric_heat_capacity
from ._api_contract import (
    ADVANCED_MODEL_API,
    COMPATIBILITY_MODEL_API,
    CORE_MODEL_API,
    model_method_contract as _model_method_contract,
    model_methods as _model_methods,
)
from .step_providers import (
    StepExecutionPolicy,
    StepOptionContract,
    StepProvider,
    StepProviderRegistry,
    register_step_provider,
    step_capability,
    step_providers,
)


def model_api(level: str = "core") -> tuple[str, ...]:
    """Return the recommended Model vocabulary at one discovery level.

    This does not remove 0.2.x compatibility methods. It gives documentation,
    agents, IDE integrations, and future GUIs a deterministic way to present
    the concise engineering language before expert builders and historical
    aliases.
    """

    return _model_methods(level)


def model_api_contract(level: str = "all") -> tuple[dict[str, object], ...]:
    """Return machine-readable lifecycle metadata for Model methods."""

    return _model_method_contract(level)


@dataclass
class Model:
    """Finite-element model registry for humans and agents."""

    study: object
    mesh: object | None = None
    name: str = "model"
    fields: list[object] = field(default_factory=list)
    amplitudes: list[object] = field(default_factory=list)
    materials: list[object] = field(default_factory=list)
    constraints: list[object] = field(default_factory=list)
    loads: list[object] = field(default_factory=list)
    boundary_models: list[object] = field(default_factory=list)
    regions: list[object] = field(default_factory=list)
    steps: list[object] = field(default_factory=list)
    engineering_steps: list[object] = field(default_factory=list)
    unit_system: object | None = None

    @property
    def domain(self):
        """Return the underlying DOLFINx domain for low-level operations."""

        return _domain(self.mesh)

    def add_field(self, field_object):
        """Register an unknown or output field and return it."""

        self.fields.append(field_object)
        return field_object

    def field(self, field_object):
        """Register an unknown or output field and return it."""

        return self.add_field(field_object)

    def add_amplitude(self, amplitude, *, name: str | None = None):
        """Register a time-history/scale-factor asset and return it."""

        item = (
            amplitude.renamed(name)
            if name is not None and hasattr(amplitude, "renamed")
            else amplitude
        )
        self.amplitudes.append(item)
        return item

    def amplitude(self, name_or_amplitude, amplitude=None):
        """Register or retrieve a named amplitude."""

        if amplitude is None and isinstance(name_or_amplitude, str):
            return self._amplitude_by_name(name_or_amplitude)
        if amplitude is None:
            return self.add_amplitude(name_or_amplitude)
        return self.add_amplitude(amplitude, name=name_or_amplitude)

    def add_material(self, material, *, region=None):
        """Register material data and return the executable behavior.

        A named ``materials.MaterialDefinition`` remains independent of the
        Study until this boundary.  Registration resolves its physics role,
        validates minimum Study requirements, and keeps the original
        definition beside the assignment for provenance.
        """

        self._register_region(region)
        definition = None
        resolved = material
        if isinstance(material, MaterialDefinition):
            definition = material
            resolved = material.resolve_for(self.study)
        self.materials.append(_WithRegion(resolved, region, definition))
        return resolved

    def material(self, material, *, region=None):
        """Register material data and return it."""

        return self.add_material(material, region=region)

    def add_constraint(self, constraint):
        """Register a strong constraint and return it."""

        self._register_regions_from_asset(constraint)
        self.constraints.append(constraint)
        return constraint

    def constraint(self, constraint):
        """Register a strong constraint and return it."""

        return self.add_constraint(constraint)

    def fix(
        self,
        target,
        *,
        on=None,
        location=None,
        value=0.0,
        component=None,
        components=None,
        name: str | None = None,
    ):
        """Create and register fixed-value Dirichlet constraints."""

        if component is not None:
            if components is not None:
                raise ValueError("Pass either component=... or components=..., not both.")
            components = component
        selected_value = (
            self._amplitude_by_name(value)
            if isinstance(value, str)
            else value
        )
        if _is_amplitude_like(selected_value):
            created = _time_dependent_fix(
                target,
                on=on,
                location=location,
                value=selected_value,
                components=components,
                name=name,
            )
            for item in constraint_api.dirichlet_constraints(created):
                history = getattr(item, "amplitude", None)
                if history is not None and not any(
                    registered is history for registered in self.amplitudes
                ):
                    self.add_amplitude(history)
            return self.add_constraint(created)
        return self.add_constraint(
            constraint_api.fixed(
                target,
                on=on,
                location=location,
                value=selected_value,
                components=components,
                name=name,
            )
        )

    def prescribe(
        self,
        target,
        value,
        *,
        on=None,
        location=None,
        component=None,
        components=None,
        name: str | None = None,
    ):
        """Register prescribed displacement, temperature, or scalar data."""

        return self.fix(
            target,
            on=on,
            location=location,
            value=value,
            component=component,
            components=components,
            name=name or "prescribed",
        )

    def remote_displacement(
        self,
        target,
        *,
        reference_point,
        on=None,
        location=None,
        translation=None,
        rotation=None,
        system=None,
        name: str = "remote_displacement",
    ):
        """Prescribe a rigid boundary motion about a reference point."""

        return self.add_constraint(
            constraint_api.remote_displacement(
                target,
                reference_point=reference_point,
                on=on,
                location=location,
                translation=translation,
                rotation=rotation,
                system=system,
                name=name,
            )
        )

    def clamp(
        self,
        target,
        *,
        on=None,
        location=None,
        value=0.0,
        name: str | None = None,
    ):
        """Clamp every component of a displacement-like field."""

        return self.add_constraint(
            constraint_api.clamped(
                target,
                on=on,
                location=location,
                value=value,
                name=name,
            )
        )

    def prescribed_temperature(
        self,
        target,
        value,
        *,
        on=None,
        location=None,
        name: str | None = None,
    ):
        """Register an essential temperature boundary condition."""

        return self.fix(
            target,
            value=value,
            on=on,
            location=location,
            name=name or "prescribed_temperature",
        )

    def periodic(
        self,
        target,
        *,
        master,
        slave,
        match_axis=0,
        method: str = "projection",
        tolerance: float = 1.0e-12,
        name: str = "periodic",
    ):
        """Create and register a periodic constraint."""

        return self.add_constraint(
            constraint_api.periodic(
                target,
                master=master,
                slave=slave,
                match_axis=match_axis,
                method=method,
                tolerance=tolerance,
                name=name,
            )
        )

    def add_load(self, load):
        """Register a natural load/source and return it."""

        self._register_regions_from_asset(load)
        self.loads.append(load)
        return load

    def _with_amplitude(self, load, amplitude):
        if amplitude is None:
            return load
        selected = (
            self._amplitude_by_name(amplitude)
            if isinstance(amplitude, str)
            else amplitude
        )
        driven = load_api.with_amplitude(
            load,
            selected,
            domain=_domain(self.mesh),
            name=getattr(load, "name", None),
        )
        history = driven.amplitude
        if not any(item is history for item in self.amplitudes):
            self.add_amplitude(history)
        return driven

    def load(self, load):
        """Register a natural load/source and return it."""

        return self.add_load(load)

    def traction(
        self,
        value,
        *,
        on=None,
        location=None,
        amplitude=None,
        system=None,
        name: str = "traction",
    ):
        """Create and register a mechanical traction load."""

        load = load_api.traction(
            value, on=on, location=location, system=system, name=name
        )
        return self.add_load(
            self._with_amplitude(load, amplitude)
        )

    def surface_force(
        self,
        resultant,
        *,
        on=None,
        location=None,
        reference_measure=None,
        amplitude=None,
        system=None,
        name: str = "surface_force",
    ):
        """Uniformly distribute a requested total force over a boundary."""

        load = load_api.surface_force(
            resultant,
            on=on,
            location=location,
            reference_measure=reference_measure,
            study=self.study,
            system=system,
            name=name,
        )
        return self.add_load(self._with_amplitude(load, amplitude))

    def distributing_coupling(
        self,
        force,
        *,
        moment=None,
        reference_point=None,
        on=None,
        location=None,
        amplitude=None,
        system=None,
        name: str = "distributing_coupling",
    ):
        """Distribute a reference-point force/moment over a solid surface."""

        if getattr(self.study, "assumption", None) == "axisymmetric":
            raise NotImplementedError(
                "Axisymmetric distributing coupling needs an explicit ring/reference "
                "kinematic definition and is not yet supported. Use surface_force "
                "for a uniform physical resultant."
            )

        load = load_api.distributing_coupling(
            force,
            moment=moment,
            reference_point=reference_point,
            on=on,
            location=location,
            system=system,
            name=name,
        )
        return self.add_load(self._with_amplitude(load, amplitude))

    def remote_force(
        self,
        force,
        *,
        reference_point,
        moment=None,
        on=None,
        location=None,
        amplitude=None,
        system=None,
        name: str = "remote_force",
    ):
        """Apply a reference-point resultant through a continuum surface."""

        if getattr(self.study, "assumption", None) == "axisymmetric":
            raise NotImplementedError(
                "Axisymmetric remote force needs an explicit ring/reference "
                "kinematic definition and is not yet supported. Use surface_force "
                "for a uniform physical resultant."
            )

        load = load_api.remote_force(
            force,
            reference_point=reference_point,
            moment=moment,
            on=on,
            location=location,
            system=system,
            name=name,
        )
        return self.add_load(self._with_amplitude(load, amplitude))

    def body_force(
        self,
        value,
        *,
        domain=None,
        target=None,
        measure=None,
        amplitude=None,
        system=None,
        name: str = "body_force",
    ):
        """Create and register a mechanical body-force load."""

        kwargs = {
            "domain": domain or self.mesh,
            "target": target,
            "system": system,
            "name": name,
        }
        if measure is not None:
            kwargs["measure"] = measure
        return self.add_load(
            self._with_amplitude(load_api.body_force(value, **kwargs), amplitude)
        )

    def heat_flux(
        self,
        value,
        *,
        on=None,
        location=None,
        amplitude=None,
        name: str = "heat_flux",
    ):
        """Create and register a prescribed heat-flux load."""

        load = load_api.heat_flux(value, on=on, location=location, name=name)
        return self.add_load(self._with_amplitude(load, amplitude))

    def heat_source(
        self,
        value,
        *,
        domain=None,
        target=None,
        measure=None,
        amplitude=None,
        name: str = "heat_source",
    ):
        """Create and register a volumetric heat-source load."""

        kwargs = {"domain": domain or self.mesh, "target": target, "name": name}
        if measure is not None:
            kwargs["measure"] = measure
        return self.add_load(
            self._with_amplitude(load_api.heat_source(value, **kwargs), amplitude)
        )

    def gravity(
        self,
        acceleration,
        *,
        material=None,
        domain=None,
        target=None,
        amplitude=None,
        system=None,
        name: str = "gravity",
    ):
        """Register ``rho g`` using one or all model material regions."""

        records = (
            (self._material_record(material),)
            if material is not None
            else tuple(self.materials)
        )
        if not records:
            raise ValueError("model.gravity requires a material with density.")
        created = []
        for index, record in enumerate(records):
            if not hasattr(record.item, "density") or record.item.density is None:
                raise ValueError(
                    f"Material {_describe(record.item)!r} does not define density."
                )
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Gravity with multiple materials requires a region for every material."
                )
            load = load_api.gravity(
                acceleration,
                density=record.item.density,
                domain=domain or self.mesh,
                target=target,
                region=record.region,
                system=system,
                name=(
                    name
                    if len(records) == 1
                    else f"{name}_{getattr(record.region, 'name', index)}"
                ),
            )
            created.append(
                self.add_load(self._with_amplitude(load, amplitude))
            )
        return created[0] if len(created) == 1 else load_api.LoadSet.create(*created)

    def centrifugal(
        self,
        angular_velocity,
        *,
        center=None,
        material=None,
        domain=None,
        target=None,
        amplitude=None,
        name: str = "centrifugal",
    ):
        """Register density-aware centrifugal loading by material region."""

        records = (
            (self._material_record(material),)
            if material is not None
            else tuple(self.materials)
        )
        if not records:
            raise ValueError("model.centrifugal requires a material with density.")
        created = []
        for index, record in enumerate(records):
            if getattr(record.item, "density", None) is None:
                raise ValueError(
                    f"Material {_describe(record.item)!r} does not define density."
                )
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Centrifugal loading with multiple materials requires regions."
                )
            item = load_api.centrifugal(
                angular_velocity,
                density=record.item.density,
                center=center,
                domain=domain or self.mesh,
                target=target,
                region=record.region,
                name=name if len(records) == 1 else f"{name}_{index}",
            )
            created.append(self.add_load(self._with_amplitude(item, amplitude)))
        return created[0] if len(created) == 1 else load_api.LoadSet.create(*created)

    def pressure(
        self,
        value,
        *,
        on=None,
        location=None,
        configuration: str = "reference",
        displacement=None,
        amplitude=None,
        name: str = "pressure",
    ):
        """Create and register dead or follower pressure."""

        load = load_api.pressure(
            value,
            on=on,
            location=location,
            configuration=configuration,
            displacement=displacement,
            name=name,
        )
        return self.add_load(self._with_amplitude(load, amplitude))

    def hydrostatic_pressure(
        self,
        *,
        density,
        gravity,
        reference_point,
        reference_pressure=0.0,
        on=None,
        location=None,
        clip_at_zero: bool = True,
        configuration: str = "reference",
        displacement=None,
        amplitude=None,
        name: str = "hydrostatic_pressure",
    ):
        """Register pressure varying with elevation from a free surface."""

        load = load_api.hydrostatic_pressure(
            density=density,
            gravity=gravity,
            reference_point=reference_point,
            reference_pressure=reference_pressure,
            on=on,
            location=location,
            clip_at_zero=clip_at_zero,
            configuration=configuration,
            displacement=displacement,
            name=name,
        )
        return self.add_load(self._with_amplitude(load, amplitude))

    def symmetry(
        self,
        target,
        *,
        on=None,
        location=None,
        normal_axis,
        name: str | None = None,
    ):
        """Create and register an axis-aligned solid symmetry condition."""

        return self.add_constraint(
            constraint_api.symmetry(
                target,
                on=on,
                location=location,
                normal_axis=normal_axis,
                name=name,
            )
        )

    def roller(
        self,
        target,
        *,
        on=None,
        location=None,
        normal_axis,
        name: str | None = None,
    ):
        """Create and register an axis-aligned roller/support condition."""

        return self.add_constraint(
            constraint_api.roller(
                target,
                on=on,
                location=location,
                normal_axis=normal_axis,
                name=name,
            )
        )

    def absorbing_boundary(
        self,
        *,
        on,
        density,
        pressure_wave_speed,
        shear_wave_speed=None,
        normal=None,
        mode: str = "normal_shear",
    ):
        """Create and register a common viscous absorbing boundary model."""

        from .boundary_models import absorbing
        from . import mesh as mesh_api

        selected_normal = normal
        if selected_normal is None and mode == "normal_shear":
            selected_normal = mesh_api.facet_normal(self.mesh)
        return self.add_boundary_model(
            absorbing.lysmer_kuhlemeyer_boundary(
                on.measure if hasattr(on, "measure") else on,
                density=density,
                pressure_wave_speed=pressure_wave_speed,
                shear_wave_speed=shear_wave_speed,
                normal=selected_normal,
                mode=mode,
                location=on,
            )
        )

    def convection(
        self,
        *,
        on=None,
        location=None,
        coefficient,
        ambient_temperature,
        name: str = "convection",
    ):
        """Register linear heat exchange with an ambient temperature."""

        from .boundary_models import thermal

        return self.add_boundary_model(
            thermal.convection(
                on=on,
                location=location,
                coefficient=coefficient,
                ambient_temperature=ambient_temperature,
                name=name,
            )
        )

    def elastic_foundation(
        self,
        *,
        on=None,
        location=None,
        stiffness,
        mode: str = "isotropic",
        normal=None,
        name: str = "elastic_foundation",
    ):
        """Register a distributed normal or isotropic spring support."""

        from .boundary_models import mechanical

        return self.add_boundary_model(
            mechanical.elastic_foundation(
                on=on,
                location=location,
                stiffness=stiffness,
                mode=mode,
                normal=normal,
                name=name,
            )
        )

    def stiffness(
        self,
        target,
        material=None,
        *,
        measure=None,
        law=None,
        temperature=None,
        study=None,
        name: str = "K",
    ):
        """Create a stiffness operator from registered material assets.

        This is the model-first path. It delegates every contribution to
        ``agentfem.operators.stiffness`` and combines regional contributions
        explicitly, so the generated operator remains inspectable.
        """

        from . import operators

        selected_study = study or self.study
        if material is not None:
            record = self._material_record(material)
            return _stiffness_from_record(
                target,
                record,
                operators=operators,
                study=selected_study,
                measure=measure,
                law=law,
                temperature=temperature,
                name=name,
            )

        if not self.materials:
            raise ValueError("model.stiffness requires at least one registered material.")
        if measure is not None and len(self.materials) > 1:
            raise ValueError(
                "model.stiffness with multiple materials cannot use one explicit measure. "
                "Pass material=... or let each material use its registered region."
            )
        if len(self.materials) == 1:
            return _stiffness_from_record(
                target,
                self.materials[0],
                operators=operators,
                study=selected_study,
                measure=measure,
                law=law,
                temperature=temperature,
                name=name,
            )

        parts = []
        missing = []
        for record in self.materials:
            if record.region is None:
                missing.append(_describe(record.item))
                continue
            parts.append(
                _stiffness_from_record(
                    target,
                    record,
                    operators=operators,
                    study=selected_study,
                    measure=record.region.measure,
                    law=law,
                    temperature=temperature,
                    name=f"K_{getattr(record.region, 'name', len(parts))}",
                )
            )
        if missing:
            raise ValueError(
                "Multiple-material stiffness requires every material to have a region. "
                f"Materials without regions: {missing}."
            )
        return operators.combine(*parts, name=name, kind="partitioned_stiffness")

    def mass(
        self,
        target,
        material=None,
        *,
        measure=None,
        name: str = "M",
    ):
        """Create a consistent mass operator from registered densities."""

        import ufl

        from . import _axisymmetric
        from . import operators

        weight = _axisymmetric.integration_weight(target, self.study)

        records = (
            (self._material_record(material),)
            if material is not None
            else tuple(self.materials)
        )
        if not records:
            raise ValueError("model.mass requires at least one registered material.")
        if measure is not None and len(records) > 1:
            raise ValueError("Pass material=... when using one explicit mass measure.")
        parts = []
        for index, record in enumerate(records):
            selected_measure = (
                measure
                if measure is not None
                else (
                    record.region.measure
                    if record.region is not None
                    else ufl.dx
                )
            )
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Multiple-material mass requires a region for every material."
                )
            parts.append(
                operators.mass_operator(
                    target,
                    _density(record.item) * weight,
                    measure=selected_measure,
                ).renamed(f"{name}_{index}" if len(records) > 1 else name)
            )
        return (
            parts[0]
            if len(parts) == 1
            else operators.combine(*parts, name=name, kind="partitioned_mass")
        )

    def damping(self, target, coefficient, *, measure=None, name: str = "C"):
        """Create a viscous damping operator."""

        import ufl

        from . import _axisymmetric
        from . import operators

        return operators.damping_operator(
            target,
            coefficient * _axisymmetric.integration_weight(target, self.study),
            measure=ufl.dx if measure is None else measure,
        ).renamed(name)

    def conduction(self, temperature, material=None, *, measure=None, name: str = "K"):
        """Create a region-aware heat-conduction operator.

        A single material may occupy the whole mesh.  Multiple materials must
        each own a cell region, matching the semantics already used by
        :meth:`stiffness` and :meth:`mass`.
        """

        import ufl

        from . import operators

        records = (
            (self._material_record(material),)
            if material is not None
            else tuple(self.materials)
        )
        if not records:
            raise ValueError("model.conduction requires at least one material.")
        if measure is not None and len(records) > 1:
            raise ValueError("Pass material=... when using one explicit conduction measure.")
        parts = []
        for index, record in enumerate(records):
            if not hasattr(record.item, "conductivity"):
                raise ValueError(
                    f"Material {_describe(record.item)!r} does not define conductivity."
                )
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Multiple-material conduction requires a region for every material."
                )
            selected_measure = (
                measure
                if measure is not None
                else (record.region.measure if record.region is not None else ufl.dx)
            )
            parts.append(
                operators.conduction_operator(
                    temperature,
                    record.item.conductivity,
                    measure=selected_measure,
                ).renamed(
                    name if len(records) == 1 else f"{name}_{getattr(record.region, 'name', index)}"
                )
            )
        return (
            parts[0]
            if len(parts) == 1
            else operators.combine(*parts, name=name, kind="partitioned_conduction")
        )

    def heat_capacity(self, temperature, material=None, *, measure=None, name: str = "C"):
        """Create region-aware ``rho c_p`` heat capacity."""

        import ufl

        from . import operators

        records = (
            (self._material_record(material),)
            if material is not None
            else tuple(self.materials)
        )
        if not records:
            raise ValueError("model.heat_capacity requires at least one material.")
        if measure is not None and len(records) > 1:
            raise ValueError("Pass material=... when using one explicit capacity measure.")
        parts = []
        for index, record in enumerate(records):
            capacity = constant_volumetric_heat_capacity(record.item)
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Multiple-material heat capacity requires a region for every material."
                )
            selected_measure = (
                measure
                if measure is not None
                else (record.region.measure if record.region is not None else ufl.dx)
            )
            parts.append(
                operators.capacity_operator(
                    temperature,
                    capacity,
                    measure=selected_measure,
                ).renamed(
                    name if len(records) == 1 else f"{name}_{getattr(record.region, 'name', index)}"
                )
            )
        return (
            parts[0]
            if len(parts) == 1
            else operators.combine(*parts, name=name, kind="partitioned_heat_capacity")
        )

    def thermal_expansion(
        self,
        target,
        temperature,
        material=None,
        *,
        measure=None,
        name: str = "F_thermal",
    ):
        """Create the equivalent force from a solved temperature field."""

        import ufl

        from . import operators

        record = (
            self._material_record(material)
            if material is not None
            else _single_material(self, "model.thermal_expansion")
        )
        selected_measure = (
            measure
            if measure is not None
            else (record.region.measure if record.region is not None else ufl.dx)
        )
        return operators.thermal_expansion_vector(
            target,
            temperature,
            record.item,
            study=self.study,
            measure=selected_measure,
            name=name,
        )

    def lumped_mass(
        self,
        target,
        material=None,
        *,
        measure=None,
        method: str = "row_sum",
        name: str = "M_lumped",
    ):
        """Assemble a lumped mass operator from registered material densities.

        The model-first path is region aware: a single material may use the
        whole domain, while multiple materials must each have a cell region.
        """

        if method.lower().replace("-", "_") not in {"row_sum", "diagonal", "lumped"}:
            raise ValueError("model.lumped_mass currently supports method='row_sum'.")

        from . import assembly
        from . import _axisymmetric
        from .operators import LumpedMassOperator

        V = _space(target)
        weight = _axisymmetric.integration_weight(target, self.study)
        if material is not None:
            record = self._material_record(material)
            selected_measure = measure if measure is not None else _record_measure(record)
            mass = _assemble_lumped_mass(
                assembly,
                V,
                _density(record.item) * weight,
                selected_measure,
            )
            return LumpedMassOperator(
                mass=mass,
                inv_mass=assembly.inverse_diagonal(mass),
            )

        if not self.materials:
            raise ValueError("model.lumped_mass requires at least one registered material.")
        if measure is not None and len(self.materials) > 1:
            raise ValueError(
                "model.lumped_mass with multiple materials cannot use one explicit measure. "
                "Pass material=... or let each material use its registered region."
            )
        if len(self.materials) == 1:
            record = self.materials[0]
            selected_measure = measure if measure is not None else _record_measure(record)
            mass = _assemble_lumped_mass(
                assembly,
                V,
                _density(record.item) * weight,
                selected_measure,
            )
            return LumpedMassOperator(
                mass=mass,
                inv_mass=assembly.inverse_diagonal(mass),
            )

        mass = None
        missing = []
        for record in self.materials:
            if record.region is None:
                missing.append(_describe(record.item))
                continue
            part = _assemble_lumped_mass(
                assembly,
                V,
                _density(record.item) * weight,
                record.region.measure,
            )
            mass = part if mass is None else mass + part
        if missing:
            raise ValueError(
                "Multiple-material lumped mass requires every material to have a region. "
                f"Materials without regions: {missing}."
            )
        return LumpedMassOperator(
            mass=mass,
            inv_mass=assembly.inverse_diagonal(mass),
        )

    def load_vector(self, target, loads=None, *, load=None):
        """Create a total load vector from registered or explicit loads."""

        from . import operators

        selected_loads = self.loads if loads is None and load is None else loads
        return operators.load_vector(
            target,
            selected_loads,
            load=load,
            study=self.study,
        )

    def external_force(self, target, loads=None, *, load=None):
        """Create the external force/source vector from model loads."""

        return self.load_vector(target, loads=loads, load=load)

    def internal_force_vector(
        self,
        displacement,
        test_function,
        material=None,
        *,
        measure=None,
        study=None,
        name: str = "F_internal",
    ):
        """Create elastic internal-force vector contributions from materials."""

        from . import operators

        selected_study = study or self.study
        if material is not None:
            record = self._material_record(material)
            return _internal_force_from_record(
                displacement,
                test_function,
                record,
                operators=operators,
                study=selected_study,
                measure=measure,
                name=name,
            )

        if not self.materials:
            raise ValueError(
                "model.internal_force_vector requires at least one registered material."
            )
        if measure is not None and len(self.materials) > 1:
            raise ValueError(
                "model.internal_force_vector with multiple materials cannot use one explicit "
                "measure. Pass material=... or let each material use its registered region."
            )
        if len(self.materials) == 1:
            return _internal_force_from_record(
                displacement,
                test_function,
                self.materials[0],
                operators=operators,
                study=selected_study,
                measure=measure,
                name=name,
            )

        parts = []
        missing = []
        for record in self.materials:
            if record.region is None:
                missing.append(_describe(record.item))
                continue
            parts.append(
                _internal_force_from_record(
                    displacement,
                    test_function,
                    record,
                    operators=operators,
                    study=selected_study,
                    measure=record.region.measure,
                    name=f"F_internal_{getattr(record.region, 'name', len(parts))}",
                )
            )
        if missing:
            raise ValueError(
                "Multiple-material internal force requires every material to have a region. "
                f"Materials without regions: {missing}."
            )
        return operators.combine(*parts, name=name, kind="partitioned_internal_force")

    def internal_force(
        self,
        displacement,
        test_function=None,
        material=None,
        *,
        measure=None,
        study=None,
        name: str = "F_internal",
    ):
        """Create the internal force vector using model materials.

        This is the user-facing alias for ``internal_force_vector``. When
        possible, the test function is inferred from the registered unknown
        sharing the same function space as ``displacement``.
        """

        return self.internal_force_vector(
            displacement,
            self._test_function_for(displacement, test_function),
            material=material,
            measure=measure,
            study=study,
            name=name,
        )

    def boundary_force(self, boundary_model, field, test_function=None):
        """Create a weak boundary-model force contribution."""

        from . import operators

        return operators.boundary_model_vector(
            boundary_model,
            field,
            self._test_function_for(field, test_function),
        )

    def force_balance(
        self,
        *,
        internal=None,
        external=None,
        damping=None,
        absorbing=None,
        boundary=None,
        name: str = "R",
        convention: str = "internal_minus_external",
    ):
        """Create a residual/force-balance vector from force contributions.

        The default convention is the elastodynamics residual
        ``R = F_internal + F_damping + F_boundary - F_external``. Explicit
        central-difference updates in AgentFEM currently use
        ``a = -M^{-1} R``.
        """

        from . import operators

        positive = []
        negative = []
        if convention == "internal_minus_external":
            positive.extend(_as_tuple(internal))
            positive.extend(_as_tuple(damping))
            positive.extend(_as_tuple(absorbing))
            positive.extend(_as_tuple(boundary))
            negative.extend(_as_tuple(external))
        elif convention == "external_minus_internal":
            positive.extend(_as_tuple(external))
            negative.extend(_as_tuple(internal))
            negative.extend(_as_tuple(damping))
            negative.extend(_as_tuple(absorbing))
            negative.extend(_as_tuple(boundary))
        else:
            raise ValueError(
                "force_balance convention must be 'internal_minus_external' "
                "or 'external_minus_internal'."
            )

        terms = [*positive, *(operators.scale(item, -1.0) for item in negative)]
        if not terms:
            raise ValueError("force_balance requires at least one force contribution.")
        return operators.combine(*terms, name=name, kind="force_balance")

    def add_step(self, step):
        """Register an analysis step and return it."""

        if hasattr(step, "step_number"):
            step.step_number = len(self.steps) + 1
        self.steps.append(step)
        return step

    def stage(
        self,
        name: str,
        *,
        previous=None,
        inherit_model_loads: bool = False,
        inherit_model_constraints: bool = True,
    ):
        """Define inherited load/constraint activation for an engineering Step."""

        from . import steps as step_api

        created = step_api.engineering_step(
            name,
            previous=previous,
            inherit_model_loads=inherit_model_loads,
            inherit_model_constraints=inherit_model_constraints,
        )
        self.engineering_steps.append(created)
        return created

    def step(
        self,
        *,
        kind: str | None = None,
        target=None,
        procedure=None,
        material=None,
        incrementation=None,
        dt: float | str | None = None,
        steps: int | None = None,
        duration: float | None = None,
        K=None,
        F=None,
        constraints=None,
        solver_options=None,
        output=None,
        history=None,
        executor=None,
        executor_options=None,
        progress: bool | None = None,
        checkpoint=None,
        name: str | None = None,
        configuration=None,
        **kwargs,
    ):
        """Create and register an analysis step.

        ``Step`` is the high-level workflow object for users and agents. It
        records the analysis intent while keeping operator-level objects visible
        instead of hiding the model behind a monolithic solver call. Pass
        ``procedure=`` only when the numerical route should be explicit or
        override the Study preference; capability inspection and lowering use
        that same resolved object.

        Common cross-physics inputs are explicit keyword-only parameters for
        useful IDE and agent discovery. Procedure-specific expert options stay
        extensible through ``kwargs`` but are checked against the selected
        provider's :class:`StepOptionContract` before assembly.
        """

        selected_kind = kind or getattr(self.study, "analysis", None)
        if selected_kind is None:
            raise ValueError("model.step requires kind=... or a study with an analysis.")
        from .step_providers import lower_step

        options = {
            "K": K,
            "F": F,
            "constraints": constraints,
            "solver_options": solver_options,
            "name": name,
        }
        common = {
            "material": material,
            "incrementation": incrementation,
            "dt": dt,
            "steps": steps,
            "duration": duration,
            "output": output,
            "history": history,
            "executor": executor,
            "executor_options": executor_options,
            "progress": progress,
            "checkpoint": checkpoint,
        }
        options.update(
            (key, value) for key, value in common.items() if value is not None
        )
        options.update(kwargs)
        if configuration is None:
            return lower_step(
                self,
                analysis=selected_kind,
                target=target,
                options=options,
                procedure=procedure,
            )
        original_loads, original_constraints = self.loads, self.constraints
        configuration.apply_predefined_fields()
        self.loads = list(configuration.resolve_loads(original_loads))
        self.constraints = list(configuration.resolve_constraints(original_constraints))
        if constraints is not None:
            options["constraints"] = constraints
        try:
            created = lower_step(
                self,
                analysis=selected_kind,
                target=target,
                options=options,
                procedure=procedure,
            )
            created.engineering_step = configuration
            return created
        finally:
            self.loads, self.constraints = original_loads, original_constraints

    def linear_static_step(
        self,
        *,
        target,
        K=None,
        F=None,
        constraints=None,
        solver_options=None,
        name: str = "linear_static",
    ):
        """Compatibility builder; prefer the stable :meth:`step` entry point."""

        from . import _step_builders

        return _step_builders.linear_static(
            self,
            target=target,
            K=K,
            F=F,
            constraints=constraints,
            solver_options=solver_options,
            name=name,
        )

    def heat_transfer_step(
        self,
        *,
        target,
        dt: float,
        steps: int,
        material=None,
        C=None,
        K=None,
        Q=None,
        constraints=None,
        solver_options=None,
        update_load=None,
        save_every: int | None = None,
        print_every: int | None = None,
        progress=True,
        status_file=None,
        checkpoint=None,
        name: str = "transient_heat",
    ):
        """Compatibility builder; prefer the stable :meth:`step` entry point."""

        from . import _step_builders

        return _step_builders.heat_transfer(
            self,
            target=target,
            dt=dt,
            steps=steps,
            material=material,
            C=C,
            K=K,
            Q=Q,
            constraints=constraints,
            solver_options=solver_options,
            update_load=update_load,
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint=checkpoint,
            name=name,
        )

    def hyperelastic_step(
        self,
        *,
        target,
        material=None,
        constraints=None,
        solver_options=None,
        measure=None,
        name: str = "hyperelastic",
        petsc_options_prefix: str = "agentfem_hyperelastic_",
        incrementation=None,
        increments: int | None = None,
        load_factors=None,
        output=None,
        output_every: int | None = None,
        progress=True,
        status_file=None,
    ):
        """Compatibility builder; prefer the stable model.step entry point."""

        from . import _step_builders

        return _step_builders.hyperelastic(
            self,
            target=target,
            material=material,
            constraints=constraints,
            solver_options=solver_options,
            measure=measure,
            name=name,
            petsc_options_prefix=petsc_options_prefix,
            incrementation=incrementation,
            increments=increments,
            load_factors=load_factors,
            output=output,
            output_every=output_every,
            progress=progress,
            status_file=status_file,
        )

    def mixed_hyperelastic_step(
        self,
        *,
        target,
        material=None,
        constraints=None,
        solver_options=None,
        measure=None,
        name: str = "mixed_hyperelastic",
        petsc_options_prefix: str = "agentfem_mixed_hyperelastic_",
        incrementation=None,
        increments: int | None = None,
        load_factors=None,
        output=None,
        output_every: int | None = None,
        progress=True,
        status_file=None,
    ):
        """Compatibility builder; prefer the stable model.step entry point."""

        from . import _step_builders

        return _step_builders.mixed_hyperelastic(
            self,
            target=target,
            material=material,
            constraints=constraints,
            solver_options=solver_options,
            measure=measure,
            name=name,
            petsc_options_prefix=petsc_options_prefix,
            incrementation=incrementation,
            increments=increments,
            load_factors=load_factors,
            output=output,
            output_every=output_every,
            progress=progress,
            status_file=status_file,
        )

    def j2_plasticity_step(
        self,
        *,
        target,
        material=None,
        constraints=None,
        incrementation=None,
        solver_options=None,
        quadrature_degree: int = 2,
        progress=True,
        status_file=None,
        amplitude=None,
        name: str = "j2_plasticity",
    ):
        """Compatibility builder; prefer the stable :meth:`step` entry point."""

        from . import _step_builders

        return _step_builders.j2_plasticity(
            self,
            target=target,
            material=material,
            constraints=constraints,
            incrementation=incrementation,
            solver_options=solver_options,
            quadrature_degree=quadrature_degree,
            progress=progress,
            status_file=status_file,
            amplitude=amplitude,
            name=name,
        )

    def creep_step(
        self,
        *,
        target,
        duration: float,
        material=None,
        constraints=None,
        incrementation=None,
        solver_options=None,
        quadrature_degree: int = 2,
        progress=True,
        status_file=None,
        amplitude=None,
        temperature=None,
        name: str = "implicit_creep",
    ):
        """Compatibility builder; prefer the stable :meth:`step` entry point."""

        from . import _step_builders

        return _step_builders.creep(
            self,
            target=target,
            duration=duration,
            material=material,
            constraints=constraints,
            incrementation=incrementation,
            solver_options=solver_options,
            quadrature_degree=quadrature_degree,
            progress=progress,
            status_file=status_file,
            amplitude=amplitude,
            temperature=temperature,
            name=name,
        )

    def explicit_dynamics_step(
        self,
        *,
        target,
        dt: float,
        steps: int,
        residual=None,
        state=None,
        mass=None,
        cohesive_force=None,
        prescribed=(),
        constraints=None,
        update_load=None,
        save_every: int | None = None,
        print_every: int | None = None,
        progress=True,
        status_file=None,
        checkpoint=None,
        name: str = "explicit_dynamics",
    ):
        """Compatibility builder; prefer the stable model.step entry point."""

        from . import _step_builders

        return _step_builders.explicit_dynamics(
            self,
            target=target,
            dt=dt,
            steps=steps,
            residual=residual,
            state=state,
            mass=mass,
            cohesive_force=cohesive_force,
            prescribed=prescribed,
            constraints=constraints,
            update_load=update_load,
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint=checkpoint,
            name=name,
        )

    def finite_strain_explicit_dynamics_step(
        self,
        *,
        target,
        dt: float | str | None = "auto",
        steps: int,
        material=None,
        state=None,
        mass=None,
        cohesive_force=None,
        constraints=None,
        update_load=None,
        save_every: int | None = None,
        print_every: int | None = None,
        history_every: int = 1,
        progress=True,
        status_file=None,
        checkpoint=None,
        stability_safety: float = 0.8,
        mass_damping: float = 0.0,
        name: str = "finite_strain_explicit_dynamics",
    ):
        """Compatibility builder; prefer the stable model.step entry point."""

        from . import _step_builders

        return _step_builders.finite_strain_explicit_dynamics(
            self,
            target=target,
            dt=dt,
            steps=steps,
            material=material,
            state=state,
            mass=mass,
            cohesive_force=cohesive_force,
            constraints=constraints,
            update_load=update_load,
            save_every=save_every,
            print_every=print_every,
            history_every=history_every,
            progress=progress,
            status_file=status_file,
            checkpoint=checkpoint,
            stability_safety=stability_safety,
            mass_damping=mass_damping,
            name=name,
        )

    def implicit_dynamics_step(
        self,
        *,
        target,
        dt: float,
        steps: int,
        method: str = "newmark",
        spectral_radius: float = 0.8,
        M=None,
        C=None,
        K=None,
        F=None,
        state=None,
        constraints=None,
        solver_options=None,
        update_load=None,
        progress=True,
        status_file=None,
        checkpoint=None,
        save_every: int | None = None,
        print_every: int | None = None,
        name: str = "implicit_dynamics",
    ):
        """Compatibility builder; prefer the stable model.step entry point."""

        from . import _step_builders

        return _step_builders.implicit_dynamics(
            self,
            target=target,
            dt=dt,
            steps=steps,
            method=method,
            spectral_radius=spectral_radius,
            M=M,
            C=C,
            K=K,
            F=F,
            state=state,
            constraints=constraints,
            solver_options=solver_options,
            update_load=update_load,
            progress=progress,
            status_file=status_file,
            checkpoint=checkpoint,
            save_every=save_every,
            print_every=print_every,
            name=name,
        )

    def operator(self, kind: str, target, **kwargs):
        """Create a model-level operator by name.

        This string entry point is useful for configuration-driven or agent
        generated workflows. Human-facing code should prefer explicit methods
        such as ``model.stiffness(...)`` and ``model.load_vector(...)``.
        """

        normalized = kind.lower().replace("-", "_")
        if normalized in {"stiffness", "k"}:
            return self.stiffness(target, **kwargs)
        if normalized in {"load", "load_vector", "force", "f"}:
            return self.load_vector(target, **kwargs)
        if normalized in {"mass", "lumped_mass", "m_lumped", "ml"}:
            return self.lumped_mass(target, **kwargs)
        if normalized in {"internal_force", "f_internal", "fint"}:
            return self.internal_force(target, **kwargs)
        if normalized in {"external_force", "f_external", "fext"}:
            return self.external_force(target, **kwargs)
        raise ValueError(f"Unknown model operator kind {kind!r}.")

    def add_boundary_model(self, boundary_model):
        """Register weak boundary physics and return it."""

        self._register_regions_from_asset(boundary_model)
        self.boundary_models.append(boundary_model)
        return boundary_model

    def add_region(self, region):
        """Register a named mesh region and return it."""

        self._register_region(region)
        return region

    def bcs(self) -> list:
        """Return all registered Dirichlet BC objects."""

        result = []
        for constraint in self.constraints:
            if hasattr(constraint, "bcs"):
                result.extend(constraint.bcs)
            elif hasattr(constraint, "bc"):
                result.append(constraint.bc)
        return result

    def validate(self, *, target=None, step_options=None):
        """Return structured, addressable model-validation results.

        This method does not assemble forms or solve a system.  It validates
        scientific registry structure and backend-independent relationships
        that can be checked cheaply before execution.  Paths such as
        ``model.materials[1].region`` are stable repair targets for humans,
        agents, and future validation tools.
        """

        from .validation import ValidationReport, issue

        issues = []
        study = self.study
        domain = _domain(self.mesh)

        if study is None:
            issues.append(
                issue(
                    "AFM-MODEL-001",
                    "model.study",
                    "A finite-element model requires a Study.",
                    hint="Create a study with agentfem.studies before the model.",
                )
            )
        elif hasattr(study, "validate"):
            try:
                study.validate()
            except (TypeError, ValueError) as exc:
                issues.append(
                    issue(
                        "AFM-STUDY-001",
                        "model.study",
                        str(exc),
                        hint="Revise the analysis, physics, dimension, or assumption.",
                    )
                )

        if study is not None and self.fields:
            from .step_providers import step_capability

            capability = step_capability(
                self,
                target=target,
                options=step_options,
            )
            if not capability["supported"]:
                issues.append(
                    issue(
                        "AFM-STUDY-002",
                        "model.study",
                        (
                            "No executable step provider supports this Study, "
                            "field, material, and procedure combination."
                        ),
                        hint=(
                            "Choose a supported Study/material combination or "
                            "register a StepProvider before solving."
                        ),
                        capability=capability,
                    )
                )

        if domain is None:
            issues.append(
                issue(
                    "AFM-MODEL-002",
                    "model.mesh",
                    "A finite-element model requires a mesh.",
                    hint="Create or import the mesh before defining regions and fields.",
                )
            )
        elif study is not None:
            geometry = getattr(domain, "geometry", None)
            mesh_dimension = getattr(geometry, "dim", None)
            if mesh_dimension is None:
                issues.append(
                    issue(
                        "AFM-MESH-001",
                        "model.mesh",
                        "The registered mesh does not expose a geometric dimension.",
                        hint="Register a DOLFINx mesh or an AgentFEM FEMMesh.",
                    )
                )
            elif int(mesh_dimension) != int(study.dimension):
                issues.append(
                    issue(
                        "AFM-MODEL-003",
                        "model.mesh.geometry.dim",
                        (
                            f"Study dimension {study.dimension} does not match "
                            f"mesh geometric dimension {mesh_dimension}."
                        ),
                        hint="Revise the Study dimension or use the intended mesh.",
                        study_dimension=int(study.dimension),
                        mesh_dimension=int(mesh_dimension),
                    )
                )
            elif (
                getattr(study, "assumption", None) == "axisymmetric"
                and int(mesh_dimension) == 2
            ):
                from mpi4py import MPI

                coordinates = np.asarray(domain.geometry.x[:, 0], dtype=float)
                local_min = float(np.min(coordinates)) if coordinates.size else np.inf
                local_max = float(np.max(np.abs(coordinates))) if coordinates.size else 0.0
                minimum_radius = float(
                    domain.comm.allreduce(local_min, op=MPI.MIN)
                )
                radius_scale = float(
                    domain.comm.allreduce(local_max, op=MPI.MAX)
                )
                tolerance = 1.0e-12 * max(1.0, radius_scale)
                if minimum_radius < -tolerance:
                    issues.append(
                        issue(
                            "AFM-AXISYM-001",
                            "model.mesh.geometry.x[:,0]",
                            "An axisymmetric meridian cannot contain negative radius.",
                            hint="Define the meridian in coordinates (r,z) with r >= 0.",
                            minimum_radius=minimum_radius,
                        )
                    )
                elif minimum_radius <= tolerance:
                    names = {
                        str(getattr(item, "name", "")).strip().lower()
                        for item in constraint_api.dirichlet_constraints(
                            self.constraints
                        )
                    }
                    has_axis_regularity = any(
                        name == "axisymmetric_axis" or name.startswith("symmetry_x")
                        for name in names
                    )
                    if not has_axis_regularity:
                        issues.append(
                            issue(
                                "AFM-AXISYM-002",
                                "model.constraints",
                                "The meridian reaches r=0 without a declared radial regularity constraint.",
                                severity="warning",
                                hint=(
                                    "Register constraints.axisymmetric_axis(u, on=axis) "
                                    "or an equivalent named x-normal symmetry constraint."
                                ),
                            )
                        )

        if not self.fields:
            issues.append(
                issue(
                    "AFM-MODEL-004",
                    "model.fields",
                    "A finite-element model requires at least one field.",
                    hint="Register an unknown with model.field(...).",
                )
            )
        elif domain is not None:
            for index, field_object in enumerate(self.fields):
                field_domain = _field_domain(field_object)
                if field_domain is not None and field_domain is not domain:
                    issues.append(
                        issue(
                            "AFM-FIELD-001",
                            f"model.fields[{index}]",
                            "The field is defined on a different mesh from the model.",
                            hint="Create the field on model.mesh or register the intended mesh.",
                        )
                    )

        selected_material = (
            None if step_options is None else step_options.get("material")
        )
        explicit_linear_system = bool(
            step_options
            and step_options.get("K") is not None
            and step_options.get("F") is not None
        )
        if (
            study is not None
            and (
                getattr(study, "is_solid_mechanics", False)
                or getattr(study, "is_heat_transfer", False)
            )
            and not self.materials
            and selected_material is None
            and not explicit_linear_system
        ):
            physics = getattr(study, "physics", "finite-element")
            issues.append(
                issue(
                    "AFM-MATERIAL-001",
                    "model.materials",
                    (
                        f"{physics.replace('_', ' ').title()} models require "
                        "at least one material."
                    ),
                    hint="Register material properties with model.material(...).",
                )
            )

        if len(self.materials) > 1:
            for index, record in enumerate(self.materials):
                if getattr(record, "region", None) is None:
                    issues.append(
                        issue(
                            "AFM-MATERIAL-002",
                            f"model.materials[{index}].region",
                            "Every material in a multi-material model needs a region.",
                            hint="Pass region=... when registering each material.",
                        )
                    )

        if study is not None:
            selected_options = {} if step_options is None else dict(step_options)
            communicator = None if domain is None else getattr(domain, "comm", None)
            compatibility = constraint_api.validate_solver_compatibility(
                constraints=selected_options.get("constraints", self.constraints),
                analysis=getattr(study, "analysis", ""),
                procedure=(
                    selected_options.get("method")
                    or getattr(study, "preferred_procedure", None)
                ),
                comm_size=int(getattr(communicator, "size", 1)),
            )
            issues.extend(compatibility.issues)

        for collection_name in (
            "fields",
            "amplitudes",
            "constraints",
            "loads",
            "boundary_models",
            "regions",
            "steps",
        ):
            duplicates = _duplicate_names(getattr(self, collection_name))
            for name in duplicates:
                issues.append(
                    issue(
                        "AFM-NAME-001",
                        f"model.{collection_name}",
                        f"Name {name!r} is used more than once.",
                        severity="warning",
                        hint="Use unique names when objects must be addressed for repair or reuse.",
                        duplicate_name=name,
                    )
                )

        if domain is not None:
            for index, region in enumerate(self.regions):
                region_domain = getattr(region, "domain", None)
                if region_domain is not None and region_domain is not domain:
                    issues.append(
                        issue(
                            "AFM-REGION-001",
                            f"model.regions[{index}]",
                            "The region belongs to a different mesh from the model.",
                            hint="Recreate the region on model.mesh.",
                        )
                    )

        return ValidationReport.from_issues(issues, scope=f"model:{self.name}")

    def check(self, *, target=None, step_options=None) -> None:
        """Raise one structured error report if model validation fails."""

        self.validate(target=target, step_options=step_options).raise_if_errors()

    def audit_boundaries(self, *, strict: bool = False) -> dict[str, object]:
        """Return physical evidence for every registered boundary region.

        This is deliberately separate from lightweight model validation: it
        assembles boundary measures and normals and, for hybrid imported
        regions, compares the physical tag with its geometric audit marker.
        """

        records = {}
        for region in self.regions:
            audit = getattr(region, "audit", None)
            if not callable(audit):
                continue
            name = str(getattr(region, "name", f"region_{len(records)}"))
            if name in records:
                name = f"{name}_{len(records)}"
            records[name] = audit(strict=strict)
        return records

    def _register_regions_from_asset(self, asset) -> None:
        for region in _regions_from_asset(asset):
            self._register_region(region)

    def _test_function_for(self, field_or_function, explicit=None):
        if explicit is not None:
            return explicit
        if hasattr(field_or_function, "test"):
            return field_or_function.test

        from . import fields as field_api

        try:
            function = field_api.unwrap(field_or_function)
            function_space = function.function_space
        except AttributeError as exc:
            raise ValueError(
                "A test_function is required unless AgentFEM can infer it from "
                "a registered field on the same function space."
            ) from exc

        for item in self.fields:
            if getattr(item, "space", None) is function_space and hasattr(item, "test"):
                return item.test
            value = getattr(item, "value", None)
            if (
                value is not None
                and getattr(value, "function_space", None) is function_space
                and hasattr(item, "test")
            ):
                return item.test
        raise ValueError(
            "Could not infer a test function for this field. Pass test_function=... "
            "or register the corresponding UnknownField with model.field(...)."
        )

    def _register_region(self, region) -> None:
        if region is None:
            return
        for existing in self.regions:
            if _same_region(existing, region):
                return
        self.regions.append(region)

    def _material_record(self, material) -> "_WithRegion":
        if isinstance(material, _WithRegion):
            return material
        for record in self.materials:
            if record.item is material or record.definition is material:
                return record
        return _WithRegion(material, None)

    def _amplitude_by_name(self, name: str):
        for amplitude in self.amplitudes:
            if getattr(amplitude, "name", None) == name:
                return amplitude
        raise KeyError(f"Unknown amplitude {name!r}.")

    def _time_update_callback(self, callback=None, *, include_constraints=True):
        """Compose registered amplitude assets with an optional user callback."""

        assets = list(self.loads) + list(self.boundary_models)
        if include_constraints:
            assets.extend(constraint_api.dirichlet_constraints(self.constraints))
        updates = []
        seen = set()
        for asset in assets:
            update = getattr(asset, "update", None)
            if update is None or id(asset) in seen:
                continue
            seen.add(id(asset))
            updates.append(update)
        if callback is None and not updates:
            return None

        def update_all(time_value):
            for update in updates:
                update(time_value)
            if callback is not None:
                callback(time_value)

        return update_all

    def summary(self) -> dict[str, object]:
        """Return an agent-readable model summary."""

        return {
            "name": self.name,
            "study": _describe(self.study),
            "unit_system": _describe(self.unit_system),
            "mesh": _mesh_summary(self.mesh),
            "fields": tuple(_describe(item) for item in self.fields),
            "amplitudes": tuple(_describe(item) for item in self.amplitudes),
            "materials": tuple(_describe(item) for item in self.materials),
            "constraints": tuple(_describe(item) for item in self.constraints),
            "loads": tuple(_describe(item) for item in self.loads),
            "boundary_models": tuple(_describe(item) for item in self.boundary_models),
            "regions": tuple(_describe(item) for item in self.regions),
            "engineering_steps": tuple(
                _describe(item) for item in self.engineering_steps
            ),
            "steps": tuple(_describe(item) for item in self.steps),
        }

    def manifest(self) -> dict[str, object]:
        """Return a stable machine-readable model manifest."""

        return {
            "kind": "agentfem_model_manifest",
            "version": 1,
            "schema": "agentfem.af-ir",
            "schema_version": "0.1.0",
            "status": "experimental",
            "model": self.summary(),
            "workflow_order": (
                "study",
                "mesh",
                "regions",
                "fields",
                "materials",
                "constraints",
                "loads",
                "boundary_models",
                "engineering_steps",
                "steps",
            ),
        }

    def to_ir(
        self,
        *,
        include_validation: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Export the supported model semantics as an AF-IR document.

        AF-IR 0.1 is a versioned scientific record, not yet a complete
        backend-neutral executable serialization.  Backend runtime objects that
        lack public semantics are marked opaque instead of being serialized
        through unstable representations.
        """

        from . import __version__
        from .backends import get_backend
        from .ir import model_document

        document = model_document(
            self,
            agentfem_version=__version__,
            backend=get_backend().descriptor.as_dict(),
            include_validation=include_validation,
            metadata=metadata,
        )
        return document.as_dict()

    def write_ir(
        self,
        path,
        *,
        include_validation: bool = True,
        metadata: dict[str, object] | None = None,
    ):
        """Write a deterministic AF-IR JSON record and return its path.

        For a distributed DOLFINx mesh, rank zero performs the file write and
        all ranks synchronize before returning.
        """

        from pathlib import Path
        from .ir import write_document

        output = Path(path)
        domain = _domain(self.mesh)
        comm = getattr(domain, "comm", None)
        rank = getattr(comm, "rank", 0)
        if rank == 0:
            write_document(
                self.to_ir(
                    include_validation=include_validation,
                    metadata=metadata,
                ),
                output,
            )
        if comm is not None and hasattr(comm, "barrier"):
            comm.barrier()
        return output

    def tree(self) -> str:
        """Return a compact text model tree for logs, notebooks, and agents."""

        sections = [
            ("study", (self.study,)),
            ("mesh", (self.mesh,) if self.mesh is not None else ()),
            ("regions", self.regions),
            ("fields", self.fields),
            ("materials", self.materials),
            ("constraints", self.constraints),
            ("loads", self.loads),
            ("boundary_models", self.boundary_models),
            ("steps", self.steps),
        ]
        lines = [f"Model: {self.name}"]
        for title, items in sections:
            lines.append(f"  {title}:")
            if not items:
                lines.append("    - <empty>")
                continue
            for item in items:
                lines.append(f"    - {_short_description(item)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _WithRegion:
    item: object
    region: object | None = None
    definition: object | None = None

    def summary(self) -> dict[str, object]:
        return {
            "item": _describe(self.item),
            "region": getattr(self.region, "name", None),
            "material_definition": (
                None if self.definition is None else _describe(self.definition)
            ),
        }


def create(*, study, mesh=None, name: str = "model", units=None) -> Model:
    """Create a lightweight model registry."""

    return Model(study=study, mesh=mesh, name=name, unit_system=units)


def _stiffness_from_record(
    target,
    record: _WithRegion,
    *,
    operators,
    study,
    measure=None,
    law=None,
    temperature=None,
    name: str = "K",
):
    from .constitutive import hyperelasticity

    if hyperelasticity.is_finite_strain_hyperelastic(record.item) and law is None:
        raise TypeError(
            "A hyperelastic material has a deformation-dependent tangent, not "
            "one linear stiffness operator. Build its finite-strain residual "
            "and use operators.linearize(...) when a tangent is required."
        )
    selected_measure = measure
    if selected_measure is None and record.region is not None:
        selected_measure = record.region.measure
    kwargs = {"study": study}
    if law is not None:
        kwargs["law"] = law
    if temperature is not None:
        kwargs["temperature"] = getattr(temperature, "value", temperature)
    if selected_measure is not None:
        kwargs["measure"] = selected_measure
    operator = operators.stiffness(target, record.item, **kwargs)
    if record.region is not None:
        return operator.renamed(name, kind="regional_stiffness")
    return operator.renamed(name)


def _internal_force_from_record(
    displacement,
    test_function,
    record: _WithRegion,
    *,
    operators,
    study,
    measure=None,
    name: str = "F_internal",
):
    import ufl

    from .constitutive import hyperelasticity

    selected_measure = measure
    if selected_measure is None and record.region is not None:
        selected_measure = record.region.measure
    if hyperelasticity.is_finite_strain_hyperelastic(record.item):
        from . import fracture

        operator = fracture.finite_strain_internal_force(
            displacement,
            test_function,
            record.item,
            measure=(ufl.dx if selected_measure is None else selected_measure),
            name=name,
        )
        if record.region is not None:
            return operator.renamed(
                name,
                kind="regional_finite_strain_internal_force",
            )
        return operator
    kwargs = {"study": study}
    if selected_measure is not None:
        kwargs["measure"] = selected_measure
    operator = operators.internal_force_vector(
        displacement,
        test_function,
        record.item,
        **kwargs,
    )
    if record.region is not None:
        return operator.renamed(name, kind="regional_internal_force")
    return operator.renamed(name)


def _space(target):
    if hasattr(target, "space"):
        return target.space
    if hasattr(target, "function_space"):
        return target.function_space
    if hasattr(target, "value") and hasattr(target.value, "function_space"):
        return target.value.function_space
    return target


def _record_measure(record: _WithRegion):
    return record.region.measure if record.region is not None else None


def _assemble_lumped_mass(assembly, V, density: float, measure):
    if measure is None:
        return assembly.assemble_lumped_mass(V, density=density)
    return assembly.assemble_lumped_mass(V, density=density, measure=measure)


def _density(material) -> float:
    if not hasattr(material, "density"):
        raise ValueError(f"Material {_describe(material)!r} does not define density.")
    density = float(material.density)
    if density <= 0.0:
        raise ValueError(f"Material {_describe(material)!r} must have positive density.")
    return density


def _single_material(model: Model, caller: str) -> "_WithRegion":
    if len(model.materials) != 1:
        raise ValueError(f"{caller} requires material=... or exactly one material.")
    return model.materials[0]


def _describe(item):
    if item is None:
        return None
    if hasattr(item, "summary"):
        return item.summary()
    if hasattr(item, "as_dict"):
        return item.as_dict()
    return getattr(item, "name", repr(item))


def _short_description(item) -> str:
    if item is None:
        return "<none>"
    if hasattr(item, "topology") and hasattr(item, "geometry"):
        return (
            f"mesh: tdim={item.topology.dim}, "
            f"gdim={item.geometry.dim}, cells={item.topology.index_map(item.topology.dim).size_global}"
        )
    name = getattr(item, "name", None)
    kind = getattr(item, "kind", None)
    if name is not None and kind is not None:
        return f"{kind}: {name}"
    if name is not None:
        return str(name)
    if hasattr(item, "summary"):
        summary = item.summary()
        if isinstance(summary, dict):
            if "item" in summary and "region" in summary:
                region = summary["region"] or "whole domain"
                return f"material: {summary['item']} on {region}"
            if "dirichlet" in summary:
                count = len(summary.get("dirichlet", ()))
                return f"constraint set: {count} dirichlet"
            summary_name = summary.get("name")
            summary_kind = summary.get("kind") or summary.get("analysis")
            if summary_name is not None and summary_kind is not None:
                return f"{summary_kind}: {summary_name}"
            if summary_name is not None:
                return str(summary_name)
            return repr(summary)
    return repr(item)


def _as_tuple(item) -> tuple:
    if item is None:
        return ()
    if isinstance(item, tuple):
        return item
    if isinstance(item, list):
        return tuple(item)
    return (item,)


def _mesh_summary(mesh):
    if mesh is None:
        return None
    mesh = _domain(mesh)
    return {
        "topological_dim": mesh.topology.dim,
        "geometric_dim": mesh.geometry.dim,
    }


def _domain(mesh):
    """Return the DOLFINx domain stored directly or inside ``FEMMesh``."""

    if mesh is None:
        return None
    return getattr(mesh, "domain", mesh)


def _field_domain(field_object):
    """Return a field mesh when it can be determined without assembly."""

    space = getattr(field_object, "space", None)
    if space is None:
        value = getattr(field_object, "value", field_object)
        space = getattr(value, "function_space", None)
    return getattr(space, "mesh", None)


def _duplicate_names(items) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for item in items:
        name = getattr(item, "name", None)
        if name is None and hasattr(item, "summary"):
            summary = item.summary()
            if isinstance(summary, dict):
                name = summary.get("name")
        if isinstance(name, str) and name:
            counts[name] = counts.get(name, 0) + 1
    return tuple(sorted(name for name, count in counts.items() if count > 1))


def _regions_from_asset(asset) -> tuple[object, ...]:
    if asset is None:
        return ()
    if isinstance(asset, (list, tuple)):
        regions = []
        for item in asset:
            regions.extend(_regions_from_asset(item))
        return tuple(regions)
    if hasattr(asset, "loads"):
        return _regions_from_asset(asset.loads)
    if hasattr(asset, "dirichlet"):
        return _regions_from_asset(asset.dirichlet)
    if hasattr(asset, "periodic"):
        regions = list(_regions_from_asset(asset.periodic))
        return tuple(regions)
    if hasattr(asset, "master") or hasattr(asset, "slave"):
        regions = []
        master = getattr(asset, "master", None)
        slave = getattr(asset, "slave", None)
        if master is not None:
            regions.append(master)
        if slave is not None:
            regions.append(slave)
        return tuple(regions)
    region = getattr(asset, "location", None) or getattr(asset, "region", None)
    return () if region is None else (region,)


def _time_dependent_fix(target, *, on=None, location=None, value, components=None, name=None):
    selected_location = location if location is not None else on
    label = name or f"time_dependent_fixed_{getattr(selected_location, 'name', 'location')}"
    if components is None:
        components = _all_components_or_none(target)
    if components is None:
        return constraint_api.time_dependent_scalar_dirichlet(
            target,
            on=selected_location,
            value=value,
            name=label,
        )
    component_ids = (int(components),) if isinstance(components, Integral) else tuple(components)
    items = [
        constraint_api.time_dependent_component_dirichlet(
            target,
            component=component,
            on=selected_location,
            value=value,
            name=f"{label}_component_{component}",
        )
        for component in component_ids
    ]
    if len(items) == 1:
        return items[0]
    return constraint_api.ConstraintSet(dirichlet=items)


def _all_components_or_none(target) -> tuple[int, ...] | None:
    value = getattr(target, "value", target)
    shape = getattr(value, "ufl_shape", ())
    if len(shape) == 0:
        return None
    return tuple(range(int(shape[0])))


def _is_amplitude_like(value) -> bool:
    return callable(value) and not isinstance(value, (str, bytes))


def _same_region(left, right) -> bool:
    if left is right:
        return True
    left_key = _region_key(left)
    right_key = _region_key(right)
    return left_key is not None and left_key == right_key


def _region_key(region):
    if region is None:
        return None
    domain = getattr(region, "domain", None)
    name = getattr(region, "name", None)
    tag = getattr(region, "tag", None)
    if domain is None and name is None and tag is None:
        return None
    return (id(domain), name, tag)
