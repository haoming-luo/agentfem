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
from .step_providers import (
    StepOptionContract,
    StepProvider,
    StepProviderRegistry,
    register_step_provider,
    step_capability,
    step_providers,
)


CORE_MODEL_API = (
    "field",
    "amplitude",
    "material",
    "constraint",
    "fix",
    "prescribe",
    "clamp",
    "prescribed_temperature",
    "periodic",
    "load",
    "traction",
    "surface_force",
    "body_force",
    "heat_flux",
    "heat_source",
    "gravity",
    "pressure",
    "symmetry",
    "roller",
    "convection",
    "stage",
    "step",
    "validate",
    "check",
    "summary",
    "tree",
)

ADVANCED_MODEL_API = (
    "remote_displacement",
    "distributing_coupling",
    "remote_force",
    "centrifugal",
    "hydrostatic_pressure",
    "absorbing_boundary",
    "elastic_foundation",
    "stiffness",
    "mass",
    "damping",
    "conduction",
    "heat_capacity",
    "thermal_expansion",
    "lumped_mass",
    "load_vector",
    "external_force",
    "internal_force",
    "boundary_force",
    "force_balance",
    "operator",
    "add_region",
    "bcs",
    "audit_boundaries",
    "manifest",
    "to_ir",
    "write_ir",
)

COMPATIBILITY_MODEL_API = (
    "add_field",
    "add_amplitude",
    "add_material",
    "add_constraint",
    "add_load",
    "add_step",
    "add_boundary_model",
    "internal_force_vector",
    "linear_static_step",
    "heat_transfer_step",
    "hyperelastic_step",
    "mixed_hyperelastic_step",
    "j2_plasticity_step",
    "creep_step",
    "explicit_dynamics_step",
    "finite_strain_explicit_dynamics_step",
    "implicit_dynamics_step",
)


def model_api(level: str = "core") -> tuple[str, ...]:
    """Return the recommended Model vocabulary at one discovery level.

    This does not remove 0.2.x compatibility methods. It gives documentation,
    agents, IDE integrations, and future GUIs a deterministic way to present
    the concise engineering language before expert builders and historical
    aliases.
    """

    selected = str(level).lower().replace("-", "_").strip()
    levels = {
        "core": CORE_MODEL_API,
        "advanced": ADVANCED_MODEL_API,
        "compatibility": COMPATIBILITY_MODEL_API,
        "all": tuple(
            dict.fromkeys(
                CORE_MODEL_API + ADVANCED_MODEL_API + COMPATIBILITY_MODEL_API
            )
        ),
    }
    try:
        return levels[selected]
    except KeyError as exc:
        raise ValueError(
            "model_api level must be core, advanced, compatibility, or all."
        ) from exc


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
        """Register material data and return it."""

        self._register_region(region)
        self.materials.append(_WithRegion(material, region))
        return material

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

        from . import operators

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
                    _density(record.item),
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

        from . import operators

        return operators.damping_operator(
            target,
            coefficient,
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
            capacity = _volumetric_heat_capacity(record.item)
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
        from . import problems

        V = _space(target)
        if material is not None:
            record = self._material_record(material)
            selected_measure = measure if measure is not None else _record_measure(record)
            mass = _assemble_lumped_mass(assembly, V, _density(record.item), selected_measure)
            return problems.LumpedMassOperator(
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
            mass = _assemble_lumped_mass(assembly, V, _density(record.item), selected_measure)
            return problems.LumpedMassOperator(
                mass=mass,
                inv_mass=assembly.inverse_diagonal(mass),
            )

        mass = None
        missing = []
        for record in self.materials:
            if record.region is None:
                missing.append(_describe(record.item))
                continue
            part = _assemble_lumped_mass(assembly, V, _density(record.item), record.region.measure)
            mass = part if mass is None else mass + part
        if missing:
            raise ValueError(
                "Multiple-material lumped mass requires every material to have a region. "
                f"Materials without regions: {missing}."
            )
        return problems.LumpedMassOperator(
            mass=mass,
            inv_mass=assembly.inverse_diagonal(mass),
        )

    def load_vector(self, target, loads=None, *, load=None):
        """Create a total load vector from registered or explicit loads."""

        from . import operators

        selected_loads = self.loads if loads is None and load is None else loads
        return operators.load_vector(target, selected_loads, load=load)

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
        """Create and register a linear-static analysis step in ``K u = F`` form."""

        from . import operators
        from . import problems

        self.check(
            target=target,
            step_options={"K": K, "F": F},
        )
        update_at_step_end = self._time_update_callback()
        if update_at_step_end is not None:
            update_at_step_end(1.0)
        if getattr(self.study, "is_heat_transfer", False):
            boundary_stiffness, boundary_source = self._thermal_boundary_terms(target)
            if K is None:
                K = self.conduction(target)
            if boundary_stiffness:
                K = operators.combine(
                    K,
                    *boundary_stiffness,
                    name="K_thermal",
                    kind="conduction_and_exchange",
                )
            if F is None:
                sources = []
                if self.loads:
                    sources.append(self.external_force(target))
                sources.extend(boundary_source)
                F = (
                    operators.combine(*sources, name="Q", kind="thermal_source")
                    if sources
                    else operators.heat_source_vector(0.0, target)
                )
        else:
            K = K if K is not None else self.stiffness(target)
            foundation_terms = tuple(
                item.operator(target)
                for item in self.boundary_models
                if item.__class__.__name__ == "ElasticFoundation"
            )
            if foundation_terms:
                K = operators.combine(
                    K,
                    *foundation_terms,
                    name="K_with_foundation",
                    kind="solid_and_foundation_stiffness",
                )
            if F is None:
                if self.loads:
                    F = self.external_force(target)
                else:
                    value_shape = tuple(getattr(target.value, "ufl_shape", ()))
                    zero = (
                        0.0
                        if not value_shape
                        else tuple(0.0 for _ in range(value_shape[0]))
                    )
                    F = self.external_force(
                        target,
                        load=load_api.body_force(
                            zero,
                            target=target,
                            name="zero_external_force",
                        ),
                    )
        result_field_factory = None
        if (
            getattr(self.study, "is_solid_mechanics", False)
            and self.materials
            and all(
                _thermal_expansion_is_zero(record.item)
                for record in self.materials
            )
        ):
            assignments = tuple(self.materials)

            def result_field_factory(requested=None):
                from . import results

                isotropic = all(
                    hasattr(record.item, "young")
                    and hasattr(record.item, "poisson")
                    for record in assignments
                )
                defaults = results.preselected_fields(
                    physics="solid_mechanics",
                    finite_strain=False,
                )[1:]
                variables = tuple(defaults if requested is None else requested)
                if (
                    not isotropic
                    and requested is None
                    and getattr(self.study, "assumption", None) == "plane_strain"
                ):
                    variables = tuple(item for item in variables if item != "MISES")
                return results.small_strain_partition_fields(
                    target,
                    assignments,
                    study=self.study,
                    variables=variables,
                )

        step = problems.linear_static(
            K,
            F,
            study=self.study,
            unknown=target,
            constraints=self.constraints if constraints is None else constraints,
            solver_options=solver_options,
            result_field_factory=result_field_factory,
            name=name,
        )
        return self.add_step(step)

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
        """Create an implicit-Euler transient heat-transfer step."""

        import ufl
        from dolfinx import fem

        from . import operators
        from . import problems

        self.check(
            target=target,
            step_options={
                "material": material,
                "K": K,
                "F": Q,
                "dt": dt,
                "steps": steps,
            },
        )
        if hasattr(self.study, "require"):
            self.study.require(
                analysis="first_order_transient",
                physics="heat_transfer",
            )
        records = (
            (self._material_record(material),)
            if material is not None
            else tuple(self.materials)
        )
        if not records:
            raise ValueError("model.heat_transfer_step requires at least one material.")
        previous = fem.Function(target.space, name="TemperaturePrevious")
        previous.x.array[:] = target.value.x.array
        previous.x.scatter_forward()
        state_dependent = any(
            bool(getattr(record.item, "state_dependent_heat_transfer", False))
            for record in records
        )
        if state_dependent:
            if C is not None or K is not None:
                raise ValueError(
                    "Temperature-dependent heat transfer builds one consistent "
                    "enthalpy/conduction residual. Do not also pass C= or K=."
                )
            return self.add_step(
                self._nonlinear_heat_transfer_step(
                    target=target,
                    previous=previous,
                    records=records,
                    dt=dt,
                    steps=steps,
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
            )
        capacity = (
            self.heat_capacity(target, material)
            if C is None
            else C
        )
        stiffness = (
            self.conduction(target, material)
            if K is None
            else K
        )
        boundary_stiffness, boundary_source = self._thermal_boundary_terms(target)
        if boundary_stiffness:
            stiffness = operators.combine(
                stiffness,
                *boundary_stiffness,
                name="K_thermal",
                kind="conduction_and_exchange",
            )
        source = Q
        if source is None and self.loads:
            source = self.external_force(target)
        if boundary_source:
            source = operators.combine(
                *((() if source is None else (source,)) + tuple(boundary_source)),
                name="Q_thermal",
                kind="thermal_source",
            )
        history_parts = []
        for index, record in enumerate(records):
            capacity_value = _volumetric_heat_capacity(record.item)
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Multiple-material heat history requires a region for every material."
                )
            history_parts.append(
                operators.heat_capacity_vector(
                    previous,
                    target,
                    capacity_value,
                    measure=(
                        record.region.measure
                        if record.region is not None
                        else ufl.dx
                    ),
                ).renamed(
                    "Q_capacity_history"
                    if len(records) == 1
                    else f"Q_capacity_{getattr(record.region, 'name', index)}"
                )
            )
        history = (
            history_parts[0]
            if len(history_parts) == 1
            else operators.combine(
                *history_parts,
                name="Q_capacity_history",
                kind="partitioned_heat_capacity_history",
            )
        )
        step = problems.first_order_transient_run(
            capacity=capacity,
            stiffness=stiffness,
            history=history,
            source=source,
            current=target.value,
            previous=previous,
            dt=dt,
            steps=steps,
            study=self.study,
            constraints=(
                self.constraints if constraints is None else constraints
            ),
            solver_options=solver_options,
            update_load=self._time_update_callback(update_load),
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            name=name,
        )
        return self.add_step(step)

    def _nonlinear_heat_transfer_step(
        self,
        *,
        target,
        previous,
        records,
        dt,
        steps,
        Q,
        constraints,
        solver_options,
        update_load,
        save_every,
        print_every,
        progress,
        status_file,
        checkpoint,
        name,
    ):
        """Build conservative ``k(T), c_p(T)`` implicit heat transfer."""

        import ufl

        from . import operators, problems
        from .diagnostics import StateDependentThermalBalanceMonitor

        temperature = target.value
        test = ufl.TestFunction(target.space)
        direction = ufl.TrialFunction(target.space)
        residual = 0
        content_form = 0
        for index, record in enumerate(records):
            if len(records) > 1 and record.region is None:
                raise ValueError(
                    "Multiple-material nonlinear heat transfer requires a region "
                    "for every material."
                )
            material = record.item
            measure = record.region.measure if record.region is not None else ufl.dx
            if not hasattr(material, "conductivity") or not hasattr(
                material, "specific_heat"
            ):
                raise ValueError(
                    f"Material {_describe(material)!r} does not define conductivity "
                    "and specific heat."
                )
            if hasattr(material, "conductivity_at"):
                conductivity = material.conductivity_at(temperature)
            else:
                conductivity = material.conductivity
            if hasattr(material, "volumetric_enthalpy"):
                current_enthalpy = material.volumetric_enthalpy(temperature)
                previous_enthalpy = material.volumetric_enthalpy(previous)
            else:
                capacity = material.density * material.specific_heat
                current_enthalpy = capacity * temperature
                previous_enthalpy = capacity * previous
            residual += (
                (current_enthalpy - previous_enthalpy) / float(dt) * test
                + conductivity * ufl.dot(ufl.grad(temperature), ufl.grad(test))
            ) * measure
            content_form += current_enthalpy * measure

        outward_forms = []
        boundary_sources = []
        unsupported = []
        for boundary in self.boundary_models:
            if hasattr(boundary, "residual"):
                residual += boundary.residual(temperature, test)
                if hasattr(boundary, "outward_heat_rate_form"):
                    outward_forms.append(
                        boundary.outward_heat_rate_form(temperature)
                    )
                if hasattr(boundary, "source"):
                    boundary_sources.append(boundary.source(target))
            else:
                unsupported.append(getattr(boundary, "name", type(boundary).__name__))
        if unsupported:
            raise ValueError(
                "Nonlinear heat transfer cannot consume these boundary models: "
                f"{unsupported}."
            )

        source = Q
        if source is None and self.loads:
            source = self.external_force(target)
        if source is not None:
            residual -= getattr(source, "expression", source)
        monitor_source = source
        if boundary_sources:
            monitor_source = operators.combine(
                *((() if source is None else (source,)) + tuple(boundary_sources)),
                name="Q_thermal_ledger",
                kind="thermal_source",
            )
        jacobian = ufl.derivative(residual, temperature, direction)
        return problems.nonlinear_first_order_transient_run(
            residual=residual,
            jacobian=jacobian,
            current=temperature,
            previous=previous,
            dt=dt,
            steps=steps,
            study=self.study,
            constraints=self.constraints if constraints is None else constraints,
            solver_options=solver_options,
            update_load=self._time_update_callback(update_load),
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            history_monitor=StateDependentThermalBalanceMonitor(
                content_form=content_form,
                source=monitor_source,
                dt=float(dt),
                outward_forms=tuple(outward_forms),
            ),
            name=name,
        )

    def _thermal_boundary_terms(self, target):
        """Return matrix/vector contributions from registered thermal boundaries."""

        stiffness = []
        source = []
        unsupported = []
        for item in self.boundary_models:
            if hasattr(item, "operator") and hasattr(item, "source"):
                stiffness.append(item.operator(target))
                source.append(item.source(target))
            else:
                unsupported.append(getattr(item, "name", type(item).__name__))
        if unsupported and getattr(self.study, "is_heat_transfer", False):
            raise ValueError(
                "Heat-transfer steps cannot consume these boundary models: "
                f"{unsupported}."
            )
        return tuple(stiffness), tuple(source)

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
        """Create a compressible Neo-Hookean nonlinear static problem.

        The first implementation intentionally supports one material
        contribution. Multiple-region finite-strain materials need explicit
        kinematic and interface verification before becoming a convenience
        path.
        """

        import ufl
        from dolfinx import fem
        from petsc4py import PETSc

        from . import problems
        from .constitutive import hyperelasticity

        if hasattr(self.mesh, "require_formulation"):
            self.mesh.require_formulation(
                "displacement",
                operation="model.hyperelastic_step",
            )

        self.check(
            target=target,
            step_options={"material": material},
        )
        if hasattr(self.study, "require"):
            self.study.require(analysis="nonlinear_static", physics="solid_mechanics")
        if material is None:
            if len(self.materials) != 1:
                raise ValueError(
                    "model.hyperelastic_step requires material=... or exactly "
                    "one registered material."
                )
            record = self.materials[0]
        else:
            record = self._material_record(material)
        properties = record.item
        if not hyperelasticity.is_finite_strain_hyperelastic(properties):
            raise TypeError(
                "model.hyperelastic_step requires a supported hyperelastic material."
            )
        if not hyperelasticity.supports_hyperelastic_study(
            properties,
            dimension=getattr(self.study, "dimension", 0),
            assumption=getattr(self.study, "assumption", None),
        ):
            raise ValueError(
                "The Study dimension/assumption has no formulation for the "
                f"selected hyperelastic material {properties.name!r}."
            )
        selected_measure = measure
        if selected_measure is None:
            selected_measure = (
                record.region.measure if record.region is not None else ufl.dx
            )
        internal_residual = hyperelasticity.internal_virtual_work(
            target.value,
            target.test,
            properties,
            measure=selected_measure,
        )
        selected_constraints = self.constraints if constraints is None else constraints
        selected_constraints = _as_tuple(selected_constraints)
        affine_constraints = [
            item
            for item in selected_constraints
            if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
        ]
        if output is not None and output_every is not None:
            raise ValueError("Pass output=... or output_every=..., not both.")
        selected_output_every = (
            getattr(output, "every", None)
            if output is not None
            else (1 if output_every is None else int(output_every))
        )
        from . import steps as step_api

        selected_incrementation = step_api.normalize(
            incrementation,
            increments=increments,
            load_factors=load_factors,
        )
        if affine_constraints:
            if any(isinstance(item, load_api.AmplitudeLoad) for item in self.loads):
                raise NotImplementedError(
                    "Amplitude-driven natural loads are not yet supported by the "
                    "affine nonlinear path. Prescribed affine loading remains supported."
                )
            if len(affine_constraints) != 1 or len(selected_constraints) != 1:
                raise ValueError(
                    "The affine hyperelastic path currently requires exactly one "
                    "AbaqusPeriodicConstraint and no separate Dirichlet constraints."
                )
            residual = internal_residual
            if self.loads:
                residual -= self.external_force(target).expression
            jacobian = hyperelasticity.tangent(
                residual,
                target.value,
                target.trial,
            )
            output_factors = (
                output.required_factors()
                if output is not None and hasattr(output, "required_factors")
                else ()
            )

            def affine_finite_strain_acceptance():
                from .results import finite_strain_diagnostics, integral

                diagnostics = finite_strain_diagnostics(
                    target,
                    quadrature_degree=2,
                )
                minimum_j = float(diagnostics["minimum_quadrature_J"])
                return {
                    "accepted": bool(minimum_j > 0.0),
                    "minimum_quadrature_J": minimum_j,
                    "maximum_quadrature_J": float(
                        diagnostics["maximum_quadrature_J"]
                    ),
                    "recoverable_strain_energy": float(
                        integral(
                            hyperelasticity.strain_energy_density(
                                target.value,
                                properties,
                            ),
                            measure=selected_measure,
                            comm=_domain(self.mesh).comm,
                        )
                    ),
                    "message": (
                        "deformation Jacobian became non-positive"
                        if minimum_j <= 0.0
                        else ""
                    ),
                }

            problem = problems.affine_nonlinear(
                residual,
                target.value,
                jacobian=jacobian,
                constraint=affine_constraints[0],
                incrementation=selected_incrementation,
                solver_options=solver_options,
                output_every=selected_output_every,
                output_factors=output_factors,
                acceptance_check=affine_finite_strain_acceptance,
                progress=progress,
                status_file=status_file,
                name=name,
            )
        else:
            load_factor = fem.Constant(
                _domain(self.mesh),
                PETSc.ScalarType(0.0),
            )
            residual = internal_residual
            if self.loads:
                proportional_loads = tuple(
                    item
                    for item in self.loads
                    if not isinstance(item, load_api.AmplitudeLoad)
                )
                amplitude_loads = tuple(
                    item
                    for item in self.loads
                    if isinstance(item, load_api.AmplitudeLoad)
                )
                if proportional_loads:
                    residual -= load_factor * self.external_force(
                        target,
                        loads=proportional_loads,
                    ).expression
                if amplitude_loads:
                    residual -= self.external_force(
                        target,
                        loads=amplitude_loads,
                    ).expression
            jacobian = hyperelasticity.tangent(
                residual,
                target.value,
                target.trial,
            )

            def finite_strain_acceptance():
                from .results import finite_strain_diagnostics, integral

                diagnostics = finite_strain_diagnostics(
                    target,
                    quadrature_degree=2,
                )
                minimum_j = float(diagnostics["minimum_quadrature_J"])
                return {
                    "accepted": bool(minimum_j > 0.0),
                    "minimum_quadrature_J": minimum_j,
                    "maximum_quadrature_J": float(
                        diagnostics["maximum_quadrature_J"]
                    ),
                    "recoverable_strain_energy": float(
                        integral(
                            hyperelasticity.strain_energy_density(
                                target.value,
                                properties,
                            ),
                            measure=selected_measure,
                            comm=_domain(self.mesh).comm,
                        )
                    ),
                    "message": (
                        "deformation Jacobian became non-positive"
                        if minimum_j <= 0.0
                        else ""
                    ),
                }

            problem = problems.incremental_nonlinear(
                residual,
                target.value,
                factor=load_factor,
                value_path=constraint_api.prescribed_value_path(
                    selected_constraints
                ),
                update_load=self._time_update_callback(
                    include_constraints=False,
                ),
                acceptance_check=finite_strain_acceptance,
                jacobian=jacobian,
                incrementation=selected_incrementation,
                constraints=selected_constraints,
                solver_options=solver_options,
                output_every=selected_output_every,
                progress=progress,
                status_file=status_file,
                name=name,
                petsc_options_prefix=petsc_options_prefix,
            )
        return self.add_step(problem)

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
        """Create P2-displacement/DG0-pressure finite-strain equilibrium.

        This is the verified AgentFEM constant-pressure hybrid route.  A
        source ``C3D10H`` mesh is consumed only when the target explicitly
        carries one discontinuous constant pressure unknown per cell.
        """

        import ufl
        from dolfinx import fem
        from petsc4py import PETSc

        from . import problems
        from .constitutive import hyperelasticity

        if getattr(target, "kind", None) != "displacement_pressure":
            raise TypeError(
                "mixed_hyperelastic_step requires fields.displacement_pressure(...)."
            )
        if int(getattr(target, "displacement_degree", -1)) != 2 or int(
            getattr(target, "pressure_degree", -1)
        ) != 0:
            raise ValueError(
                "The verified constant-pressure hybrid route requires P2/DG0."
            )
        if hasattr(self.mesh, "require_formulation"):
            self.mesh.require_formulation(
                "hybrid",
                operation="model.mixed_hyperelastic_step",
            )
        if hasattr(self.study, "require"):
            self.study.require(analysis="nonlinear_static", physics="solid_mechanics")
        if getattr(self.study, "dimension", None) == 2 and getattr(
            self.study, "assumption", None
        ) != "plane_strain":
            raise NotImplementedError(
                "The mixed Neo-Hookean 2D route currently represents plane strain."
            )
        if material is None:
            if len(self.materials) != 1:
                raise ValueError(
                    "mixed_hyperelastic_step requires material=... or exactly one material."
                )
            record = self.materials[0]
        else:
            record = self._material_record(material)
        properties = record.item
        if not isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
            raise TypeError(
                "mixed_hyperelastic_step requires MixedNeoHookeanProperties."
            )
        selected_measure = measure or (
            record.region.measure if record.region is not None else ufl.dx
        )
        w = target.value
        displacement, pressure = ufl.split(w)
        test = ufl.TestFunction(target.space)
        trial = ufl.TrialFunction(target.space)
        internal_energy = hyperelasticity.mixed_strain_energy_density(
            displacement,
            pressure,
            properties,
        ) * selected_measure
        residual = ufl.derivative(internal_energy, w, test)
        load_factor = fem.Constant(_domain(self.mesh), PETSc.ScalarType(0.0))
        if self.loads:
            residual -= load_factor * self.external_force(
                target.displacement
            ).expression
        jacobian = ufl.derivative(residual, w, trial)
        selected_constraints = _as_tuple(
            self.constraints if constraints is None else constraints
        )
        affine_constraints = [
            item
            for item in selected_constraints
            if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
        ]
        if output is not None and output_every is not None:
            raise ValueError("Pass output=... or output_every=..., not both.")
        selected_output_every = (
            getattr(output, "every", None)
            if output is not None
            else (1 if output_every is None else int(output_every))
        )
        from . import steps as step_api

        selected_incrementation = step_api.normalize(
            incrementation,
            increments=increments,
            load_factors=load_factors,
        )

        def mixed_acceptance():
            from .results import finite_strain_diagnostics, integral

            displacement_field = target.collapsed_displacement()
            diagnostics = finite_strain_diagnostics(
                displacement_field,
                quadrature_degree=3,
            )
            minimum_j = float(diagnostics["minimum_quadrature_J"])
            return {
                "accepted": bool(minimum_j > 0.0),
                "minimum_quadrature_J": minimum_j,
                "maximum_quadrature_J": float(
                    diagnostics["maximum_quadrature_J"]
                ),
                "mixed_potential": float(
                    integral(
                        hyperelasticity.mixed_strain_energy_density(
                            displacement,
                            pressure,
                            properties,
                        ),
                        measure=selected_measure,
                        comm=_domain(self.mesh).comm,
                    )
                ),
                "message": (
                    "deformation Jacobian became non-positive"
                    if minimum_j <= 0.0
                    else ""
                ),
            }

        if affine_constraints:
            if len(affine_constraints) != 1 or len(selected_constraints) != 1:
                raise ValueError(
                    "The mixed affine hyperelastic path requires exactly one "
                    "AbaqusPeriodicConstraint and no separate Dirichlet constraints."
                )
            if self.loads:
                raise NotImplementedError(
                    "Natural loads are not yet combined with the mixed affine "
                    "periodic path; prescribe the macroscopic deformation gradient."
                )
            output_factors = (
                output.required_factors()
                if output is not None and hasattr(output, "required_factors")
                else ()
            )
            problem = problems.affine_nonlinear(
                residual,
                w,
                jacobian=jacobian,
                constraint=affine_constraints[0],
                incrementation=selected_incrementation,
                solver_options=solver_options,
                output_every=selected_output_every,
                output_factors=output_factors,
                acceptance_check=mixed_acceptance,
                progress=progress,
                status_file=status_file,
                name=name,
            )
        else:
            problem = problems.incremental_nonlinear(
                residual,
                w,
                factor=load_factor,
                value_path=constraint_api.prescribed_value_path(selected_constraints),
                update_load=self._time_update_callback(include_constraints=False),
                acceptance_check=mixed_acceptance,
                jacobian=jacobian,
                incrementation=selected_incrementation,
                constraints=selected_constraints,
                solver_options=solver_options,
                output_every=selected_output_every,
                progress=progress,
                status_file=status_file,
                name=name,
                petsc_options_prefix=petsc_options_prefix,
            )
        problem.primary_fields = {
            "U": target.displacement,
            "PRESSURE": target.pressure,
        }
        problem.result_field_factory = lambda: (
            target.collapsed_displacement(name="U"),
            target.collapsed_pressure(name="PRESSURE"),
        )
        problem.snapshot_field_factory = lambda: (
            target.collapsed_displacement(name="U"),
            {"PRESSURE": target.collapsed_pressure(name="PRESSURE")},
        )
        return self.add_step(problem)

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
        """Create a global 3D small-strain J2 plasticity step."""

        from . import mechanics
        from .constitutive.plasticity import J2LinearIsotropicHardening
        from .constitutive.quadrature import QuadratureMaterialMap

        self.check(
            target=target,
            step_options={"material": material},
        )
        if hasattr(self.study, "require"):
            self.study.require(
                analysis="nonlinear_static",
                physics="solid_mechanics",
            )
        if material is None:
            if not self.materials:
                raise ValueError(
                    "model.j2_plasticity_step requires registered material data."
                )
            material = QuadratureMaterialMap.from_assignments(
                target.value.function_space.mesh,
                self.materials,
                material_type=J2LinearIsotropicHardening,
            )
            if len(self.materials) == 1 and self.materials[0].region is None:
                material = self.materials[0].item
        else:
            material = self._material_record(material).item
        if not isinstance(material, (J2LinearIsotropicHardening, QuadratureMaterialMap)):
            raise TypeError(
                "model.j2_plasticity_step requires "
                "J2LinearIsotropicHardening."
            )
        time_dependent_constraints = tuple(
            item
            for item in constraint_api.dirichlet_constraints(
                self.constraints if constraints is None else constraints
            )
            if isinstance(item, constraint_api.TimeDependentDirichlet)
        )
        if time_dependent_constraints:
            raise NotImplementedError(
                "The J2 step does not accept absolute time-dependent Dirichlet "
                "histories. Prescribe the end-of-step value and use the step "
                "amplitude as its dimensionless load path."
            )
        selected_loads = tuple(self.loads)
        amplitude_loads = tuple(
            item
            for item in selected_loads
            if isinstance(item, load_api.AmplitudeLoad)
        )
        if amplitude_loads:
            ordinary_loads = tuple(
                item
                for item in selected_loads
                if not isinstance(item, load_api.AmplitudeLoad)
            )
            histories = {id(item.amplitude): item.amplitude for item in amplitude_loads}
            if ordinary_loads or len(histories) != 1:
                raise ValueError(
                    "A J2 step requires one shared load path. Do not mix ordinary "
                    "loads with amplitude-driven loads or use multiple amplitudes."
                )
            if amplitude is not None:
                raise ValueError(
                    "Pass a load amplitude or step amplitude to J2, not both."
                )
            amplitude = next(iter(histories.values()))
            selected_loads = tuple(item.load for item in amplitude_loads)
        step = mechanics.j2_plasticity_step(
            displacement=target,
            material=material,
            external_force=(
                self.external_force(target, loads=selected_loads)
                if selected_loads
                else None
            ),
            constraints=(
                self.constraints if constraints is None else constraints
            ),
            study=self.study,
            incrementation=incrementation,
            solver_options=solver_options,
            quadrature_degree=quadrature_degree,
            progress=progress,
            status_file=status_file,
            amplitude=amplitude,
            name=name,
        )
        return self.add_step(step)

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
        """Create a global 3D small-strain implicit creep step.

        ``duration`` and amplitudes use physical time. Ordinary loads and
        prescribed values are held at their full value by default; pass an
        amplitude when a ramp or other history is part of the model.
        """

        from . import mechanics
        from .constitutive.creep import IsotropicPowerLawCreepMaterial
        from .constitutive.quadrature import QuadratureMaterialMap

        self.check(target=target, step_options={"material": material})
        if hasattr(self.study, "require"):
            self.study.require(
                analysis="nonlinear_transient",
                physics="solid_mechanics",
            )
        if material is None:
            if not self.materials:
                raise ValueError(
                    "model.creep_step requires registered material data."
                )
            material = QuadratureMaterialMap.from_assignments(
                target.value.function_space.mesh,
                self.materials,
                material_type=IsotropicPowerLawCreepMaterial,
            )
            if len(self.materials) == 1 and self.materials[0].region is None:
                material = self.materials[0].item
        else:
            material = self._material_record(material).item
        if not isinstance(
            material, (IsotropicPowerLawCreepMaterial, QuadratureMaterialMap)
        ):
            raise TypeError(
                "model.creep_step requires IsotropicPowerLawCreepMaterial."
            )

        selected_loads = tuple(self.loads)
        amplitude_loads = tuple(
            item
            for item in selected_loads
            if isinstance(item, load_api.AmplitudeLoad)
        )
        if amplitude_loads:
            ordinary_loads = tuple(
                item
                for item in selected_loads
                if not isinstance(item, load_api.AmplitudeLoad)
            )
            histories = {id(item.amplitude): item.amplitude for item in amplitude_loads}
            if ordinary_loads or len(histories) != 1:
                raise ValueError(
                    "An implicit creep step currently requires one shared "
                    "physical-time load amplitude. Do not mix ordinary and "
                    "amplitude-driven loads or use multiple amplitudes."
                )
            if amplitude is not None:
                raise ValueError(
                    "Pass a load amplitude or step amplitude to creep, not both."
                )
            amplitude = next(iter(histories.values()))
            selected_loads = tuple(item.load for item in amplitude_loads)

        step = mechanics.implicit_creep_step(
            displacement=target,
            material=material,
            duration=duration,
            external_force=(
                self.external_force(target, loads=selected_loads)
                if selected_loads
                else None
            ),
            constraints=(self.constraints if constraints is None else constraints),
            study=self.study,
            incrementation=incrementation,
            solver_options=solver_options,
            quadrature_degree=quadrature_degree,
            progress=progress,
            status_file=status_file,
            amplitude=amplitude,
            temperature=temperature,
            name=name,
        )
        return self.add_step(step)

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
        """Create and register a second-order explicit dynamics step."""

        from . import problems
        from . import time as time_api
        from .constitutive import hyperelasticity

        if (
            residual is None
            and len(self.materials) == 1
            and hyperelasticity.is_finite_strain_hyperelastic(
                self.materials[0].item
            )
        ):
            if prescribed:
                raise ValueError(
                    "Use registered constraints for automatic finite-strain "
                    "Explicit, or pass an expert residual explicitly."
                )
            return self.finite_strain_explicit_dynamics_step(
                target=target,
                dt=dt,
                steps=steps,
                material=self.materials[0].item,
                state=state,
                mass=mass,
                cohesive_force=cohesive_force,
                constraints=constraints,
                update_load=update_load,
                save_every=save_every,
                print_every=print_every,
                progress=progress,
                status_file=status_file,
                checkpoint=checkpoint,
                name=name,
            )

        self.check(
            target=target,
            step_options={
                "mass": mass,
                "residual": residual,
                "method": "central_difference",
                "dt": dt,
                "steps": steps,
            },
        )
        selected_state = state if state is not None else problems.second_order_state(target)
        selected_mass = mass if mass is not None else self.lumped_mass(target)
        integrator = time_api.explicit.central_difference(
            state=selected_state,
            mass=selected_mass,
        )
        energy_stiffness = None
        if residual is None:
            energy_stiffness = self.stiffness(target)
            residual = self.force_balance(
                internal=self.internal_force(selected_state.u),
                external=(self.external_force(target) if self.loads else None),
            )
        selected_constraints = (
            self.constraints if constraints is None else constraints
        )
        selected_prescribed = (
            tuple(_as_tuple(prescribed))
            + constraint_api.dirichlet_constraints(selected_constraints)
        )
        step = problems.explicit_dynamics(
            state=selected_state,
            integrator=integrator,
            residual=residual,
            stiffness=energy_stiffness,
            study=self.study,
            prescribed=selected_prescribed,
            constraints=selected_constraints,
            update_load=self._time_update_callback(
                update_load,
                include_constraints=False,
            ),
            dt=dt,
            steps=steps,
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            name=name,
        )
        return self.add_step(step)

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
        """Create Total-Lagrangian Neo-Hookean central-difference dynamics."""

        import ufl

        from . import fracture
        from . import problems
        from . import time as time_api
        from .constitutive import hyperelasticity

        self.check(
            target=target,
            step_options={
                "material": material,
                "method": "central_difference",
                "dt": dt,
                "steps": steps,
            },
        )
        if hasattr(self.study, "require"):
            self.study.require(
                analysis="second_order_dynamics",
                physics="solid_mechanics",
            )
        record = (
            _single_material(self, "model.finite_strain_explicit_dynamics_step")
            if material is None
            else self._material_record(material)
        )
        properties = record.item
        if not hyperelasticity.is_finite_strain_hyperelastic(properties):
            raise TypeError(
                "finite_strain_explicit_dynamics_step requires a supported "
                "hyperelastic material."
            )
        if not hyperelasticity.supports_hyperelastic_study(
            properties,
            dimension=getattr(self.study, "dimension", 0),
            assumption=getattr(self.study, "assumption", None),
        ):
            raise ValueError(
                "The Study dimension/assumption has no formulation for the "
                f"selected hyperelastic material {properties.name!r}."
            )
        if properties.density is None:
            raise ValueError("Finite-strain Explicit requires material density.")
        selected_measure = (
            record.region.measure if record.region is not None else ufl.dx
        )
        selected_state = (
            state if state is not None else problems.second_order_state(target)
        )
        if cohesive_force is not None:
            cohesive_force = cohesive_force.for_displacement(selected_state.u)
        selected_mass = (
            mass
            if mass is not None
            else problems.LumpedMassOperator.assemble(
                _space(target),
                density=properties.density,
                measure=selected_measure,
            )
        )
        if hyperelasticity.is_plane_stress_hyperelastic(properties):
            reference_gradient = np.eye(2)
            membrane_modes = (
                fracture.incremental_wave_speeds(
                    reference_gradient,
                    direction,
                    properties,
                    direction_configuration="reference",
                )
                for direction in ((1.0, 0.0), (0.0, 1.0))
            )
            body_screening_speed = max(
                float(mode.reference_speeds[-1]) for mode in membrane_modes
            )
        else:
            body_screening_speed = fracture.isotropic_reference_wave_speeds(
                properties
            ).pressure
        interface_stability = (
            {}
            if cohesive_force is None
            else cohesive_force.stability_inputs(selected_mass)
        )
        stability = fracture.estimate_stable_time_increment(
            characteristic_length=fracture.minimum_cell_nodal_spacing(
                _domain(self.mesh)
            ),
            dilatational_speed=body_screening_speed,
            safety_factor=stability_safety,
            **interface_stability,
        )
        if dt is None or str(dt).strip().lower() == "auto":
            selected_dt = stability.selected
        else:
            selected_dt = float(dt)
            if selected_dt <= 0.0:
                raise ValueError("Finite-strain Explicit requires dt > 0.")
            if selected_dt > stability.selected:
                raise ValueError(
                    "The requested dt exceeds the current body/interface "
                    "screening limit "
                    f"({selected_dt:.6g} > {stability.selected:.6g}; "
                    f"controller={stability.controller})."
                )
        internal = fracture.finite_strain_internal_force(
            selected_state.u,
            target.test,
            properties,
            measure=selected_measure,
        )
        external = self.external_force(target) if self.loads else None
        residual = self.force_balance(internal=internal, external=external)
        if cohesive_force is not None:
            residual = fracture.FiniteStrainCohesiveResidual(
                residual,
                cohesive_force,
            )
        damping_residual = None
        if float(mass_damping) != 0.0:
            damping_residual = fracture.MassProportionalDampingResidual(
                residual,
                mass=selected_mass,
                velocity=selected_state.v_mid,
                coefficient=mass_damping,
                dt=selected_dt,
            )
            residual = damping_residual
        selected_constraints = self.constraints if constraints is None else constraints
        selected_prescribed = constraint_api.dirichlet_constraints(
            selected_constraints
        )
        base_energy = (
            fracture.FiniteStrainEnergyMonitor(
                mass=selected_mass,
                material=properties,
                measure=selected_measure,
            )
            if cohesive_force is None
            else fracture.FiniteStrainCohesiveEnergyMonitor(
                bulk=fracture.FiniteStrainEnergyMonitor(
                    mass=selected_mass,
                    material=properties,
                    measure=selected_measure,
                ),
                cohesive=cohesive_force,
            )
        )
        if damping_residual is not None:
            base_energy = fracture.DampingEnergyMonitor(
                energy=base_energy,
                damping=damping_residual,
            )
        integrator = time_api.explicit.central_difference(
            state=selected_state,
            mass=selected_mass,
        )
        step = problems.explicit_dynamics(
            state=selected_state,
            integrator=integrator,
            residual=residual,
            stiffness=None,
            study=self.study,
            prescribed=selected_prescribed,
            constraints=selected_constraints,
            update_load=self._time_update_callback(
                update_load,
                include_constraints=False,
            ),
            dt=selected_dt,
            steps=steps,
            save_every=save_every,
            print_every=print_every,
            history_every=history_every,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            history_monitor=fracture.DynamicEnergyLedger(
                energy=base_energy,
                state=selected_state,
                mass=selected_mass,
                residual=residual,
                natural_force=external,
                prescribed=selected_prescribed,
            ),
            stability=stability,
            name=name,
        )
        return self.add_step(step)

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
        """Create Newmark/generalized-alpha structural dynamics."""

        from . import problems
        from . import time as time_api

        self.check(
            target=target,
            step_options={
                "M": M,
                "K": K,
                "F": F,
                "method": method,
                "dt": dt,
                "steps": steps,
            },
        )
        selected_state = (
            state if state is not None else problems.second_order_state(target)
        )
        selected_method = method.lower().replace("-", "_")
        if selected_method == "newmark":
            parameters = time_api.newmark()
        elif selected_method == "generalized_alpha":
            parameters = time_api.generalized_alpha(
                spectral_radius=spectral_radius
            )
        else:
            raise ValueError(
                "Implicit dynamics method must be 'newmark' or 'generalized_alpha'."
            )
        step = problems.implicit_dynamics(
            state=selected_state,
            mass=self.mass(target) if M is None else M,
            damping=C,
            stiffness=self.stiffness(target) if K is None else K,
            force=self.external_force(target) if F is None else F,
            dt=dt,
            steps=steps,
            parameters=parameters,
            study=self.study,
            constraints=(
                self.constraints if constraints is None else constraints
            ),
            solver_options=solver_options,
            update_load=self._time_update_callback(update_load),
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            save_every=save_every,
            print_every=print_every,
            name=name,
        )
        return self.add_step(step)

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

        if (
            study is not None
            and getattr(study, "analysis", None) == "second_order_dynamics"
            and getattr(study, "preferred_procedure", None)
            in {"newmark", "generalized_alpha"}
        ):
            for index, constraint in enumerate(
                constraint_api.dirichlet_constraints(self.constraints)
            ):
                if isinstance(constraint, constraint_api.TimeDependentDirichlet):
                    issues.append(
                        issue(
                            "AFM-CONSTRAINT-002",
                            f"model.constraints[{index}]",
                            (
                                "Time-dependent prescribed supports are not yet "
                                "implemented for implicit structural dynamics."
                            ),
                            hint=(
                                "Use the Explicit procedure or prescribe consistent "
                                "displacement, velocity, and acceleration histories "
                                "through an expert formulation."
                            ),
                        )
                    )

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
            if record.item is material:
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

    def summary(self) -> dict[str, object]:
        return {
            "item": _describe(self.item),
            "region": getattr(self.region, "name", None),
        }


def create(*, study, mesh=None, name: str = "model", units=None) -> Model:
    """Create a lightweight model registry."""

    return Model(study=study, mesh=mesh, name=name, unit_system=units)


def _thermal_expansion_is_zero(material) -> bool:
    selected = getattr(material, "thermal_expansion", 0.0)
    if hasattr(selected, "values"):
        return bool(np.all(np.asarray(selected.values, dtype=float) == 0.0))
    return float(selected or 0.0) == 0.0


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


def _volumetric_heat_capacity(material) -> float:
    if hasattr(material, "volumetric_heat_capacity"):
        return float(material.volumetric_heat_capacity)
    specific_heat = getattr(material, "specific_heat", None)
    if getattr(material, "density", None) is not None and np.isscalar(specific_heat):
        capacity = float(material.density) * float(specific_heat)
        if capacity > 0.0:
            return capacity
    raise ValueError(
        f"Material {_describe(material)!r} does not define one constant "
        "volumetric heat capacity. Use the automatic nonlinear heat Step for "
        "tabulated specific heat."
    )


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
