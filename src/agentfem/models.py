"""Lightweight model registries for AgentFEM workflows.

The model layer records mesh, fields, amplitudes, materials, constraints,
loads, and boundary models. It is an audit and validation object, not a solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

from . import constraints as constraint_api
from . import loads as load_api
from ._model_support import describe as _describe
from ._model_support import domain as _domain
from .materials.definitions import MaterialDefinition
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

    def add_material(self, material, *, region=None, orientation=None):
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
        if orientation is not None:
            from .materials.orientations import OrientedMaterial, oriented

            if isinstance(resolved, OrientedMaterial):
                raise ValueError(
                    "Material orientation was supplied twice. Pass an oriented material "
                    "or orientation=..., not both."
                )
            resolved = oriented(resolved, orientation)
        self.materials.append(_WithRegion(resolved, region, definition))
        return resolved

    def material(self, material, *, region=None, orientation=None):
        """Register material data with an optional independent material frame."""

        return self.add_material(material, region=region, orientation=orientation)

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
                raise ValueError(
                    "Pass either component=... or components=..., not both."
                )
            components = component
        selected_value = (
            self._amplitude_by_name(value) if isinstance(value, str) else value
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
        return self.add_load(self._with_amplitude(load, amplitude))

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
            created.append(self.add_load(self._with_amplitude(load, amplitude)))
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
        """Register a distributed normal, isotropic, or matrix spring support."""

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

        This model-first facade preserves the readable public workflow.  The
        operator layer owns regional measure resolution and form composition.
        """

        from .operators import _model_lowering

        return _model_lowering.lower_stiffness(
            target,
            assignments=tuple(self.materials),
            selected=(
                self._material_record(material) if material is not None else None
            ),
            study=study or self.study,
            measure=measure,
            law=law,
            temperature=temperature,
            name=name,
        )

    def mass(
        self,
        target,
        material=None,
        *,
        measure=None,
        name: str = "M",
    ):
        """Create a consistent mass operator from registered densities."""

        from .operators import _model_lowering

        return _model_lowering.lower_mass(
            target,
            assignments=tuple(self.materials),
            selected=(
                self._material_record(material) if material is not None else None
            ),
            study=self.study,
            measure=measure,
            name=name,
        )

    def damping(self, target, coefficient, *, measure=None, name: str = "C"):
        """Create a viscous damping operator."""

        from .operators import _model_lowering

        return _model_lowering.lower_damping(
            target,
            coefficient,
            study=self.study,
            measure=measure,
            name=name,
        )

    def conduction(self, temperature, material=None, *, measure=None, name: str = "K"):
        """Create a region-aware heat-conduction operator.

        A single material may occupy the whole mesh.  Multiple materials must
        each own a cell region, matching the semantics already used by
        :meth:`stiffness` and :meth:`mass`.
        """

        from .operators import _model_lowering

        return _model_lowering.lower_conduction(
            temperature,
            assignments=tuple(self.materials),
            selected=(
                self._material_record(material) if material is not None else None
            ),
            measure=measure,
            name=name,
        )

    def heat_capacity(
        self, temperature, material=None, *, measure=None, name: str = "C"
    ):
        """Create region-aware ``rho c_p`` heat capacity."""

        from .operators import _model_lowering

        return _model_lowering.lower_heat_capacity(
            temperature,
            assignments=tuple(self.materials),
            selected=(
                self._material_record(material) if material is not None else None
            ),
            measure=measure,
            name=name,
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

        record = (
            self._material_record(material)
            if material is not None
            else _single_material(self, "model.thermal_expansion")
        )

        from .operators import _model_lowering

        return _model_lowering.lower_thermal_expansion(
            target,
            temperature,
            selected=record,
            study=self.study,
            measure=measure,
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

        from .operators import _model_lowering

        return _model_lowering.lower_lumped_mass(
            target,
            assignments=tuple(self.materials),
            selected=(
                self._material_record(material) if material is not None else None
            ),
            study=self.study,
            measure=measure,
            method=method,
        )

    def load_vector(self, target, loads=None, *, load=None):
        """Create a total load vector from registered or explicit loads."""

        from .operators import _model_lowering

        selected_loads = self.loads if loads is None and load is None else loads
        return _model_lowering.lower_load_vector(
            target,
            loads=selected_loads,
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

        from .operators import _model_lowering

        return _model_lowering.lower_internal_force(
            displacement,
            test_function,
            assignments=tuple(self.materials),
            selected=(
                self._material_record(material) if material is not None else None
            ),
            study=study or self.study,
            measure=measure,
            name=name,
        )

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

        from .operators import _model_lowering

        return _model_lowering.lower_boundary_force(
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

        from .operators import _model_lowering

        return _model_lowering.lower_force_balance(
            internal=internal,
            external=external,
            damping=damping,
            absorbing=absorbing,
            boundary=boundary,
            name=name,
            convention=convention,
        )

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
        M=None,
        C=None,
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
            raise ValueError(
                "model.step requires kind=... or a study with an analysis."
            )
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
            "M": M,
            "C": C,
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

        from ._model_validation import validate_model

        return validate_model(self, target=target, step_options=step_options)

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

        from ._model_inspection import model_summary

        return model_summary(self)

    def manifest(self) -> dict[str, object]:
        """Return a stable machine-readable model manifest."""

        from ._model_inspection import model_manifest

        return model_manifest(self)

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

        from ._model_inspection import model_to_ir

        return model_to_ir(
            self,
            include_validation=include_validation,
            metadata=metadata,
        )

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

        from ._model_inspection import write_model_ir

        return write_model_ir(
            self,
            path,
            include_validation=include_validation,
            metadata=metadata,
        )

    def tree(self) -> str:
        """Return a compact text model tree for logs, notebooks, and agents."""

        from ._model_inspection import model_tree

        return model_tree(self)


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


def _single_material(model: Model, caller: str) -> "_WithRegion":
    if len(model.materials) != 1:
        raise ValueError(f"{caller} requires material=... or exactly one material.")
    return model.materials[0]


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


def _time_dependent_fix(
    target, *, on=None, location=None, value, components=None, name=None
):
    selected_location = location if location is not None else on
    label = (
        name or f"time_dependent_fixed_{getattr(selected_location, 'name', 'location')}"
    )
    if components is None:
        components = _all_components_or_none(target)
    if components is None:
        return constraint_api.time_dependent_scalar_dirichlet(
            target,
            on=selected_location,
            value=value,
            name=label,
        )
    component_ids = (
        (int(components),) if isinstance(components, Integral) else tuple(components)
    )
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
