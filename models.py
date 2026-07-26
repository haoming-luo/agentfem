"""Lightweight model registries for AgentFEM workflows.

The model layer records mesh, fields, amplitudes, materials, constraints,
loads, and boundary models. It is an audit and validation object, not a solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

from . import constraints as constraint_api
from . import loads as load_api


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
        if _is_amplitude_like(value):
            return self.add_constraint(
                _time_dependent_fix(
                    target,
                    on=on,
                    location=location,
                    value=value,
                    components=components,
                    name=name,
                )
            )
        return self.add_constraint(
            constraint_api.fixed(
                target,
                on=on,
                location=location,
                value=value,
                components=components,
                name=name,
            )
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

    def load(self, load):
        """Register a natural load/source and return it."""

        return self.add_load(load)

    def traction(self, value, *, on=None, location=None, name: str = "traction"):
        """Create and register a mechanical traction load."""

        return self.add_load(
            load_api.traction(value, on=on, location=location, name=name)
        )

    def body_force(self, value, *, domain=None, target=None, measure=None, name: str = "body_force"):
        """Create and register a mechanical body-force load."""

        kwargs = {"domain": domain or self.mesh, "target": target, "name": name}
        if measure is not None:
            kwargs["measure"] = measure
        return self.add_load(load_api.body_force(value, **kwargs))

    def heat_flux(self, value, *, on=None, location=None, name: str = "heat_flux"):
        """Create and register a prescribed heat-flux load."""

        return self.add_load(load_api.heat_flux(value, on=on, location=location, name=name))

    def heat_source(self, value, *, domain=None, target=None, measure=None, name: str = "heat_source"):
        """Create and register a volumetric heat-source load."""

        kwargs = {"domain": domain or self.mesh, "target": target, "name": name}
        if measure is not None:
            kwargs["measure"] = measure
        return self.add_load(load_api.heat_source(value, **kwargs))

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
            )
        )

    def stiffness(
        self,
        target,
        material=None,
        *,
        measure=None,
        law=None,
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
                    name=f"K_{getattr(record.region, 'name', len(parts))}",
                )
            )
        if missing:
            raise ValueError(
                "Multiple-material stiffness requires every material to have a region. "
                f"Materials without regions: {missing}."
            )
        return operators.combine(*parts, name=name, kind="partitioned_stiffness")

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

        self.steps.append(step)
        return step

    def step(
        self,
        *,
        kind: str | None = None,
        target=None,
        K=None,
        F=None,
        constraints=None,
        solver_options=None,
        name: str | None = None,
        **kwargs,
    ):
        """Create and register an analysis step.

        ``Step`` is the high-level workflow object for users and agents. It
        records the analysis intent while keeping operator-level objects visible
        instead of hiding the model behind a monolithic solver call.
        """

        selected_kind = kind or getattr(self.study, "analysis", None)
        if selected_kind is None:
            raise ValueError("model.step requires kind=... or a study with an analysis.")
        normalized = selected_kind.lower().replace("-", "_")
        if normalized in {"linear_static", "static"}:
            return self.linear_static_step(
                target=target,
                K=K,
                F=F,
                constraints=constraints,
                solver_options=solver_options,
                name=name or "linear_static",
                **kwargs,
            )
        if normalized in {"second_order_dynamics", "explicit_dynamics", "explicit"}:
            return self.explicit_dynamics_step(
                target=target,
                constraints=constraints,
                name=name or "explicit_dynamics",
                **kwargs,
            )
        raise ValueError(f"Unsupported model step kind {selected_kind!r}.")

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

        from . import problems

        self.check()
        K = K if K is not None else self.stiffness(target)
        F = F if F is not None else self.external_force(target)
        step = problems.linear_static(
            K,
            F,
            study=self.study,
            unknown=target,
            constraints=self.constraints if constraints is None else constraints,
            solver_options=solver_options,
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
        prescribed=(),
        constraints=None,
        save_every: int = 1,
        print_every: int = 1,
        name: str = "explicit_dynamics",
    ):
        """Create and register a second-order explicit dynamics step."""

        from . import problems
        from . import time as time_api

        self.check()
        selected_state = state if state is not None else problems.second_order_state(target)
        selected_mass = mass if mass is not None else self.lumped_mass(target)
        integrator = time_api.explicit.central_difference(
            state=selected_state,
            mass=selected_mass,
        )
        if residual is None:
            residual = self.force_balance(internal=self.internal_force(selected_state.u))
        step = problems.explicit_dynamics(
            state=selected_state,
            integrator=integrator,
            residual=residual,
            study=self.study,
            prescribed=prescribed,
            constraints=self.constraints if constraints is None else constraints,
            dt=dt,
            steps=steps,
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

    def check(self) -> None:
        """Run lightweight modeling checks without assembling a system."""

        if self.study is None:
            raise ValueError("Model.check requires a study.")
        if self.mesh is None:
            raise ValueError("Model.check requires a mesh.")
        if self.mesh.geometry.dim != self.study.dimension:
            raise ValueError(
                f"Study dimension {self.study.dimension} does not match mesh "
                f"geometric dimension {self.mesh.geometry.dim}."
            )
        if not self.fields:
            raise ValueError("Model.check requires at least one field.")
        if getattr(self.study, "is_solid_mechanics", False) and not self.materials:
            raise ValueError("Solid-mechanics models require at least one material.")

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

    def summary(self) -> dict[str, object]:
        """Return an agent-readable model summary."""

        return {
            "name": self.name,
            "study": _describe(self.study),
            "mesh": _mesh_summary(self.mesh),
            "fields": tuple(_describe(item) for item in self.fields),
            "amplitudes": tuple(_describe(item) for item in self.amplitudes),
            "materials": tuple(_describe(item) for item in self.materials),
            "constraints": tuple(_describe(item) for item in self.constraints),
            "loads": tuple(_describe(item) for item in self.loads),
            "boundary_models": tuple(_describe(item) for item in self.boundary_models),
            "regions": tuple(_describe(item) for item in self.regions),
            "steps": tuple(_describe(item) for item in self.steps),
        }

    def manifest(self) -> dict[str, object]:
        """Return a stable machine-readable model manifest."""

        return {
            "kind": "agentfem_model_manifest",
            "version": 1,
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
                "steps",
            ),
        }

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


def create(*, study, mesh=None, name: str = "model") -> Model:
    """Create a lightweight model registry."""

    return Model(study=study, mesh=mesh, name=name)


def _stiffness_from_record(
    target,
    record: _WithRegion,
    *,
    operators,
    study,
    measure=None,
    law=None,
    name: str = "K",
):
    selected_measure = measure
    if selected_measure is None and record.region is not None:
        selected_measure = record.region.measure
    kwargs = {"study": study}
    if law is not None:
        kwargs["law"] = law
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
    selected_measure = measure
    if selected_measure is None and record.region is not None:
        selected_measure = record.region.measure
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
    return {
        "topological_dim": mesh.topology.dim,
        "geometric_dim": mesh.geometry.dim,
    }


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
    if len(component_ids) != 1:
        raise ValueError(
            "Time-dependent model.fix currently supports one component at a time. "
            "Call model.fix(..., components=component_id, value=amplitude)."
        )
    return constraint_api.time_dependent_component_dirichlet(
        target,
        component=component_ids[0],
        on=selected_location,
        value=value,
        name=label,
    )


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
