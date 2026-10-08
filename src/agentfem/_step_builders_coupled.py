# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded coupled lowering; Model remains an engineering asset registry."""

from .provenance import collective_call


def thermoelastic(model, request):
    from dolfinx import fem
    from .materials.properties import ThermoElasticIsotropicProperties
    from .operators.identity import _coupled_input_identity
    from ._step_provider_support import selected_material
    from .time.thermoelastic import _ThermoelasticStep
    from .constraints import constraint_assets

    options = dict(request.options)
    u = request.target
    theta = options.pop("temperature_departure")
    material = selected_material(model, request)
    heat_assets = tuple(options.pop("heat_loads", ()))
    mechanical_assets = tuple(options.pop("mechanical_loads", ()))
    boundaries = options.pop("constraints", None)
    boundaries = constraint_assets(
        model.constraints if boundaries is None else boundaries
    )

    def preflight():
        if not isinstance(material, ThermoElasticIsotropicProperties):
            raise ValueError(
                "AFM-COUPLING-011: constant isotropic thermoelastic material required."
            )
        if len(model.materials) > 1 or any(
            r.region is not None for r in model.materials
        ):
            raise ValueError(
                "AFM-COUPLING-011: regional coupled materials are not supported."
            )
        if model.boundary_models or model.eigenstrains:
            raise ValueError(
                "AFM-COUPLING-011: boundary models and additional eigenstrains are unsupported."
            )
        units = model.unit_system
        if units is None or (
            units.length,
            units.mass,
            units.time,
            units.temperature,
        ) != ("m", "kg", "s", "K"):
            raise ValueError(
                "AFM-COUPLING-011: declare units.si(); numeric conversion is not automatic."
            )
        if theta.space.mesh is not u.space.mesh or model.domain is not u.space.mesh:
            raise ValueError(
                "AFM-COUPLING-011: both participants must use the Model mesh."
            )
        if options.get("K") is not None or options.get("F") is not None:
            raise ValueError(
                "AFM-COUPLING-011: use explicit heat_loads and mechanical_loads; custom K/F is unsupported."
            )
        if any(
            not any(item is selected for selected in (*heat_assets, *mechanical_assets))
            for item in model.loads
        ):
            raise ValueError(
                "AFM-COUPLING-011: assign every registered load explicitly to heat_loads or mechanical_loads."
            )
        if len({id(x) for x in (*heat_assets, *mechanical_assets)}) != len(
            heat_assets
        ) + len(mechanical_assets):
            raise ValueError("AFM-COUPLING-011: duplicate participant load assignment.")
        for item in (*heat_assets, *mechanical_assets):
            if hasattr(item, "update") or hasattr(item, "amplitude"):
                raise ValueError(
                    "AFM-COUPLING-011: time-dependent natural loads are not supported yet."
                )
        mechanical_bcs, thermal_bcs = [], []
        for item in boundaries:
            bc = getattr(item, "bc", item)
            if not isinstance(bc, fem.DirichletBC):
                raise ValueError(
                    "AFM-COUPLING-007: coupled route requires strong Dirichlet boundaries."
                )
            if u.space._cpp_object.contains(bc.function_space):
                mechanical_bcs.append(item)
            elif theta.space._cpp_object.contains(bc.function_space):
                thermal_bcs.append(item)
            else:
                raise ValueError(
                    "AFM-COUPLING-007: boundary belongs to neither participant."
                )
        return mechanical_bcs, thermal_bcs

    mechanical_bcs, thermal_bcs = collective_call(
        preflight, comm=u.space.mesh.comm, label="preflight coupled Step"
    )

    def combined(assets, field):
        forms = [item.form(field.test) for item in assets]
        return sum(forms[1:], forms[0]) if forms else None

    heat_load = collective_call(
        lambda: combined(heat_assets, theta),
        comm=u.space.mesh.comm,
        label="lower heat loads",
    )
    mechanical_load = collective_call(
        lambda: combined(mechanical_assets, u),
        comm=u.space.mesh.comm,
        label="lower mechanical loads",
    )
    identity = _coupled_input_identity(
        (
            ("displacement", u.value, mechanical_bcs),
            ("temperature_departure", theta.value, thermal_bcs),
        ),
        {"heat": heat_load, "mechanical": mechanical_load},
        material=material,
        unit_system=model.unit_system,
    )
    iteration = {
        key: options.pop(key)
        for key in (
            "relaxation",
            "rtol",
            "max_iterations",
            "displacement_atol",
            "temperature_atol",
        )
        if key in options
    }
    for key in ("K", "F", "material", "output"):
        options.pop(key, None)
    options["name"] = options.get("name") or "coupled_thermoelastic"
    created = _ThermoelasticStep(
        u,
        theta,
        material=material,
        mechanical_bcs=mechanical_bcs,
        thermal_bcs=thermal_bcs,
        heat_load=heat_load,
        mechanical_load=mechanical_load,
        input_identity=identity,
        iteration_options=iteration,
        **options,
    )
    return model.add_step(created)
