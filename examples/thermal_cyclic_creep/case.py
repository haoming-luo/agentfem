"""Synthetic small-strain thermal hold/reversal; not calibrated design data."""

from agentfem import (
    amplitudes, constitutive, fields, loads, mesh, models, project,
    solvers, steps, studies, units,
)


def main():
    run = project.current_run()
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1))
    model = models.create(study=studies.creep_solid(), mesh=domain,
                          name="thermal_hold_and_reversal", units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))

    # One shared elastic/thermal asset; stress unit MPa, temperature kelvin.
    thermal = constitutive.thermoelastic(
        young=200_000., poisson=0.3, density=1., thermal_expansion=1.e-5,
        conductivity=1., specific_heat=1., reference_temperature=800.,
    )
    # Solve heat on the same mesh first. An insulated homogeneous cube with
    # rho*c=1 and source=25 has T(t)=800+25t. Only accepted thermal frames
    # drive mechanics; dissipative mechanical heating is not fed back.
    heat = models.create(study=studies.transient_heat_transfer(dimension=3),
                         mesh=domain, name="uniform_heating", units=units.n_mm_mpa())
    temperature = heat.field(fields.temperature(domain, value=800.))
    heat.material(thermal)
    heat.load(loads.heat_source(25., target=temperature))
    heat_step = heat.step(target=temperature, dt=.25, steps=16, progress=False)
    thermal_path = heat_step.capture_history(name="temperature", unit="K")
    heat_step.run()
    creep = constitutive.isotropic_arrhenius_power_law(
        elastic=thermal, coefficient=1.e-6, stress_exponent=3., reference_stress=100.,
        activation_energy=120_000., reference_temperature=800.,
    )
    cyclic = constitutive.ChabocheCombinedHardening(
        young=thermal.young, poisson=thermal.poisson, yield_stress=200.,
        backstress_moduli=(12_000., 3_000.), dynamic_recovery=(100., 8.),
        isotropic_saturation=60., isotropic_rate=10.,
    )
    model.material(constitutive.CreepPlasticityMaterial(creep=creep, plastic=cyclic))

    loading = amplitudes.tabular([0., 1., 2., 3., 4.], [0., 1., 1., -0.5, -0.5])
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1),
                  component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=0.004)

    step = model.step(
        target=u, duration=4., amplitude=loading, temperature=thermal_path,
        incrementation=steps.fixed(16),
        solver_options=solvers.newton(maximum_iterations=25, line_search="backtracking"),
    )
    result = step.solve_result(output=run.artifact("results.xdmf"))
    step.save_checkpoint(run.artifact("final.npz"))
    result.add_checkpoint(step.checkpoints[-1])
    result.metadata["scientific_scope"] = {
        "parameters": "synthetic, not calibrated",
        "verification": "small-strain serial 3D demonstration",
        "thermal_energy": "not a full first-law balance",
        "thermal_transfer": "accepted same-mesh heat solution; one-way only",
    }
    run.publish(result)
    return result


if __name__ == "__main__":
    main()
