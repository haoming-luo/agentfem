"""Synthetic small-strain thermal hold/reversal; not calibrated design data."""

from agentfem import (
    amplitudes, constitutive, fields, histories, mesh, models, project,
    solvers, steps, studies,
)


def main():
    run = project.current_run()
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1))
    model = models.create(study=studies.creep_solid(), mesh=domain,
                          name="thermal_hold_and_reversal")
    u = model.field(fields.displacement(domain))
    temperature = model.field(fields.temperature(domain, value=800.))

    # One shared elastic/thermal asset; stress unit MPa, temperature kelvin.
    thermal = constitutive.thermoelastic(
        young=200_000., poisson=0.3, density=1., thermal_expansion=1.e-5,
        conductivity=1., specific_heat=1., reference_temperature=800.,
    )
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

    thermal_path = histories.temperature(temperature)
    thermal_path.record(0.)
    temperature.value.x.array[:] = 900.
    temperature.value.x.scatter_forward()
    thermal_path.record(4.)
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
    }
    run.publish(result)
    return result


if __name__ == "__main__":
    main()
