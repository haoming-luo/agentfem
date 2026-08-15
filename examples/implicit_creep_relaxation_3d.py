"""Global 3D power-law creep relaxation with physical-time cutback."""

from pathlib import Path

from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)


def main() -> dict[str, float]:
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
    )
    model = models.create(study=studies.creep_solid(), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    steel = model.material(
        constitutive.isotropic_power_law(
            young=200.0e3,
            poisson=0.3,
            density=1.0,
            coefficient=1.0e-6,
            stress_exponent=3.0,
            reference_stress=100.0,
        )
    )

    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=0.002,
    )

    output = Path(__file__).resolve().parents[1] / "examples_output" / "implicit_creep"
    step = model.step(
        target=displacement,
        material=steel,
        duration=10.0,
        incrementation=steps.automatic(
            initial=0.25,
            minimum=1.0e-4,
            maximum=0.25,
            maximum_inelastic_increment=1.0e-4,
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        output=output / "relaxation.xdmf",
    )
    simulation = step.solve_result()
    observables = {
        "mean_axial_stress": results.average(
            step.state.stress.function[0, 0], measure=step.state.measure
        ),
        "mean_equivalent_creep_strain": results.average(
            step.state.equivalent_creep_strain.function,
            measure=step.state.measure,
        ),
        "creep_dissipation": simulation.histories["creep_dissipation"].latest,
    }

    simulation.add_quantities(observables, kind="verification_observable")
    required_fields = {"S", "CE", "CEEQ", "MISES", "RF"}
    missing_fields = required_fields.difference(simulation.fields)
    if missing_fields:
        raise RuntimeError(f"Missing standard creep fields: {sorted(missing_fields)}")
    simulation.verify(
        "engineering",
        required_quantities=tuple(observables),
        required_histories=("creep_dissipation", "maximum_creep_increment"),
        required_artifacts=("fields_xdmf", "fields_hdf5"),
    ).require()
    simulation.write_manifest(output / "result.json", include_histories=True)
    print(simulation)
    return observables


if __name__ == "__main__":
    main()
