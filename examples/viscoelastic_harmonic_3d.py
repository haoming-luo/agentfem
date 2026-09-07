"""Direct harmonic response of a three-dimensional viscoelastic bar."""

from pathlib import Path

from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, results, studies


length = 2.0
frequency = 0.25
domain = mesh.cuboid(
    (0.0, 0.0, 0.0),
    (length, 1.0, 1.0),
    (12, 2, 2),
    cell_type="hexahedron",
)
model = models.create(
    study=studies.harmonic_solid(dimension=3),
    mesh=domain,
    name="viscoelastic_harmonic_bar",
)
u = model.field(fields.displacement(domain))
material = model.material(
    constitutive.IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1.0e6,
        instantaneous_poisson_ratio=0.30,
        shear_relaxation_ratios=(0.25, 0.15),
        bulk_relaxation_ratios=(0.25, 0.15),
        relaxation_times=(0.2, 2.0),
        name="two_branch_polymer",
    )
)

model.fix(u, on=mesh.face(domain, axis="x", value=0.0), component=0)
model.fix(u, on=mesh.face(domain, axis="y", value=0.0), component=1)
model.fix(u, on=mesh.face(domain, axis="z", value=0.0), component=2)
loaded_end = mesh.face(domain, axis="x", value=length)
model.traction((1.0e3, 0.0, 0.0), on=loaded_end)

step = model.step(
    target=u,
    frequency=frequency,
    density=1000.0,
)
output = Path("outputs/viscoelastic_harmonic_3d")
simulation = step.solve_result(output=output / "fields.xdmf")

tip_real = results.average(step.solution_real[0], measure=loaded_end.measure)
tip_imaginary = results.average(
    step.solution_imaginary[0],
    measure=loaded_end.measure,
)
if MPI.COMM_WORLD.rank == 0:
    output.mkdir(parents=True, exist_ok=True)
    simulation.write_manifest(output / "result.json")
    print(f"Tip displacement phasor: {tip_real:+.6e}{tip_imaginary:+.6e}j")
    print(f"Result: {output / 'result.json'}")
