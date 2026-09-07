"""Three-dimensional generalized-Maxwell stress relaxation."""

from pathlib import Path

from mpi4py import MPI

from agentfem import (
    amplitudes,
    checkpointing,
    constitutive,
    fields,
    mesh,
    models,
    results,
    steps,
    studies,
)


domain = mesh.cuboid(
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (2, 2, 2),
    cell_type="tetrahedron",
)
model = models.create(
    study=studies.viscoelastic_solid(dimension=3),
    mesh=domain,
    name="viscoelastic_relaxation",
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
model.fix(
    u,
    on=mesh.face(domain, axis="x", value=1.0),
    component=0,
    value=0.01,
)

load_then_hold = amplitudes.tabular(
    times=(0.0, 0.5, 5.0),
    values=(0.0, 1.0, 1.0),
    name="load_then_hold",
)
step = model.step(
    target=u,
    material=material,
    duration=5.0,
    incrementation=steps.automatic(
        initial=0.1,
        minimum=1.0e-4,
        maximum=0.25,
        max_increments=100,
        max_cutbacks=8,
        cutback_factor=0.5,
    ),
    time_error_tolerance=1.0e-3,
    amplitude=load_then_hold,
    checkpoint=checkpointing.every(
        10,
        directory="outputs/viscoelastic_relaxation_3d/checkpoints",
        keep_last=2,
    ),
)

output = Path("outputs/viscoelastic_relaxation_3d")
simulation = step.solve_result(output=output / "fields.xdmf")
mean_stress = results.average(
    step.state.stress.function[0, 0],
    measure=step.state.measure,
)
if MPI.COMM_WORLD.rank == 0:
    output.mkdir(parents=True, exist_ok=True)
    simulation.write_manifest(output / "result.json")
    print(f"Final mean axial stress: {mean_stress:.6g}")
    print(f"Result: {output / 'result.json'}")
