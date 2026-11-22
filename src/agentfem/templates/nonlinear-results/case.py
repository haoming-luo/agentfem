# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial SI displacement-controlled hyperelastic patch and accepted path."""
from pathlib import Path
import json
import sys
from mpi4py import MPI
from agentfem import constitutive, fields, mechanics, mesh, models, project, provenance, results, steps, studies, units


def main(*, review=False):
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (2, 1, 1), comm=MPI.COMM_SELF,
        cell_type="hexahedron")
    model = models.create(mesh=domain, study=studies.static_solid(dimension=3, nonlinear=True), units=units.si())
    u = model.field(fields.displacement(domain))
    model.material(constitutive.neo_hookean(young=1e6, poisson=.3))
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component+1), component=component)
    loaded = mesh.face(domain, axis="x", value=1., tag=4, name="displaced_end")
    model.fix(u, on=loaded, component=0, value=.1)
    options = dict(incrementation=steps.fixed(4), progress=False)
    if review:
        card = model.review(target=u, step_options=options, output_fields=(u, "S"),
            output_directory=Path(__file__).parent/"outputs")
        if domain.comm.rank == 0:
            print(json.dumps(card, ensure_ascii=False, allow_nan=False))
        return card
    run = project.current_run(project_root=Path(__file__).parent)
    output = results.OutputPlan(directory=run.output_directory,
        field=results.field_output("U", "S", every=1, configuration="reference"), basename="fields",
        write_manifest=False)  # RunContext publishes once, after curve attachment.
    settings = provenance.model_settings(model, target=u, step_options=options, source_files=(Path(__file__),))
    step = model.step(target=u, output=output, **options)
    result = step.solve_result()
    result.metadata["model_settings"] = settings
    path = mechanics.displacement_controlled_response(model, step, on=loaded, component=0, monitor_component=0)
    result.add_history("loaded_end_reaction", path["displacement"], path["reaction"], unit="N",
        abscissa_name="mean_loaded_end_displacement", abscissa_unit="m",
        description="Loaded support-on-body +x reaction at retained accepted stations; displacement is an area mean.")
    result.add_history("load_factor", path["displacement"], path["load_factor"], unit="1",
        abscissa_name="mean_loaded_end_displacement", abscissa_unit="m")
    result.verify(required_histories=("loaded_end_reaction", "load_factor"))
    run.publish(result)
    return result


if __name__ == "__main__":
    main(review="--review" in sys.argv[1:])
