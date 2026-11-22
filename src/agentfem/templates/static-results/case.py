# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""SI 3D cantilever: fields, a spatial point and the fixed-end reaction."""
from pathlib import Path
import json
import sys
from mpi4py import MPI
from agentfem import constitutive, fields, mesh, models, project, provenance, results, studies, units


def main(*, review=False):
    domain = mesh.cuboid((0., 0., 0.), (1., .2, .2), (6, 2, 2), comm=MPI.COMM_WORLD)
    study = studies.static_solid(dimension=3)
    model = models.create(mesh=domain, study=study, units=units.si())
    u = model.field(fields.displacement(domain, degree=2))
    material = constitutive.isotropic_elastic(young=210e9, poisson=.3, density=7800.)
    model.material(material)
    fixed = mesh.face(domain, axis="x", value=0., tag=1, name="fixed")
    loaded = mesh.face(domain, axis="x", value=1., tag=2, name="loaded")
    model.clamp(u, on=fixed)
    model.traction((0., -1e6, 0.), on=loaded)
    if review:
        card = model.review(target=u)
        if domain.comm.rank == 0:
            print(json.dumps(card, ensure_ascii=False, allow_nan=False))
        return card
    run = project.current_run(project_root=Path(__file__).parent)
    settings = provenance.model_settings(model, target=u, source_files=(Path(__file__),))
    step = model.step(target=u)
    result = step.solve_result(field_variables=("S",))
    result.metadata["model_settings"] = settings
    result.add_quantity("tip_U2", results.probe(u, at=(1., .1, .1))[1], unit="m",
        description="Interpolated displacement at the specified reference point.")
    result.add_quantity("fixed_reaction", results.reaction_resultant(step, on=fixed), unit="N",
        description="Support-on-body force at the fixed end, global xyz components.")
    results.complete_result(step, result, output=run.artifact("fields.xdmf"), strict_output=True)
    result.verify(required_quantities=("tip_U2", "fixed_reaction"), required_artifacts=("fields_xdmf", "fields_hdf5"))
    run.publish(result, comm=MPI.COMM_WORLD)
    return result


if __name__ == "__main__":
    main(review="--review" in sys.argv[1:])
