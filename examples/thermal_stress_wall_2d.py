"""Sequential heat-transfer and thermal-stress analysis of a hot wall."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import __version__, constitutive, eigenstrains, fields, mesh, models, studies
from agentfem.provenance import collective_call


def main(*, output_dir=None, resume_heat=False, smoke=False):
    """Resume accepted heat independently of the downstream elastic solve."""
    output_dir = Path(output_dir or "outputs/thermal_stress_wall")
    checkpoint = output_dir / "accepted_heat.checkpoint.json"
    comm = MPI.COMM_WORLD
    recipe = {
        "example_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "agentfem_version": __version__, "smoke": smoke,
    }
    if not resume_heat and comm.bcast(checkpoint.exists() if comm.rank == 0 else None, root=0):
        raise FileExistsError("Accepted heat exists: use --resume-heat or a new --output directory.")
    domain = mesh.rectangle(
        (0.0, 0.0),
        (0.12, 1.0),
        (4, 6) if smoke else (24, 60),
        comm=comm,
        cell_type="quadrilateral",
    )
    steel = constitutive.thermoelastic(
        name="hot-section steel",
        young=180.0e9,
        poisson=0.3,
        density=7800.0,
        thermal_expansion=13.0e-6,
        conductivity=32.0,
        specific_heat=560.0,
        reference_temperature=573.15,
    )
    hot = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="hot_face", tag=1
    )
    cold = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.12), name="cold_face", tag=2
    )

    heat = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
        name="wall_heat_transfer",
    )
    temperature = heat.field(fields.temperature(domain, value=573.15))
    heat.material(steel)
    heat.fix(temperature, on=hot, value=823.15)
    heat.fix(temperature, on=cold, value=573.15)
    heat_step = heat.step(
        target=temperature,
        dt=60.0,
        steps=3 if smoke else 60,
        save_every=5,
        output=output_dir / "thermal_stress_wall_temperature.xdmf",
    )
    try:
        if resume_heat:
            def require_same_recipe():
                if comm.rank == 0:
                    previous = json.loads((output_dir / "heat.result.json").read_text())
                    if previous["metadata"].get("restart_recipe") != recipe:
                        raise ValueError("Heat restart recipe changed; use a new output directory.")
            collective_call(require_same_recipe, comm=comm, label="validate upstream recipe")
            heat_step.load_checkpoint(checkpoint)
            if heat_step.completed_steps != (3 if smoke else 60):
                raise ValueError("Resume requires the completed upstream stage, not a partial heat run.")
        else:
            heat_result = heat_step.solve_result()
            heat_result.verify("engineering").require()
            heat_step.save_checkpoint(checkpoint, portable=True)
            heat_result.metadata["restart_recipe"] = recipe
            collective_call(
                lambda: heat_result.write_manifest(output_dir / "heat.result.json")
                if comm.rank == 0 else None,
                comm=comm, label="publish accepted heat result",
            )
        accepted_time = heat_step.completed_steps * heat_step.dt
        thermal_solves = heat_step.operator_lifecycle_summary()["solve_count"]
    finally:
        heat_step.close()
    digest = collective_call(
        lambda: sha256(checkpoint.read_bytes()).hexdigest() if comm.rank == 0 else None,
        comm=comm, label="identify upstream checkpoint",
    )
    digest = comm.bcast(digest, root=0)

    mechanics = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="wall_thermal_stress",
    )
    displacement = mechanics.field(fields.displacement(domain))
    mechanics.material(steel)
    bottom = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="bottom", tag=3
    )
    mechanics.fix(displacement, on=bottom, component=1, value=0.0)
    mechanics.fix(displacement, on=cold, component=0, value=0.0)
    structural_temperature = fields.temperature(domain, value=573.15)
    mechanics.eigenstrain(eigenstrains.thermal(structural_temperature))
    stage = mechanics.stage("accepted-heat-to-solid")
    stage.predefine(structural_temperature, temperature, method="interpolate",
                    source_time=accepted_time, target_time=accepted_time)
    with stage.field_transaction(displacement=displacement):
        mechanics_result = mechanics.step(
            target=displacement, configuration=stage,
            output=output_dir / "thermal_stress_wall_displacement.xdmf",
        ).solve_result()
        mechanics_result.verify("engineering").require()
    mechanics_result.metadata["upstream_stage"] = {
        "name": "wall_heat_transfer", "checkpoint": checkpoint.name,
        "manifest_sha256": digest, "accepted_time": accepted_time,
        "resumed": resume_heat, "thermal_solves_this_execution": thermal_solves,
        "recovery_policy": "restore_heat_then_recompute_elastic_structure",
    }
    collective_call(
        lambda: mechanics_result.write_manifest(output_dir / "mechanics.result.json")
        if comm.rank == 0 else None,
        comm=comm, label="publish structural result",
    )
    if comm.rank == 0:
        print(f"Thermal solves this execution: {thermal_solves}; result: {output_dir / 'mechanics.result.json'}")
    return mechanics_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/thermal_stress_wall"))
    parser.add_argument("--resume-heat", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use a small mesh and three heat increments.")
    args = parser.parse_args()
    main(output_dir=args.output, resume_heat=args.resume_heat, smoke=args.smoke)
