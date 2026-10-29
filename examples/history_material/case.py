"""One structural workflow, two replaceable local history laws."""

import argparse

from agentfem import (
    amplitudes,
    checkpointing,
    constitutive,
    fields,
    mesh,
    models,
    project,
    steps,
    studies,
)
from material import HistoryMaterial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material", choices=("viscoelastic", "damage"), default="viscoelastic"
    )
    parser.add_argument("--restart", default=None)
    parser.add_argument(
        "--until",
        type=float,
        default=1.0,
        help="Stop at this normalized path coordinate (0 < value <= 1).",
    )
    parser.add_argument(
        "--assumption", choices=("plane_strain", "plane_stress"), default="plane_strain"
    )
    args = parser.parse_args()
    run = project.current_run()
    domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (8, 8))
    model = models.create(
        study=studies.static_solid(
            dimension=2, assumption=args.assumption, nonlinear=True
        ),
        mesh=domain,
        name="history_material_plate",
    )
    u = model.field(fields.displacement(domain))
    model.material(HistoryMaterial(mechanism=args.material))
    model.clamp(u, on=mesh.face(domain, axis="x", value=0.0))
    model.fix(u, on=mesh.face(domain, axis="x", value=1.0), component=0, value=0.12)
    cycle = amplitudes.tabular((0.0, 0.25, 0.5, 0.75, 1.0), (0.0, 1.0, 1.0, -0.2, 0.0))
    step = model.step(
        target=u,
        duration=2.0,
        time_unit="s",
        amplitude=cycle,
        incrementation=steps.fixed(16),
        output=run.artifact("results.xdmf"),
        checkpoint=checkpointing.every(
            4, directory=run.artifact("checkpoints"), keep_last=2
        ),
        plane_stress=(
            constitutive.PlaneStressReduction(symmetry="reflection_symmetric")
            if args.assumption == "plane_stress"
            else None
        ),
    )
    if args.restart:
        step.load_checkpoint(args.restart)
    result = step.solve_result(until=args.until)
    checkpoint = step.save_checkpoint(
        run.artifact("final.npz" if result.status == "completed" else "accepted.npz")
    )
    result.add_checkpoint(step.checkpoints[-1])
    result.add_artifact("restart", checkpoint)
    result.metadata["scientific_scope"] = (
        f"Educational finite-strain {args.assumption}; unit reference thickness; synthetic material, not hydrogel calibration or general energy closure."
    )
    run.publish(result)


if __name__ == "__main__":
    main()
