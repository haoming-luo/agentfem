"""Ramp, hold, then unload a small-strain solid over two physical seconds."""

import argparse
from agentfem import amplitudes, fields, mesh, models, project, steps, studies
from material import StandardLinearSolid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restart", help="Accepted .checkpoint.json from this same project."
    )
    args = parser.parse_args()
    run = project.current_run()
    domain = mesh.cuboid((0.0, 0.0, 0.0), (1.0, 0.2, 0.2), (4, 2, 2))
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(StandardLinearSolid())
    for axis, component, length in (("x", 0, 1.0), ("y", 1, 0.2), ("z", 2, 0.2)):
        for location in (0.0, length):
            model.fix(
                u,
                on=mesh.face(domain, axis=axis, value=location),
                component=component,
                value=0.002 if axis == "x" and location == 1.0 else 0.0,
            )
    step = model.step(
        target=u,
        duration=2.0,
        time_unit="s",
        incrementation=steps.fixed(16),
        amplitude=amplitudes.tabular([0.0, 0.25, 0.75, 1.0], [0.0, 1.0, 1.0, 0.0]),
    )
    if args.restart:
        step.load_checkpoint(args.restart)
    # This checkpoint is a real mid-path restart point, not another solver loop.
    if step.accepted_load_factor < 0.5:
        step.solve(until=0.5)
        step.save_checkpoint(run.artifact("midpoint"))
    result = step.solve_result(output=run.artifact("results.xdmf"))
    step.save_checkpoint(run.artifact("final"))
    result.add_checkpoint(step.checkpoints[-1])
    result.metadata["scientific_scope"] = (
        "Synthetic isothermal small-strain relaxation; not calibrated service-life data."
    )
    run.publish(result)


if __name__ == "__main__":
    main()
