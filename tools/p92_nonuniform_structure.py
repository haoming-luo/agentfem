"""Nonuniform bending/hold composition of the P92 research tensor law.

This is a spatial/time numerical screen, not an experimental component or
lifetime validation. The ordinary AgentFEM Step owns the solve and restart.
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI

from agentfem import amplitudes, fields, mesh, models, results, solvers, steps, studies, units
from tools.p92_tensor_material_reference import P92TensorResearchMaterial

SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def build(*, resolution=1, refinement=1):
    for value in (resolution, refinement):
        if type(value) is not int or value < 1:
            raise ValueError("resolution and refinement must be positive integers")
    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError("This physical-time research screen is serial")
    domain = mesh.cuboid((0., -.5, -.5), (4., .5, .5),
                         (4 * resolution, resolution, resolution), cell_type="hexahedron")
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True),
                          mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    model.material(P92TensorResearchMaterial(temperature_C=650, poisson=.3))
    left, right = mesh.face(domain, axis="x", value=0.), mesh.face(domain, axis="x", value=4.)
    for component in range(3):
        model.fix(u, on=left, component=component)
    model.fix(u, on=right, component=1, value=.02)
    times = np.concatenate((np.linspace(0, 4, 8 * refinement + 1)[1:],
                            np.linspace(4, 64, 12 * refinement + 1)[1:]))
    step = model.step(
        target=u, duration=64., time_unit="s", temperature=923.15,
        material_points=(results.material_point("sample", cell=0),), progress=False,
        amplitude=amplitudes.tabular([0., 4 / 64, 1.], [0., 1., 1.]),
        incrementation=steps.at(*(times / 64)),
        solver_options=solvers.newton(maximum_iterations=30, line_search="backtracking"),
    )
    return step, left, right


def observation(step, left, right):
    reaction = float(results.reaction_resultant(step, on=right, component=1))
    support = np.asarray(results.reaction_resultant(step, on=left))
    total = np.asarray(results.reaction_resultant(step))
    constraint_balance = support + np.array([0., reaction, 0.])
    stress = step.response.cauchy_stress.values
    accumulated = step.state.committed["P"].values
    return dict(time_s=step.time, tip_reaction_N=reaction, support_reaction_N=support.tolist(),
                global_force_residual_N=total.tolist(),
                constraint_force_balance_N=constraint_balance.tolist(),
                relative_force_balance=float(np.linalg.norm(constraint_balance) / max(abs(reaction), 1.e-30)),
                accepted_equilibrium_residual_norm=step.accepted_increments[-1].residual_norm,
                maximum_accumulated_inelastic_strain=float(np.max(accumulated)),
                sampled_axial_stress_range_MPa=[float(np.min(stress[:, 0, 0])), float(np.max(stress[:, 0, 0]))])


def run(*, resolution=1, refinement=1, checkpoint_dir=None):
    started = perf_counter()
    options = dict(resolution=resolution, refinement=refinement)
    step, left, right = build(**options)
    step.solve(until=4 / 64)
    loaded = observation(step, left, right)
    checkpoint = None
    if checkpoint_dir is not None:
        checkpoint = step.save_checkpoint(Path(checkpoint_dir) / "loaded")
    step.solve()
    held = observation(step, left, right)
    restart_exact = None
    if checkpoint is not None:
        restored, rleft, rright = build(**options)
        restored.load_checkpoint(checkpoint)
        restored.solve()
        restart_exact = all(a.name == b.name and np.array_equal(a.abscissa, b.abscissa)
                            and np.array_equal(a.values, b.values)
                            for a, b in zip(step.material_point_histories, restored.material_point_histories, strict=True))
        if not restart_exact or observation(restored, rleft, rright) != held:
            raise AssertionError("Nonuniform restart differs from continuous accepted response")
    return dict(schema="agentfem.research.p92-nonuniform-structure.v1", resolution=resolution,
                tool_sha256=SOURCE_SHA256,
                refinement=refinement, cells=4 * resolution**3, loaded=loaded, held=held,
                geometry_mm=[4, 1, 1], tip_displacement_mm=.02, ramp_s=4, hold_s=60,
                temperature_c=650, poisson_assumption=.3, source_doi="10.3390/met10030307",
                accepted_increments=len(step.accepted_increments), restart_exact=restart_exact,
                elapsed_seconds=perf_counter() - started, engineering_qualified=False,
                scope="small-strain research tensor extension; nonuniform numerical screen, not component life",
                point_comparison="cell 0 records one reference point per mesh; no cross-mesh point-stress comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.checkpoint_dir is not None and args.checkpoint_dir.exists()):
        parser.error("Choose new output/checkpoint paths; existing evidence is not overwritten")
    report = run(resolution=args.resolution, refinement=args.refinement, checkpoint_dir=args.checkpoint_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
