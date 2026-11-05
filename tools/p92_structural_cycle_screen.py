"""Repeated FEM holds versus continuous scalar P92 reference; not life prediction."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from agentfem import amplitudes, fields, mesh, models, results, steps, studies, units
from tools.p92_tensor_material_reference import P92TensorResearchMaterial
from tools.p92_published_recovery_reference import run_cycles


def waveform(*, cycles, dwell, refinement=1):
    if type(cycles) is not int or cycles < 1 or type(refinement) is not int or refinement < 1:
        raise ValueError("Positive integer cycles and refinement are required.")
    if isinstance(dwell, bool) or not np.isfinite(dwell) or dwell <= 0:
        raise ValueError("Positive finite dwell seconds are required.")
    knots, strain, accepted, peaks, ends = [0.], [0.], [], [], []

    def segment(duration, target, maximum_dt):
        start = knots[-1]
        end = start + duration
        count = int(np.ceil(duration / maximum_dt)) * refinement
        accepted.extend(np.linspace(start, end, count + 1)[1:].tolist())
        knots.append(end)
        strain.append(target)

    segment(4., .004, .5)
    for cycle in range(cycles):
        peaks.append(knots[-1])
        segment(dwell, .004, 5.)
        ends.append(knots[-1])
        if cycle != cycles - 1:
            segment(8., -.004, .5)
            segment(8., .004, .5)
    return tuple(map(np.asarray, (knots, strain, accepted, peaks, ends)))


def build(*, temperature, cycles, dwell, refinement):
    knots, strain, accepted, _, _ = waveform(cycles=cycles, dwell=dwell, refinement=refinement)
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1), cell_type="hexahedron")
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True),
                          mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    model.material(P92TensorResearchMaterial(temperature_C=temperature, poisson=.3))
    for axis, component in (("x", 0), ("y", 1), ("z", 2)):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0.), component=component)
    model.fix(u, on=mesh.face(domain, axis="x", value=1.), component=0, value=.004)
    return model.step(
        target=u, duration=float(knots[-1]), time_unit="s", temperature=temperature + 273.15,
        material_points=(results.material_point("specimen", cell=0),), progress=False,
        amplitude=amplitudes.tabular(knots / knots[-1], strain / .004),
        incrementation=steps.at(*(accepted / knots[-1])),
    )


def run(*, temperature=600, cycles=4, dwell=60., refinement=1, checkpoint_dir=None):
    started = perf_counter()
    options = dict(temperature=temperature, cycles=cycles, dwell=dwell, refinement=refinement)
    knots, strain, _, peaks, ends = waveform(cycles=cycles, dwell=dwell, refinement=refinement)
    reference = run_cycles(temperature_C=temperature, strain_range=.008,
                           dwell=dwell, cycles=cycles, rtol=1e-10)
    step = build(**options)
    step.solve()
    histories = {h.name: h for h in step.material_point_histories}
    stress, deformation = histories["specimen.S"], histories["specimen.E"]

    def at(time):
        indices = np.flatnonzero(np.isclose(stress.abscissa, time, rtol=0, atol=1e-9))
        if len(indices) != 1:
            raise ValueError("A waveform endpoint lacks one accepted history sample.")
        return float(stress.values[indices[0], 0, 0])

    comparisons = []
    for peak, end, row in zip(peaks, ends, reference["cycles"], strict=True):
        comparisons.append(dict(cycle=row["cycle"], peak_time_s=float(peak), end_time_s=float(end),
                                peak_error_MPa=at(peak) - row["peak_stress_MPa"],
                                end_hold_error_MPa=at(end) - row["end_hold_stress_MPa"]))
    restart_exact = None
    if checkpoint_dir is not None:
        if cycles < 2:
            raise ValueError("Restart screen requires at least two cycles.")
        partial = build(**options)
        partial.solve(until=float(ends[cycles // 2 - 1] / knots[-1]))
        checkpoint = partial.save_checkpoint(Path(checkpoint_dir) / "accepted")
        restored = build(**options)
        restored.load_checkpoint(checkpoint)
        restored.solve()
        restart_exact = all(
            a.name == b.name and np.array_equal(a.abscissa, b.abscissa)
            and np.array_equal(a.values, b.values)
            for a, b in zip(step.material_point_histories, restored.material_point_histories, strict=True)
        )
        if not restart_exact:
            raise AssertionError("Restarted material histories differ from continuous run.")
    off_axial = stress.values.copy()
    off_axial[:, 0, 0] = 0
    return dict(
        schema="agentfem.research.p92-structural-cycles.v1", temperature_C=temperature,
        strain_range=.008, strain_rate_per_s=.001, poisson_assumption=.3,
        reference={key: reference[key] for key in
                   ("source_doi", "source_pdf_sha256", "parameters", "method", "rtol", "startup")},
        cycles=cycles, dwell_s=dwell, refinement=refinement, comparisons=comparisons,
        maximum_stress_error_MPa=max(abs(r[k]) for r in comparisons for k in ("peak_error_MPa", "end_hold_error_MPa")),
        maximum_lateral_stress_MPa=float(np.max(np.abs(off_axial))),
        maximum_prescribed_strain_error=float(np.max(np.abs(
            deformation.values[:, 0, 0] - np.interp(deformation.abscissa, knots, strain)))),
        accepted_increments=len(step.accepted_increments), restart_histories_exact=restart_exact,
        elapsed_seconds=perf_counter() - started, engineering_qualified=False,
        timing_scope="reference integration, FEM construction/solve and optional complete restart rerun",
        scope="homogeneous structural numerical verification; scalar reference uses same published equations; no life prediction",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=int, choices=(600, 650), default=600)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--dwell", type=float, default=60.)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.checkpoint_dir is not None and args.checkpoint_dir.exists()):
        parser.error("Choose new evidence/checkpoint paths; existing results are not overwritten.")
    report = run(temperature=args.temperature, cycles=args.cycles, dwell=args.dwell,
                 refinement=args.refinement, checkpoint_dir=args.checkpoint_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, allow_nan=False))


if __name__ == "__main__":
    main()
