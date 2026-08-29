"""Run the AgentFEM candidate for the Lewandowski et al. beam gate.

This is an opt-in evidence driver, not a unit test.  It writes the candidate
curve and an assessment that remains incomplete unless an independently
executed, pinned MGIS/FEniCS curve and all promotion evidence are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
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

from lewandowski_2023_self_weight_beam_fixture import (
    DEFINITION,
    UPSTREAM_BEHAVIOUR_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_SOLVER_SHA256,
    assess_external_curve,
)


def _candidate_step(comm, *, subdivisions, increments):
    definition = DEFINITION
    domain = mesh.cuboid(
        (0.0, -0.5 * definition.width, -0.5 * definition.height),
        (definition.length, 0.5 * definition.width, 0.5 * definition.height),
        subdivisions,
        comm=comm,
        cell_type="tetrahedron",
    )
    study = studies.nonlinear_static(physics="solid_mechanics", dimension=3)
    model = models.create(
        study=study,
        mesh=domain,
        name="lewandowski_2023_self_weight_beam_candidate",
    )
    displacement = model.field(
        fields.displacement(domain, degree=definition.displacement_degree)
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left_clamp",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], definition.length),
        name="right_symmetry",
        tag=2,
    )
    model.fix(displacement, on=left, value=(0.0, 0.0, 0.0))
    model.fix(displacement, on=right, component=0, value=0.0)
    material = constitutive.finite_strain_j2_logarithmic(
        young=definition.young,
        poisson=definition.poisson,
        yield_stress=definition.yield_stress,
        hardening_modulus=definition.hardening_modulus,
    )
    model.material(material)
    model.body_force(
        (0.0, 0.0, -definition.maximum_body_force),
        target=displacement,
        name="self_weight_body_force",
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-8,
            maximum_iterations=30,
            line_search="backtracking",
        ),
        progress=False,
        name="lewandowski_2023_self_weight_beam_candidate",
    )
    return step, displacement


def _read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, names=True, delimiter=",")
    names = tuple(table.dtype.names or ())
    required = ("load_factor", "downward_displacement_m")
    if not set(required).issubset(names):
        raise ValueError(f"Reference CSV must contain columns {required}.")
    return (
        np.atleast_1d(table["load_factor"]).astype(float),
        np.atleast_1d(table["downward_displacement_m"]).astype(float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--subdivisions", type=int, nargs=3, default=DEFINITION.subdivisions)
    parser.add_argument("--increments", type=int, default=DEFINITION.increments)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--promotion-evidence-json", type=Path)
    arguments = parser.parse_args()
    if any(value <= 0 for value in arguments.subdivisions):
        raise ValueError("All subdivisions must be positive.")
    if arguments.increments < 5:
        raise ValueError("At least five load increments are required.")

    comm = MPI.COMM_WORLD
    step, displacement = _candidate_step(
        comm,
        subdivisions=tuple(arguments.subdivisions),
        increments=arguments.increments,
    )
    factors = np.linspace(0.0, 1.0, arguments.increments + 1)
    downward = [0.0]
    for factor in factors[1:]:
        step.solve(until=float(factor))
        value = results.probe(displacement, at=DEFINITION.observer)
        downward.append(-float(value[2]))

    reference_load = None
    reference_displacement = None
    evidence = {}
    source = {}
    declared_reference_curve_sha256 = None
    actual_reference_curve_sha256 = None
    if arguments.reference_csv is not None:
        reference_load, reference_displacement = _read_reference(arguments.reference_csv)
        actual_reference_curve_sha256 = hashlib.sha256(
            arguments.reference_csv.read_bytes()
        ).hexdigest()
    if arguments.promotion_evidence_json is not None:
        promotion = json.loads(arguments.promotion_evidence_json.read_text())
        evidence = dict(promotion.get("evidence", {}))
        source = dict(promotion.get("source", {}))
        declared_reference_curve_sha256 = promotion.get(
            "reference_curve_sha256"
        )
    assessment = assess_external_curve(
        candidate_load_factors=factors,
        candidate_displacements=downward,
        reference_load_factors=reference_load,
        reference_displacements=reference_displacement,
        reference_source_commit=source.get("commit"),
        reference_solver_sha256=source.get("solver_sha256"),
        reference_behaviour_sha256=source.get("behaviour_sha256"),
        reference_curve_sha256=actual_reference_curve_sha256,
        declared_reference_curve_sha256=declared_reference_curve_sha256,
        convergence_evidence=evidence,
    )
    manifest = {
        "schema": "agentfem.external-benchmark-candidate.v1",
        "benchmark": "lewandowski_2023_self_weight_beam",
        "status": assessment["status"],
        "scientific_definition": DEFINITION.summary(),
        "candidate": {
            "formulation": (
                "multiplicative_Fp_quadratic_Hencky_Kirchhoff_J2_"
                "linear_isotropic_hardening"
            ),
            "subdivisions": tuple(arguments.subdivisions),
            "increments": arguments.increments,
            "mpi_ranks": comm.size,
            "step": step.summary(),
        },
        "reference": {
            "formulation": (
                "MFront_total_Hencky_Hooke_Mises_linear_isotropic_hardening"
            ),
            "commit": UPSTREAM_COMMIT,
            "solver_sha256": UPSTREAM_SOLVER_SHA256,
            "behaviour_sha256": UPSTREAM_BEHAVIOUR_SHA256,
            "curve_supplied": reference_load is not None,
            "curve_sha256": actual_reference_curve_sha256,
            "declared_curve_sha256": declared_reference_curve_sha256,
        },
        "assessment": assessment,
    }
    if comm.rank == 0:
        arguments.output.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            arguments.output / "candidate_curve.csv",
            np.column_stack((factors, np.asarray(downward))),
            delimiter=",",
            header="load_factor,downward_displacement_m",
            comments="",
        )
        (arguments.output / "assessment.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )


if __name__ == "__main__":
    main()
