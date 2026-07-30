"""FEniCSx campaign -> scientific dataset -> surrogate -> guarded prediction.

The example intentionally varies one physical input so a transparent ridge
baseline is enough to demonstrate the full contract. Larger studies can add
loads, geometry, material fields, or time histories without changing the
campaign/dataset interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import campaigns, datasets, fields, mesh, models, studies, surrogates
from agentfem.constitutive import elasticity
from agentfem.diagnostics import print_on_root
from agentfem.solvers import LinearSolverOptions


@dataclass
class StaticCase:
    """Objects required to execute and inspect one finite-element case."""

    model: object
    step: object
    displacement: object


def build_case(parameters) -> StaticCase:
    """Build a fresh immutable-by-construction model variant."""

    domain = mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.2),
        cells=(20, 4),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
        name="campaign_plane_strain",
    )
    model = models.create(study=study, mesh=domain, name="campaign_cantilever")
    displacement = model.field(fields.displacement(domain, degree=1))
    model.material(
        elasticity.isotropic_elastic(
            young=parameters["young"],
            poisson=0.30,
            density=7800.0,
            name="parameterized_steel",
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction(value=(0.0, -1.0e6), on=right)
    step = model.linear_static_step(
        target=displacement,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
    )
    return StaticCase(model=model, step=step, displacement=displacement)


def evaluate_case(case: StaticCase):
    """Solve and extract a declared scalar quantity of interest."""

    case.step.solve()
    return campaigns.CaseOutcome(
        outputs={"max_abs_displacement": case.displacement.max_abs()},
        provenance={
            "backend": "fenicsx",
            "quantity_definition": "global maximum absolute displacement dof",
        },
    )


def high_fidelity_fallback(parameters):
    """Run one new FEM case when the surrogate is asked to extrapolate."""

    return evaluate_case(build_case(parameters)).outputs


def main() -> None:
    sample_count = int(os.environ.get("AGENTFEM_CAMPAIGN_SAMPLES", "10"))
    if sample_count < 4:
        raise ValueError("AGENTFEM_CAMPAIGN_SAMPLES must be at least 4.")
    run_id = os.environ.get("AGENTFEM_CAMPAIGN_RUN_ID", "").strip()
    output_name = "static_elasticity_surrogate_campaign"
    if run_id:
        output_name = f"{output_name}_{run_id}"
    output = (
        Path(__file__).resolve().parents[1]
        / "examples_output"
        / output_name
    )
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter(
            "young",
            150.0e9,
            250.0e9,
            unit="Pa",
            description="isotropic Young's modulus",
            nominal=210.0e9,
        ),
        name="cantilever_material_space",
    )
    campaign = campaigns.create(
        name="cantilever_young_sweep",
        parameter_space=space,
        outputs=(
            datasets.Quantity(
                "max_abs_displacement",
                unit="m",
                description="global maximum absolute displacement dof",
            ),
        ),
        build=build_case,
        evaluate=evaluate_case,
        metadata={
            "purpose": "AgentFEM campaign/dataset/surrogate demonstration",
            "fidelity": "FEniCSx finite-element solve",
        },
    )
    sampling = campaigns.latin_hypercube(space, sample_count, seed=2026)
    report = campaign.run(
        sampling,
        output_directory=output / "campaign",
        comm=MPI.COMM_WORLD,
    )
    if report.dataset is None:
        raise RuntimeError("The campaign produced no successful training samples.")

    split = report.dataset.split(validation_fraction=0.2, seed=2026)
    trained = surrogates.RidgeSurrogate(alpha=1.0e-10).fit(split.train)
    validation = trained.validate(split.validation)
    if MPI.COMM_WORLD.rank == 0:
        trained.write(output / "surrogate")
    MPI.COMM_WORLD.barrier()

    domain = surrogates.BoxApplicabilityDomain.from_dataset(split.train)
    guarded = surrogates.GuardedSurrogate(
        trained,
        domain,
        fallback=high_fidelity_fallback,
    )
    prediction = guarded.predict({"young": 210.0e9})

    print_on_root(MPI.COMM_WORLD, validation.format())
    print_on_root(
        MPI.COMM_WORLD,
        f"Prediction source={prediction.source}, outputs={prediction.outputs}",
    )
    print_on_root(MPI.COMM_WORLD, f"Campaign artifacts: {output}")


if __name__ == "__main__":
    main()
