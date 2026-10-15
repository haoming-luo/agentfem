# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Small real-FEM campaign: wall temperature -> verified linear heat solution."""
import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from agentfem import campaigns, constitutive, datasets, fields, mesh, models, project, results, studies
from agentfem.time import ProgressPrinter


def evaluate(parameters):
    domain = mesh.rectangle((0., 0.), (1., .2), (8, 2), comm=MPI.COMM_WORLD)
    model = models.create(study=studies.steady_heat_transfer(dimension=2), mesh=domain)
    temperature = model.field(fields.temperature(domain, value=300.))
    model.material(constitutive.thermoelastic(
        young=200e9, poisson=.3, density=7800., thermal_expansion=12e-6,
        conductivity=45., specific_heat=500., reference_temperature=300.))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="hot", tag=1)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="cold", tag=2)
    hot = parameters["hot_temperature"]
    model.prescribed_temperature(temperature, hot, on=left)
    model.prescribed_temperature(temperature, 300., on=right)
    simulation = model.step(target=temperature).solve_result()
    maximum = temperature.max_abs()
    # This simple maximum principle check is not general thermal validation.
    if not np.isclose(maximum, hot, rtol=1e-10, atol=1e-8):
        raise RuntimeError("Maximum-temperature reference check failed")
    simulation.add_quantity("temperature_maximum", maximum, unit="K")
    simulation.verify("exploratory", required_quantities=("temperature_maximum",)).require()
    return simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    if not 2 <= args.samples <= 1000:
        parser.error("samples must be between 2 and 1000")
    root = Path(__file__).resolve().parent
    run = project.current_run(project_root=root)
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("hot_temperature", 350., 450., unit="K"))
    campaign = campaigns.create(
        name="heated_wall", parameter_space=space,
        outputs=(datasets.Quantity("temperature_maximum", unit="K"),), evaluate=evaluate)
    plan = campaigns.latin_hypercube(space, args.samples, seed=2026)
    # Stable project-local ledger enables resume across ordinary CLI run IDs.
    report = campaign.run(plan, output_directory=root / "campaign-data", comm=MPI.COMM_WORLD)
    dataset = report.require_dataset(minimum_samples=args.samples, quality="exploratory")
    if MPI.COMM_WORLD.rank == 0:
        manifest = dataset.write(run.artifact("dataset"))
        audit = report.write_audit(run.artifact("campaign-audit.json"))
        result = results.SimulationResult("heat_campaign")
        result.add_quantity("sample_count", len(dataset.samples), unit="1")
        result.add_artifact("dataset_manifest", manifest)
        result.add_artifact("dataset_arrays", manifest.parent / "arrays.npz")
        result.add_artifact("campaign_audit", audit)
        run.publish(result)
        ProgressPrinter(args.samples, 0, MPI.COMM_WORLD).status(
            "Campaign completed", quantities={"Samples": (len(dataset.samples), "")})
    MPI.COMM_WORLD.barrier()


if __name__ == "__main__":
    main()
