from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI
import numpy as np

from agentfem import campaigns, datasets, responses


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: portable_response_driver.py OUTPUT_DIRECTORY")
    comm = MPI.COMM_WORLD
    output = Path(sys.argv[1])
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("load", -2.0, 2.0),
    )
    operator = responses.finite_difference(
        parameter_space=space,
        baseline={"load": 0.5},
        outputs=(datasets.Quantity("response"),),
        perturbation=0.1,
        step_mode="absolute",
    )
    campaign_report, response = operator.run(
        evaluate=lambda values: {"response": 4.0 * values["load"]},
        output_directory=output,
        comm=comm,
    )
    if not response.complete or not np.isclose(response.jacobian[0, 0], 4.0):
        raise RuntimeError("Distributed response experiment returned a wrong Jacobian.")
    if campaign_report.runtime["identity"]["mpi"]["rank_count"] != comm.size:
        raise RuntimeError("Campaign runtime evidence lost the MPI rank count.")
    comm.barrier()
    if comm.rank == 0:
        stored = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if stored["runtime"]["identity"]["mpi"]["rank_count"] != comm.size:
            raise RuntimeError("Persisted Campaign evidence has a wrong MPI rank count.")


if __name__ == "__main__":
    main()
