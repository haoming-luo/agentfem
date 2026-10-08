# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Public-provider 1 -> 2 -> 1 restart acceptance; run phases sequentially.

python tests/portable_coupled_checkpoint_driver.py --directory DIR --phase seed
agentfem mpi-run -n 2 --timeout 120 -- python tests/portable_coupled_checkpoint_driver.py --directory DIR --phase continue
python tests/portable_coupled_checkpoint_driver.py --directory DIR --phase finish

The example is a repository fixture; AgentFEM is imported from the configured
runtime, so this driver also works against an installed wheel.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from mpi4py import MPI
from agentfem import amplitudes, constraints


def create_step():
    path = Path(__file__).resolve().parents[1] / "examples/coupled_thermoelastic_3d.py"
    spec = importlib.util.spec_from_file_location("coupled_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, u, theta, heating = module.create_model()
    for axis, length in enumerate((2.0, 1.0, 3.0)):
        model.constraint(
            constraints.time_dependent_component_dirichlet(
                u,
                axis,
                marker=lambda x, axis=axis, length=length: np.isclose(x[axis], length),
                amplitude=amplitudes.ramp(end_value=1e-4 * length),
                name=f"upper_motion_{axis}",
            )
        )
    return model.step(
        target=u,
        temperature_departure=theta,
        dt=0.1,
        steps=3,
        heat_loads=(heating,),
    )


def run(directory, phase):
    directory.mkdir(parents=True, exist_ok=True)
    with create_step() as step:
        if phase == "seed":
            step.advance()
            step.save_checkpoint(directory / "one")
        else:
            step.load_checkpoint(directory / ("one" if phase == "continue" else "two"))
            assert step.completed_steps == (1 if phase == "continue" else 2)
            step.advance()
            if phase == "continue":
                step.save_checkpoint(directory / "two")

        # Compare every restored station against uninterrupted execution on
        # the current partition, including both force stations and path work.
        with create_step() as reference:
            for _ in range(step.completed_steps):
                reference.advance()
            for actual, expected in (
                (step.u.value, reference.u.value),
                (step.theta.value, reference.theta.value),
                (step.energy.old_reaction, reference.energy.old_reaction),
                (step.energy.old_external, reference.energy.old_external),
            ):
                np.testing.assert_allclose(
                    actual.x.array, expected.x.array, atol=1e-10, rtol=1e-9
                )
            for actual, expected in zip(step.history, reference.history):
                for key, value in expected.items():
                    if isinstance(value, (float, int)):
                        np.testing.assert_allclose(
                            actual[key], value, atol=1e-10, rtol=1e-8
                        )
    if MPI.COMM_WORLD.rank == 0:
        print(f"{phase}: accepted on {MPI.COMM_WORLD.size} rank(s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("seed", "continue", "finish"), required=True
    )
    args = parser.parse_args()
    run(args.directory, args.phase)
