# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded two-way thermoelastic heating through the ordinary Model workflow.

SI units; quasi-static small strain, constant isotropic material, one shared
mesh. TemperatureDeparture is theta=T-T0. Output is a final snapshot, not a
time series. Run with an installed AgentFEM; no source-path insertion needed.
"""

import argparse
from pathlib import Path
import numpy as np
from mpi4py import MPI
from agentfem import constitutive, fields, loads, mesh, models, studies, units


def run(output):
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 3.0),
        (3, 3, 3),
        comm=MPI.COMM_WORLD,
        cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.coupled_thermoelastic(),
        mesh=domain,
        units=units.si(),
        name="coupled_heating",
    )
    u = model.field(fields.displacement(domain))
    theta = model.field(fields.temperature(domain, value=0.0))
    model.material(
        constitutive.thermoelastic(
            young=1000.0,
            poisson=0.25,
            density=1.0,
            specific_heat=100.0,
            conductivity=1.0,
            thermal_expansion=0.002,
            reference_temperature=300.0,
        )
    )
    for axis in range(3):
        face = mesh.boundary(
            domain,
            lambda x, axis=axis: np.isclose(x[axis], 0.0),
            name=f"symmetry_{axis}",
            tag=axis + 1,
        )
        model.fix(u, on=face, component=axis, value=0.0)
    heating = model.load(loads.heat_source(10.0, target=theta))
    with model.step(
        target=u,
        temperature_departure=theta,
        dt=0.1,
        steps=3,
        heat_loads=(heating,),
        output=output,
    ) as step:
        result = step.solve_result(strict_output=True)
        # A benchmark check is distinct from solver completion.
        expected_theta = 3.0 / (100.0 + 9.0 * (1000.0 / 1.5) * 0.002**2 * 300.0)
        local_error = float(
            np.max(np.abs(theta.value.x.array - expected_theta), initial=0.0)
        )
        error = domain.comm.allreduce(local_error, op=MPI.MAX)
        result.add_quantity("uniform_temperature_absolute_error", error, unit="K")
        if error > 2e-9:
            raise RuntimeError("Uniform thermoelastic analytical check failed.")
        if domain.comm.rank == 0:
            result.write_manifest(Path(output).with_suffix(".result.json"))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("coupled_heating.xdmf"))
    run(parser.parse_args().output)
