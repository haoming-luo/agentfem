"""Sequential heat-transfer and thermal-stress analysis of a hot wall."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, io, mesh, models, studies


def main() -> None:
    domain = mesh.rectangle(
        (0.0, 0.0),
        (0.12, 1.0),
        (24, 60),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    steel = constitutive.thermoelastic(
        name="hot-section steel",
        young=180.0e9,
        poisson=0.3,
        density=7800.0,
        thermal_expansion=13.0e-6,
        conductivity=32.0,
        specific_heat=560.0,
        reference_temperature=573.15,
    )
    hot = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="hot_face", tag=1
    )
    cold = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.12), name="cold_face", tag=2
    )

    heat = models.create(
        study=studies.first_order_transient(
            physics="heat_transfer",
            dimension=2,
        ),
        mesh=domain,
        name="wall_heat_transfer",
    )
    temperature = heat.field(fields.temperature(domain, value=573.15))
    heat.material(steel)
    heat.fix(temperature, on=hot, value=823.15)
    heat.fix(temperature, on=cold, value=573.15)
    thermal_step = heat.step(
        target=temperature,
        dt=60.0,
        steps=60,
        save_every=5,
    )
    output_dir = Path(__file__).resolve().parents[1] / "examples_output"
    thermal_step.run(output=output_dir / "thermal_stress_wall_temperature.xdmf")

    mechanics = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="wall_thermal_stress",
    )
    displacement = mechanics.field(fields.displacement(domain))
    mechanics.material(steel)
    bottom = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="bottom", tag=3
    )
    mechanics.fix(displacement, on=bottom, component=1, value=0.0)
    mechanics.fix(displacement, on=cold, component=0, value=0.0)
    mechanics.step(
        target=displacement,
        K=mechanics.stiffness(displacement),
        F=mechanics.thermal_expansion(displacement, temperature),
    ).solve()
    with io.XDMFTimeSeries(
        output_dir / "thermal_stress_wall_displacement.xdmf",
        domain,
    ) as writer:
        writer.write_fields(1.0, displacement.value, temperature.value)


if __name__ == "__main__":
    main()
