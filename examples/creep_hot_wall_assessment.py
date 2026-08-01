"""Sequential hot-wall FEM and Kachanov--Rabotnov creep assessment.

This release demo is intentionally honest about its boundary: heat transfer
and thermoelastic stress are finite-element analyses; creep is then evaluated
at the governing sampled equivalent stress with a verified material-point
law. It demonstrates the data path needed by power-component screening while
the global quadrature creep step remains a release gate, not a hidden claim.
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import ufl
from mpi4py import MPI

from agentfem import constitutive, fields, io, mesh, models, results, studies


def main() -> None:
    smoke = os.environ.get("AGENTFEM_RELEASE_SMOKE") == "1"
    domain = mesh.rectangle(
        (0.0, 0.0),
        (0.12, 1.0),
        (6, 10) if smoke else (16, 32),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    steel = constitutive.thermoelastic(
        name="illustrative high-temperature steel",
        young=180.0e9,
        poisson=0.3,
        density=7800.0,
        thermal_expansion=13.0e-6,
        conductivity=32.0,
        specific_heat=560.0,
        reference_temperature=573.15,
    )
    hot = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="hot_face", tag=1)
    cold = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.12), name="cold_face", tag=2)
    bottom = mesh.boundary(domain, lambda x: np.isclose(x[1], 0.0), name="bottom", tag=3)
    output = Path(__file__).resolve().parents[1] / "examples_output" / "creep_hot_wall"

    heat = models.create(
        study=studies.first_order_transient(physics="heat_transfer", dimension=2),
        mesh=domain,
        name="hot_wall_heat_transfer",
    )
    temperature = heat.field(fields.temperature(domain, value=573.15))
    heat.material(steel)
    heat.fix(temperature, on=hot, value=823.15)
    heat.fix(temperature, on=cold, value=573.15)
    heat.step(
        target=temperature,
        dt=120.0,
        steps=4 if smoke else 30,
        save_every=1 if smoke else 5,
    ).run(
        output=output / "temperature.xdmf"
    )

    mechanics = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="hot_wall_thermal_stress",
    )
    displacement = mechanics.field(fields.displacement(domain))
    mechanics.material(steel)
    mechanics.fix(displacement, on=bottom, component=1, value=0.0)
    mechanics.fix(displacement, on=cold, component=0, value=0.0)
    mechanics.step(
        target=displacement,
        K=mechanics.stiffness(displacement),
        F=mechanics.thermal_expansion(displacement, temperature),
    ).solve()

    stress = constitutive.thermoelastic_stress(
        displacement,
        temperature,
        steel,
        study=mechanics.study,
    )
    deviator = stress - ufl.tr(stress) / 2.0 * ufl.Identity(2)
    mises = ufl.sqrt(1.5 * ufl.inner(deviator, deviator))
    _, governing_stress = results.quadrature_extrema(mises, domain, degree=4)
    with io.XDMFTimeSeries(output / "thermoelastic_state.xdmf", domain) as writer:
        writer.write_fields(1.0, displacement.value, temperature.value)

    creep = constitutive.KachanovRabotnovCreep(
        creep_coefficient=2.0e-8,
        creep_exponent=4.0,
        damage_coefficient=2.0e-8,
        damage_exponent=5.0,
        damage_power=3.0,
        reference_stress=100.0e6,
        reference_time=3600.0,
        failure_damage=0.95,
    )
    rupture_seconds = creep.rupture_time(governing_stress)
    times = np.linspace(0.0, 0.9 * rupture_seconds, 41)
    state = constitutive.CreepDamageState()
    strains = [0.0]
    damage = [0.0]
    for start, end in zip(times, times[1:]):
        state = creep.update(governing_stress, end - start, state).state
        strains.append(state.equivalent_creep_strain)
        damage.append(state.damage)

    theta = constitutive.ModifiedThetaProjection.fit(
        times / 3600.0,
        strains,
    )
    simulation = results.SimulationResult(
        "creep_hot_wall_assessment",
        metadata={
            "workflow": "sequential_thermoelastic_then_material_point_creep",
            "creep_scope": "governing sampled stress; not a global creep field solve",
            "calibration": "illustrative parameters; replace with traceable material data",
            "material": creep.as_dict(),
            "theta_projection": theta.as_dict(),
        },
    )
    simulation.add_quantities(
        {
            "governing_von_mises_stress": governing_stress,
            "predicted_rupture_time": rupture_seconds / 3600.0,
            "assessment_end_damage": damage[-1],
        },
        units={
            "governing_von_mises_stress": "Pa",
            "predicted_rupture_time": "h",
        },
    )
    simulation.add_histories(
        times / 3600.0,
        {
            "equivalent_creep_strain": strains,
            "creep_damage": damage,
            "modified_theta_strain": theta.strain(times / 3600.0),
        },
        abscissa_name="time",
        abscissa_unit="h",
    )
    simulation.add_artifact("temperature", output / "temperature.xdmf")
    simulation.add_artifact("thermoelastic_state", output / "thermoelastic_state.xdmf")
    if domain.comm.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output / "creep_history.csv",
            np.column_stack((times / 3600.0, strains, damage, theta.strain(times / 3600.0))),
            delimiter=",",
            header="time_h,equivalent_creep_strain,damage,modified_theta_strain",
            comments="",
        )
        simulation.add_artifact("creep_history", output / "creep_history.csv")
        simulation.write_manifest(output / "result.json", include_histories=True)
        print(simulation)


if __name__ == "__main__":
    main()
