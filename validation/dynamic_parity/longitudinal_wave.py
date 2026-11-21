# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Constrained 3D longitudinal standing wave: public explicit workflow.

This is an analytical convergence experiment, not the half-sine Gate C or
an Abaqus comparison. Lateral displacement is zero throughout the prism.
The exact longitudinal wave speed is sqrt((lambda+2*mu)/rho).
"""

import json
import argparse
from agentfem import elements
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dm
from agentfem import (
    models,
    studies,
    fields,
    constitutive,
    constraints,
    operators,
    state,
    results,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--element", choices=["tet10_hrz", "tet15_nodal"], default="tet10_hrz"
)
selected_element = parser.parse_args().element
E, nu, rho = 210e9, 0.3, 7800.0
cp = np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))
omega = np.pi * cp
period = 2 * np.pi / omega
out = (
    Path(__file__).parent
    / "evidence"
    / ("uniform_refinement" if selected_element == "tet10_hrz" else "tet15_refinement")
)
out.mkdir(parents=True, exist_ok=True)
records = []
for nx in (8, 16, 32):
    for fraction in (0.8, 0.4):
        domain = dm.create_box(
            MPI.COMM_SELF,
            [[0, 0, 0], [1, 0.1, 0.1]],
            [nx, max(1, nx // 8), max(1, nx // 8)],
        )
        model = models.create(
            study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
        )
        u = model.field(
            fields.displacement(domain, degree=2)
            if selected_element == "tet10_hrz"
            else fields.displacement(domain, element=elements.tet15_mass_lumped())
        )
        model.material(constitutive.isotropic_elastic(young=E, poisson=nu, density=rho))
        V = u.function_space
        for component in (1, 2):
            model.constraint(
                constraints.DirichletConstraint.component(
                    V, component, lambda x: np.ones(x.shape[1], dtype=bool)
                )
            )
        mass = model.lumped_mass(u, method=selected_element)
        st = state.second_order_state(u)
        st.u.interpolate(
            lambda x: np.vstack(
                (np.cos(np.pi * x[0]) * 1e-6, np.zeros((2, x.shape[1])))
            )
        )
        step = model.step(
            target=u, state=st, mass=mass, dt="auto", steps=1, progress=False
        )
        step.dt = step.stability.selected * fraction
        step.steps = int(np.ceil(period / step.dt))
        step.dt = period / step.steps
        initial = operators.assemble_vector(step.residual)
        step.integrator.solve_acceleration(initial)
        initial.destroy()
        st.a.x.array[:] = st.a_next.x.array
        from agentfem.time.explicit import project_homogeneous_kinematics

        project_homogeneous_kinematics(
            st.a, prescribed=step.prescribed, constraints=step.constraints
        )
        h = results.dynamic_history(
            "tip", at=(1, 0.05, 0.05), variables=("U",), length_unit="m", time_unit="s"
        )
        step.solve_result(history=h)
        t = np.array([r["time"] for r in step.history_records])
        y = np.array([r["tip.U1"] for r in step.history_records])
        exact = -1e-6 * np.cos(omega * t)
        err = np.sqrt(np.trapezoid((y - exact) ** 2, t) / np.trapezoid(exact**2, t))
        energies = np.array(
            [r["total_mechanical_energy"] for r in step.history_records]
        )
        row = {
            "element": selected_element,
            "nx": nx,
            "dt_fraction": fraction,
            "dt": step.dt,
            "steps": step.steps,
            "wave_speed": float(cp),
            "relative_trace_L2": float(err),
            "energy_relative_range": float(np.ptp(energies) / energies[0]),
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        np.savetxt(
            out / f"wave_{nx}_{fraction}.csv",
            np.column_stack((t, y, exact, energies)),
            delimiter=",",
            header="time,U1,analytic_U1,energy",
            comments="",
        )
        (out / "longitudinal_wave.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
