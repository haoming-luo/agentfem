# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent Gmsh Hex blocks, elastic nonmatching bond and moving plane.

Run alongside nonmatching_hex_bond.py, which supplies only example geometry.
This serial finite-J2 demonstration is not the tester's orthotropic material.
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import boundary_models, constitutive, elements, fields, fracture
from agentfem import interfaces, mesh, models, studies
from nonmatching_hex_bond import make_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/finite_hex_bond_contact"))
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The combined finite-bond/contact route is serial only.")
    imported = make_mesh()
    domain = imported.domain
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=0.02, hardening_modulus=5, density=2,
    ))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(0, 1))
    model.fix(u, on=lambda x: np.isclose(x[2], -1), components=2)
    traces = [interfaces.reference_trace_from_boundary(u, mesh.tagged_boundary_region(
        domain, imported.facet_tags, tag=tag, name=f"bond_side_{tag}",
    )) for tag in (1, 2)]
    (negative, dn), (positive, dp) = traces
    bond = fracture.nonmatching_cohesive_force(
        interfaces.pair_reference_traces(negative, positive, tolerance=1e-10), u,
        interfaces.elastic_cohesive(normal_stiffness=100, tangential_stiffness=100),
        negative_dofs=dn, positive_dofs=dp,
    )
    slave = mesh.boundary(domain, lambda x: np.isclose(x[2], 1), name="loaded_end")
    tool = boundary_models.rigid_body(
        boundary_models.rigid_plane(point=(0, 0, 1), normal=(0, 0, -1)),
        motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(translation=(0, 0, -0.01)), end_time=0.1,
        ),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(slave, tool, penalty=200))
    step = model.step(
        target=u, cohesive_force=bond,
        element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=40, hourglass_scale=0.1, kinematics="finite_strain",
        ), omega_squared_bound=1e7, maximum_negative_growth_per_increment=0.1,
        dt=1e-4, steps=1000, history_every=100, save_every=100, progress=False,
    )
    if args.restart:
        step.load_checkpoint(args.restart)
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=args.output / "fields.xdmf",
                               field_variables=("S", "F", "PEEQ", "PDENER"))
    step.save_checkpoint(args.output / "checkpoint")
    result.write_manifest(args.output / "result.json")
    row = step.history_records[-1]
    for name in ("contact_motion_work", "interface_stored_energy", "material_dissipation",
                 "relative_energy_balance_error"):
        print(f"{name}: {row[name]:.9g}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
