# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial elastic bond between independent coarse/fine Gmsh Hex8 meshes.

The example is a small-strain compliance check, not a forming simulation.
Requires the optional Gmsh dependency and an installed AgentFEM candidate.
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, fracture, interfaces, mesh, models, studies


def make_mesh():
    gmsh = mesh.require_gmsh()
    if gmsh.isInitialized():
        raise RuntimeError("Run this standalone example with a fresh Gmsh session.")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("independent_hex_blocks")
        volumes = [gmsh.model.occ.addBox(0, 0, z, 1, 1, 1) for z in (-1, 0)]
        gmsh.model.occ.synchronize()
        # Do not fragment or merge: the two interface traces need independent DOFs.
        for tag, (volume, cells) in enumerate(zip(volumes, (2, 3)), start=1):
            faces = gmsh.model.getBoundary([(3, volume)], oriented=False)
            curves = gmsh.model.getBoundary(faces, combined=False, oriented=False)
            for _, curve in set(curves):
                gmsh.model.mesh.setTransfiniteCurve(curve, cells + 1)
            for _, face in faces:
                gmsh.model.mesh.setTransfiniteSurface(face)
                gmsh.model.mesh.setRecombine(2, face)
            gmsh.model.mesh.setTransfiniteVolume(volume)
            gmsh.model.addPhysicalGroup(3, [volume], tag)
            interface = [
                face
                for _, face in faces
                if abs(gmsh.model.occ.getCenterOfMass(2, face)[2]) < 1e-12
            ]
            gmsh.model.addPhysicalGroup(2, interface, tag)
        gmsh.model.mesh.generate(3)
        return mesh.import_gmsh_model(gmsh.model, MPI.COMM_SELF)
    finally:
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/nonmatching_hex_bond")
    )
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This experimental interface workflow is serial only.")
    imported = make_mesh()
    domain = imported.domain
    model = models.create(study=studies.static_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(0, 1))
    model.fix(u, on=lambda x: np.isclose(x[2], -1), components=2)
    model.fix(u, on=lambda x: np.isclose(x[2], 1), components=2, value=0.02)
    traces = []
    for tag in (1, 2):
        boundary = mesh.tagged_boundary_region(
            domain, imported.facet_tags, tag=tag, name=f"bond_side_{tag}"
        )
        traces.append(interfaces.reference_trace_from_boundary(u, boundary))
    (negative, dn), (positive, dp) = traces
    pairing = interfaces.pair_reference_traces(negative, positive, tolerance=1e-10)
    law = interfaces.elastic_cohesive(normal_stiffness=1000, tangential_stiffness=500)
    force = fracture.nonmatching_cohesive_force(
        pairing, u, law, negative_dofs=dn, positive_dofs=dp
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = model.step(target=u, cohesive_force=force).solve_result(
        output=args.output / "fields.xdmf"
    )
    result.write_manifest(args.output / "result.json")
    exact_force = 0.02 / (2 / 100 + 1 / 1000)
    exact_energy = exact_force**2 / 2000
    computed = result.quantities["interface_stored_energy"].value
    relative = abs(computed / exact_energy - 1)
    if relative > 1e-8:
        raise RuntimeError(f"Series-compliance reference failed: {relative:g}")
    print(f"Elastic Q1 bond completed; interface-energy relative error {relative:.3g}.")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
