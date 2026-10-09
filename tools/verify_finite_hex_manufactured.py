# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial spatial verification of the private finite Hex8 contribution.

An independent compressible Neo-Hookean provider and UFL manufactured body
force exercise nonuniform deformation. Newton/CG here is a verification oracle,
not a registered product Procedure. All displacement components vanish on the
boundary. Report physical and artificial energy separately.
"""

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from scipy.sparse.linalg import LinearOperator, cg

from agentfem import constitutive as c
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual


class ReferenceMaterial:
    """Independent test provider, not an addition to the material library."""

    name = "independent Neo-Hookean verification provider"
    state_schema = c.MaterialStateSchema(
        "verification.neo_hookean", (c.MaterialStateVariable("unused"),)
    )
    tangent_convention = c.MaterialTangentConvention.first_piola_deformation_gradient()
    mu, lam = 30.0, 40.0

    def update(self, point):
        return self.update_batch(c.MaterialPointBatchInput((point,))).responses[0]

    def update_batch(self, request):
        f = np.stack([p.deformation_gradient_new for p in request.points])
        inverse = np.linalg.inv(f).swapaxes(1, 2)
        jac = np.linalg.det(f)
        logj = np.log(jac)
        p = self.mu * f + (self.lam * logj - self.mu)[:, None, None] * inverse
        w = (
            0.5 * self.mu * (np.sum(f * f, axis=(1, 2)) - 3)
            - self.mu * logj
            + 0.5 * self.lam * logj**2
        )
        a = self.mu * np.einsum("ik,JL->iJkL", np.eye(3), np.eye(3))[None]
        a = a + self.lam * np.einsum("ciJ,ckL->ciJkL", inverse, inverse)
        a -= (self.lam * logj - self.mu)[:, None, None, None, None] * np.einsum(
            "ciL,ckJ->ciJkL", inverse, inverse
        )
        stress = (p @ f.swapaxes(1, 2)) / jac[:, None, None]
        return c.MaterialPointBatchOutput(
            tuple(
                c.MaterialPointOutput(
                    cauchy_stress=stress[i],
                    consistent_tangent=a[i].reshape(9, 9),
                    state_new=point.state_old,
                    strain_energy_density=w[i],
                    tangent_convention=self.tangent_convention,
                    state_schema=self.state_schema,
                    dissipation_density_increment=0.0,
                )
                for i, point in enumerate(request.points)
            )
        )


def run(size, amplitude=0.08, distortion=0.0):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Private finite Hex8 verification is serial only.")
    start = perf_counter()
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, size, size, size, cell_type=mesh.CellType.hexahedron
    )
    if not 0 <= distortion <= 0.2:
        raise ValueError("This verification admits distortion fractions 0..0.2 only.")
    if distortion:
        points = domain.geometry.x
        interior_nodes = np.all((points > 1e-10) & (points < 1 - 1e-10), axis=1)
        points[interior_nodes] += np.random.default_rng(1729).uniform(
            -distortion / size, distortion / size, (np.count_nonzero(interior_nodes), 3)
        )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u = fem.Function(space)
    law = ReferenceMaterial()
    response = MaterialQuadratureResponse.create(domain, law.state_schema, degree=1)
    internal = FiniteUniformHexResidual(
        u, response, density=2, hourglass_modulus=law.mu, hourglass_scale=0.1
    )
    x = ufl.SpatialCoordinate(domain)
    bubble = (
        amplitude
        * ufl.sin(np.pi * x[0])
        * ufl.sin(np.pi * x[1])
        * ufl.sin(np.pi * x[2])
    )
    exact = ufl.as_vector((bubble, 0.4 * bubble, -0.2 * bubble))
    f = ufl.Identity(3) + ufl.grad(exact)
    p = law.mu * f + (law.lam * ufl.ln(ufl.det(f)) - law.mu) * ufl.inv(f).T
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})
    load = petsc.assemble_vector(
        fem.form(ufl.inner(-ufl.div(p), ufl.TestFunction(space)) * dx)
    )
    coordinates = space.tabulate_dof_coordinates()
    interior = np.all((coordinates > 1e-10) & (coordinates < 1 - 1e-10), axis=1)
    free = (3 * np.flatnonzero(interior)[:, None] + np.arange(3)).ravel()
    old = np.tile(np.eye(3), (size**3, 1, 1))
    linear_iterations = 0
    try:
        for iteration in range(15):
            vector, trial = internal.evaluate(
                law, deformation_gradient_old=old, time=0, time_increment=1
            )
            rhs = vector.array[free].copy() - load.array[free]
            vector.destroy()
            relative = np.linalg.norm(rhs) / max(
                np.linalg.norm(load.array[free]), 1e-30
            )
            if relative < 1e-9:
                break

            def action(value):
                direction = np.zeros_like(u.x.array)
                direction[free] = value
                result = internal.tangent_action(direction)
                try:
                    return result.array[free].copy()
                finally:
                    result.destroy()

            def count(_):
                nonlocal linear_iterations
                linear_iterations += 1

            increment, info = cg(
                LinearOperator((len(free), len(free)), matvec=action, dtype=float),
                -rhs,
                rtol=1e-10,
                atol=1e-12,
                maxiter=1000,
                callback=count,
            )
            if info:
                raise RuntimeError(f"Verification CG did not converge: {info}")
            u.x.array[free] += increment
        else:
            raise RuntimeError("Verification Newton did not converge")
        error = float(
            np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u - exact, u - exact) * dx)))
        )
        norm = float(
            np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(exact, exact) * dx)))
        )
        physical = float(trial.element_response.physical_energy.sum())
        artificial = float(trial.element_response.hourglass_energy.sum())
        return dict(
            size=size,
            cells=size**3,
            distortion_fraction=distortion,
            displacement_relative_l2=error / norm,
            equilibrium_relative=relative,
            physical_energy=physical,
            hourglass_energy=artificial,
            hourglass_to_physical=artificial / physical,
            newton_iterations=iteration,
            cg_iterations=linear_iterations,
            wall_seconds=perf_counter() - start,
        )
    finally:
        load.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--distortion", type=float, default=0.0)
    args = parser.parse_args()
    if any(n < 2 or n > 24 for n in args.sizes):
        parser.error("sizes must be between 2 and 24")
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    records = []
    for size in args.sizes:
        records.append(run(size, distortion=args.distortion))
        print(json.dumps(records[-1]), flush=True)
    if args.output is not None:
        record = {
            "schema": "agentfem.private-finite-hex-spatial-verification.v1",
            "revision": revision,
            "tracked_source_dirty": dirty,
            "scope": "serial_regular_hex_neo_hookean_manufactured_solution",
            "numpy_version": np.__version__,
            "amplitude": 0.08,
            "mu": 30,
            "lambda": 40,
            "hourglass_scale": 0.1,
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
