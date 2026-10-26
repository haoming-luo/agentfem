# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent discrete virtual-work oracle for the elastic cube diagnostic.

Read-only NumPy assembly from the mesh and displacement coefficients. No
AgentFEM material update, contact trace, residual, or MPC transformation is used
to construct the reference. Basix and the DOLFINx mesh/DOF numbering are shared.
This checks discrete equilibrium, not continuum accuracy or plastic history.
The physical fixture is intentionally fixed: unit cube, square x=1 patch,
fixed plane x=1.005, reference-area penalty, E=1000, nu=.3.
"""

import basix
import numpy as np


def hencky_energy_stress(F, *, young=1000., poisson=.3):
    """Right-metric logarithm, independent of the production left-SVD return."""
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (3, 3) or not np.isfinite(F).all() or np.any(np.linalg.det(F) <= 0.):
        raise ValueError("Expected finite orientation-preserving 3D deformation gradients.")
    mu = young / (2. * (1. + poisson))
    lam = young * poisson / ((1. + poisson) * (1. - 2. * poisson))
    values, vectors = np.linalg.eigh(np.swapaxes(F, -1, -2) @ F)
    if np.any(values <= 0.):
        raise ValueError("Right metric must be positive definite.")
    E = (vectors * (.5 * np.log(values))[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    tr = np.trace(E, axis1=-2, axis2=-1)
    energy = mu * np.sum(E * E, axis=(-2, -1)) + .5 * lam * tr**2
    stress = np.swapaxes(np.linalg.inv(F), -1, -2) @ (
        2. * mu * E + lam * tr[..., None, None] * np.eye(3))
    return energy, stress


def periodic_groups(coordinates):
    """Direct fluctuation equivalence classes, without runtime equation lowering.

    Corner fluctuations vanish; periodic faces share their wrapped class. The
    physical z=0/1 x-motion guide removes that component of each such class.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1.+1e-12):
        raise ValueError("Expected finite unit-cube DOF coordinates.")
    keys = np.rint(x * 1e12).astype(np.int64)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Duplicate DOF coordinates are not admitted.")
    wrapped = keys % 10**12
    _, classes = np.unique(wrapped, axis=0, return_inverse=True)
    groups = 3 * classes[:, None] + np.arange(3)
    groups[np.all(wrapped == 0, axis=1)] = -1
    groups[wrapped[:, 2] == 0, 0] = -1
    free = groups >= 0
    groups[free] = np.unique(groups[free], return_inverse=True)[1]
    return groups


class ElasticCubeOracle:
    """Independent fixed-fixture integration; not an application solver."""

    def __init__(self, function, *, bulk_degree=6, contact_degree=11, penalty=40000.):
        V = function.function_space
        domain = V.mesh
        element = V.element.basix_element
        if (domain.comm.size != 1 or domain.topology.cell_type.name != "tetrahedron"
                or V.dofmap.bs != 3 or element.degree not in (1, 2)
                or not element.dof_transformations_are_identity or element.value_shape):
            raise ValueError("Oracle admits serial affine tetrahedra with blocked P1/P2 displacement only.")
        if penalty <= 0. or not np.isfinite(penalty):
            raise ValueError("Positive finite penalty required.")
        self.penalty = penalty
        self.coordinates = V.tabulate_dof_coordinates().copy()
        self.groups = periodic_groups(self.coordinates)
        self.size = len(self.coordinates)
        count = domain.topology.index_map(3).size_local
        self.dofs = np.array([V.dofmap.cell_dofs(c) for c in range(count)])
        geometry = domain.geometry.dofmaps[0]
        if geometry.shape[1] != 4:
            raise ValueError("Oracle requires affine four-node geometry.")
        vertices = domain.geometry.x[geometry[:count], :3]
        J = np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)
        determinant = np.linalg.det(J)
        if np.any(np.abs(determinant) < 1e-15):
            raise ValueError("Degenerate tetrahedron.")
        points, weights = basix.make_quadrature(basix.CellType.tetrahedron, bulk_degree)
        self.weights = np.abs(determinant)[:, None] * weights
        derivatives = element.tabulate(1, points)[1:4, :, :, 0]
        self.gradients = np.einsum("rqa,crj->cqaj", derivatives, np.linalg.inv(J))
        if not np.isclose(self.weights.sum(), 1., atol=1e-12, rtol=0.):
            raise ValueError("Expected unit reference volume.")
        reference_vertices = basix.geometry(basix.CellType.tetrahedron)
        face_points, face_weights = basix.make_quadrature(basix.CellType.triangle, contact_degree)
        self.faces = []
        for ids in basix.topology(basix.CellType.tetrahedron)[2]:
            xyz = vertices[:, ids]
            on_patch = np.all(np.isclose(xyz[:, :, 0], 1., atol=1e-12, rtol=0.), axis=1)
            on_patch &= np.all((xyz[:, :, 1:] >= .25-1e-12) & (xyz[:, :, 1:] <= .75+1e-12), axis=(1, 2))
            cells = np.flatnonzero(on_patch)
            if not len(cells):
                continue
            ref = reference_vertices[ids]
            mapped = ref[0] + face_points @ (ref[1:] - ref[0])
            basis = element.tabulate(0, mapped)[0, :, :, 0]
            area_scale = np.linalg.norm(np.cross(xyz[cells, 1] - xyz[cells, 0],
                                                  xyz[cells, 2] - xyz[cells, 0]), axis=1)
            self.faces.append((self.dofs[cells], basis, area_scale[:, None] * face_weights))
        self.area = sum(w.sum() for _, _, w in self.faces)
        if not np.isclose(self.area, .25, rtol=0., atol=1e-12):
            raise ValueError("Expected complete square contact patch of area 1/4.")

    def evaluate(self, displacement):
        u = np.asarray(displacement, dtype=float).reshape(self.size, 3)
        F = np.eye(3) + np.einsum("cai,cqaj->cqij", u[self.dofs], self.gradients)
        density, P = hencky_energy_stress(F)
        residual = np.zeros_like(u)
        cell_force = np.einsum("cq,cqij,cqaj->cai", self.weights, P, self.gradients)
        np.add.at(residual, self.dofs, cell_force)
        contact_energy = 0.
        force = 0.
        for dofs, basis, weights in self.faces:
            penetration = np.maximum(np.einsum("qa,ca->cq", basis, u[dofs, 0]) - .005, 0.)
            pressure = self.penalty * penetration
            local_force = np.einsum("cq,cq,qa->ca", weights, pressure, basis)
            np.add.at(residual[:, 0], dofs, local_force)
            force += np.sum(weights * pressure)
            contact_energy += .5 * np.sum(weights * pressure * penetration)
        return {"bulk_energy": float(np.sum(self.weights * density)),
                "contact_energy": float(contact_energy), "contact_force": float(force),
                "full_residual": residual}

    def reduce(self, residual):
        free = self.groups >= 0
        return np.bincount(self.groups[free], weights=np.asarray(residual).reshape(self.size, 3)[free],
                           minlength=int(self.groups.max()) + 1)

    def lift(self, reduced):
        result = np.zeros((self.size, 3))
        free = self.groups >= 0
        result[free] = np.asarray(reduced)[self.groups[free]]
        return result


def audit_elastic_station(step, *, bulk_degree=6, contact_degree=11, penalty=40000.):
    """Compare, without updating the material or modifying the accepted state."""
    material = step.material.summary()
    parameters = material.get("parameters", {})
    pair = step.contact_provider.pair
    surface = pair.rigid_body.surface
    if (material.get("kind") != "finite_strain_j2_logarithmic"
            or parameters.get("young") != 1000. or parameters.get("poisson") != .3
            or type(surface).__name__ != "RigidPlaneSurface"
            or tuple(surface.point) != (1.005, 0., 0.) or tuple(surface.normal) != (-1., 0., 0.)
            or pair.rigid_body.motion_schedule is not None or pair.friction is not None
            or pair.law.penalty != penalty or step.contact_provider.contact_integration != "gauss"):
        raise ValueError("Oracle requires the declared elastic cube/plane fixture.")
    state = step.response.state.committed_state_vectors()
    # This fixture's material identity is deliberately checked, not guessed
    # from its response being apparently elastic at one endpoint.
    if state.shape[-1] != 11 or not np.allclose(state[:, :9], np.eye(3).ravel(), atol=1e-13, rtol=0.) or np.any(state[:, 9:] != 0.):
        raise ValueError("Elastic oracle cannot audit a plastic-history state.")
    oracle = ElasticCubeOracle(step.solution, bulk_degree=bulk_degree,
                               contact_degree=contact_degree, penalty=penalty)
    reference = oracle.evaluate(step.solution.x.array)
    full = step.assemble_full_residual()
    try:
        actual = full.array.copy().reshape(-1, 3)
    finally:
        full.destroy()
    reduced = oracle.reduce(reference["full_residual"])
    difference = actual - reference["full_residual"]
    scale = max(np.linalg.norm(reference["full_residual"]), 1.)
    report = {k: v for k, v in reference.items() if k != "full_residual"}
    from dolfinx import fem
    stored = float(fem.assemble_scalar(fem.form(
        step.response.strain_energy_density.function * step.response.measure)))
    potential = float(step.contact_provider.evaluate().assembly.potential_energy)
    energy_error = max(abs(reference["bulk_energy"] - stored),
                       abs(reference["contact_energy"] - potential))
    energy_scale = max(abs(reference["bulk_energy"]), abs(reference["contact_energy"]), 1e-30)
    report.update(scope="independent discrete elastic virtual work, not continuum or plastic verification",
                  reduced_dofs=int(oracle.groups.max()) + 1,
                  full_residual_difference_l2=float(np.linalg.norm(difference)),
                  full_residual_relative_difference=float(np.linalg.norm(difference) / scale),
                  energy_relative_difference=float(energy_error / energy_scale),
                  independent_equilibrium_l2=float(np.linalg.norm(reduced)),
                  contact_reference_area=float(oracle.area))
    report["criteria"] = {"full_residual_relative_difference": 1e-10,
                          "independent_equilibrium_l2": 1e-7,
                          "energy_relative_difference": 1e-10}
    report["passed"] = all(report[key] <= limit for key, limit in report["criteria"].items())
    return report
