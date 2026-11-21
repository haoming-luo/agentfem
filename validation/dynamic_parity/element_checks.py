# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent quadrature checks for the explicitly named P2/HRZ route."""

import json
from pathlib import Path
import basix
import numpy as np
from scipy.linalg import eigh
from agentfem.elements.tet10_mass import hrz_tet10_mass


def reference_matrices(x, E=100.0, nu=0.25, rho=2.0, element=None):
    el = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.tetrahedron,
        2,
        basix.LagrangeVariant.equispaced,
    )
    if element is not None:
        el = element
    q, w = basix.make_quadrature(basix.CellType.tetrahedron, 8)
    tab = el.tabulate(1, q)
    J = (x[1:4] - x[0]).T
    w = w * abs(np.linalg.det(J))
    grad = np.stack([tab[j, :, :, 0] for j in (1, 2, 3)], axis=-1) @ np.linalg.inv(J)
    eps = np.zeros((len(q), len(x) * 3, 3, 3))
    for node in range(len(x)):
        for c in range(3):
            eps[:, node * 3 + c, c, :] += 0.5 * grad[:, node, :]
            eps[:, node * 3 + c, :, c] += 0.5 * grad[:, node, :]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    stress = 2 * mu * eps + lam * np.trace(eps, axis1=2, axis2=3)[
        :, :, None, None
    ] * np.eye(3)
    K = np.einsum("q,qijk,qljk->il", w, eps, stress)
    N = tab[0, :, :, 0]
    M = np.kron(rho * np.einsum("q,qi,qj->ij", w, N, N), np.eye(3))
    return K, M


def check_element(J):
    el = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.tetrahedron,
        2,
        basix.LagrangeVariant.equispaced,
    )
    x = el.points @ J.T
    K, M = reference_matrices(x)
    m = hrz_tet10_mass(x, 2.0)
    D = np.diag(np.repeat(m, 3))
    rigid = []
    for axis in np.eye(3):
        rigid.extend([np.tile(axis, 10), np.cross(axis, x).ravel()])
    residual = max(
        np.linalg.norm(K @ r) / (np.linalg.norm(K) * np.linalg.norm(r)) for r in rigid
    )
    eigenvalues = np.linalg.eigvalsh(K)
    assert sum(abs(eigenvalues) < 1e-10 * np.max(eigenvalues)) == 6
    assert residual < 1e-12
    strains = [
        np.diag([1, 0, 0]),
        np.eye(3),
        np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
    ]
    errors = []
    for e in strains:
        u = (x @ e.T).ravel()
        stress = 80 * e + 40 * np.trace(e) * np.eye(3)
        exact = abs(np.linalg.det(J)) / 6 * np.sum(e * stress)
        errors.append(abs(u @ K @ u / exact - 1))
    assert max(errors) < 1e-12
    np.testing.assert_allclose(m.sum(), 2 * abs(np.linalg.det(J)) / 6, rtol=1e-13)
    np.testing.assert_allclose(m @ x / m.sum(), x[:4].mean(0), atol=1e-14)
    rotations = [np.cross(axis, x - x[:4].mean(0)).ravel() for axis in np.eye(3)]
    inertia = [abs(r @ D @ r / (r @ M @ r) - 1) for r in rotations]
    wc = np.sqrt(eigh(K, M, eigvals_only=True)[6:])
    wl = np.sqrt(eigh(K, D, eigvals_only=True)[6:])
    return {
        "jacobian": J.tolist(),
        "rigid_residual": float(residual),
        "zero_modes": 6,
        "constant_strain_energy_errors": errors,
        "mass": float(m.sum()),
        "minimum_nodal_mass": float(m.min()),
        "rotation_inertia_relative_errors": inertia,
        "consistent_angular_frequencies": wc.tolist(),
        "hrz_angular_frequencies": wl.tolist(),
        "frequency_relative_errors": (abs(wl / wc - 1)).tolist(),
        "artificial_energy": 0.0,
        "hourglass_control": "none; full-integration P2 has six rigid null modes only",
        "formulation": "P2 Lagrange + HRZ, NOT C3D10M",
    }


if __name__ == "__main__":
    records = [
        check_element(J)
        for J in (np.eye(3), np.array([[1, 0.2, 0], [0, 2, 0.1], [0, 0, 3.0]]))
    ]
    out = Path(__file__).parent / "evidence"
    out.mkdir(exist_ok=True)
    (out / "element_checks.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
