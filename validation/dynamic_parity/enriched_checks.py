# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent consistent-mass reference for the published enriched element."""

from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh
from element_checks import reference_matrices
from agentfem.elements.tet15_mass import scalar_element, nodal_weights

rows = []
for J in (np.eye(3), np.array([[1, 0.2, 0], [0, 2, 0.1], [0, 0, 3.0]])):
    el = scalar_element()
    x = el.points @ J.T
    K, M = reference_matrices(x, element=el)
    m = 2 * abs(np.linalg.det(J)) * nodal_weights()
    D = np.diag(np.repeat(m, 3))
    nulls = np.linalg.eigvalsh(K)
    assert sum(abs(nulls) < 1e-10 * max(nulls)) == 6
    rigid = []
    for axis in np.eye(3):
        rigid.extend([np.tile(axis, len(x)), np.cross(axis, x).ravel()])
    residual = max(
        np.linalg.norm(K @ r) / (np.linalg.norm(K) * np.linalg.norm(r)) for r in rigid
    )
    assert residual < 1e-12
    np.testing.assert_allclose(M.sum(axis=1), np.repeat(m, 3), atol=1e-13)
    rotations = [np.cross(axis, x - x[:4].mean(0)).ravel() for axis in np.eye(3)]
    inertia = [abs(r @ D @ r / (r @ M @ r) - 1) for r in rotations]
    assert max(inertia) < 1e-12
    errors = []
    for e in [
        np.diag([1, 0, 0]),
        np.eye(3),
        np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
    ]:
        u = (x @ e.T).ravel()
        stress = 80 * e + 40 * np.trace(e) * np.eye(3)
        exact = abs(np.linalg.det(J)) / 6 * np.sum(e * stress)
        errors.append(abs(u @ K @ u / exact - 1))
    assert max(errors) < 1e-12
    wc = np.sqrt(eigh(K, M, eigvals_only=True)[6:])
    wl = np.sqrt(eigh(K, D, eigvals_only=True)[6:])
    rows.append(
        {
            "element": "tet15_nodal",
            "jacobian": J.tolist(),
            "rigid_residual": float(residual),
            "zero_modes": 6,
            "constant_strain_energy_errors": errors,
            "minimum_nodal_mass": float(m.min()),
            "rotation_inertia_relative_errors": inertia,
            "consistent_angular_frequencies": wc.tolist(),
            "lumped_angular_frequencies": wl.tolist(),
            "frequency_relative_errors": abs(wl / wc - 1).tolist(),
            "artificial_energy": 0.0,
            "abaqus_C3D10M_equivalent": False,
        }
    )
out = Path(__file__).parent / "evidence"
(out / "tet15_element_checks.json").write_text(json.dumps(rows, indent=2) + "\n")
print(
    "Tet15: six rigid modes, constant strain and exact rotation inertia passed",
    flush=True,
)
