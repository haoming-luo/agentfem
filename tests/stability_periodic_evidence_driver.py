# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import json, runpy, hashlib
from pathlib import Path
import numpy as np
from agentfem import mesh, models, studies, fields, mechanics
from agentfem.constitutive import isotropic_elastic

column = runpy.run_path("tests/test_stability_periodic.py")["column"]
report = {
    "scope": "initial-stress buckling and linear matching-periodic elasticity; no postbuckling claim",
    "column": [],
    "honeycomb": [],
}
for n in (12, 24, 48):
    r = column(nx=n, length=20.0).solve_result()
    factor = float(r.quantity("load_factors")[0])
    expected = np.pi**2 * 1000.0 / (48.0 * 400.0)
    report["column"].append(
        {
            "nx": n,
            "factor": factor,
            "euler_stress": expected,
            "relative_error": abs(factor / expected - 1),
            "residual": float(r.quantity("relative_residuals")[0]),
        }
    )
for h in (0.10, 0.07, 0.04):
    d = mesh.reentrant_honeycomb(mesh_size=h).domain
    m = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=d
    )
    w = m.field(fields.displacement(d, degree=2))
    m.material(isotropic_elastic(young=1000.0, poisson=0.3, density=1.0))
    r = mechanics.periodic_elasticity(m, w, anchor=(0.0, 0.0))
    report["honeycomb"].append(
        {
            "mesh_size": h,
            "stiffness": r.quantity("effective_stiffness").tolist(),
            "poisson": r.quantity("poisson_ratios").tolist(),
            "solid_fraction": r.quantity("solid_fraction"),
            "hill_mandel_max": float(max(r.quantity("hill_mandel_relative_errors"))),
            "equilibrium_max": float(max(r.quantity("equilibrium_relative_errors"))),
        }
    )
report["source_hashes"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        Path("src/agentfem/mechanics/homogenization.py"),
        Path("src/agentfem/backends/_mpc_algebraic.py"),
        Path("src/agentfem/mechanics/buckling.py"),
        Path("src/agentfem/backends/_buckling.py"),
    ]
}
Path("evidence/stability_periodic/2026-10-10.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
