# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Same-case rank-count and checkpoint evidence; no private solver assembly."""

from pathlib import Path
import json
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dm
from agentfem import models, studies, fields, constitutive, amplitudes, results

comm = MPI.COMM_WORLD
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def build():
    domain = dm.create_unit_cube(comm, 2, 2, 2)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
    )
    u = m.field(fields.displacement(domain, degree=2))
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    m.nodal_force(
        (1, 2, 3),
        at=(1, 1, 1),
        amplitude=amplitudes.tabular([0, 0.0004, 0.001], [0, 1, 0]),
    )
    s = m.step(
        target=u,
        mass=m.lumped_mass(u, method="tet10_hrz"),
        dt=1e-4,
        steps=10,
        progress=False,
    )
    h = results.dynamic_history(
        "sensor", at=(0.75, 0.9, 0.95), length_unit="m", time_unit="s"
    )
    return s, h


full, h = build()
full.solve_result(history=h)
partial, h = build()
partial.run(history=h, until_step=4)
partial.save_checkpoint(out / "checkpoint")
resumed, h = build()
resumed.load_checkpoint(out / "checkpoint")
result = resumed.solve_result(history=h)
for key in ("u", "v", "a"):
    np.testing.assert_allclose(
        getattr(resumed.state, key).x.array,
        getattr(full.state, key).x.array,
        rtol=1e-12,
        atol=1e-14,
    )
keys = sorted(full.history_records[0])
a = np.array([[r[k] for k in keys] for r in full.history_records])
b = np.array([[r[k] for k in keys] for r in resumed.history_records])
np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
if comm.rank == 0:
    results.write_dynamic_history(result, out / "history.csv")
    (out / "records.json").write_text(
        json.dumps(
            {
                "ranks": comm.size,
                "history": full.history_records,
                "restart_max_abs": float(np.max(abs(a - b))),
                "stability": full.stability.summary(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{comm.size} ranks: U/V/A and all history channels restart identically",
        flush=True,
    )
