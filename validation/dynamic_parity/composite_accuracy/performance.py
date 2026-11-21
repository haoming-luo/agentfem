# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Public Step end-to-end timing; identical synthetic workload for each revision."""

import argparse
import json
import platform
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
from mpi4py import MPI
from agentfem import mesh, models, studies, fields, elements, constitutive


def sample(output):
    t = perf_counter()
    d = mesh.cuboid(
        (0, 0, 0), (1, 1, 1), (3, 3, 3), comm=MPI.COMM_SELF, cell_type="tetrahedron"
    )
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=d
    )
    u = m.field(
        fields.displacement(
            d, element_policy=elements.composite_tet10_policy(mean_dilatation=True)
        )
    )
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    m.clamp(u, on=mesh.boundary_region(d, lambda x: np.isclose(x[0], 0)))
    m.nodal_force((1, 0.2, -0.1), at=(1, 1, 1))
    s = m.step(target=u, dt=0.001, steps=100, save_every=10, progress=False)
    setup = perf_counter() - t
    with tempfile.TemporaryDirectory() as directory:
        result = s.solve_result(
            output=Path(directory) / "fields.xdmf" if output else None
        )
    return dict(
        setup_s=setup,
        total_s=perf_counter() - t,
        displacement_norm=float(np.linalg.norm(s.state.u.x.array)),
        external_work=float(s.history_records[-1]["external_work"]),
        final_u=s.state.u.x.array.tolist(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    data = dict(
        revision=a.revision,
        platform=platform.platform(),
        numpy=np.__version__,
        cells=162,
        steps=100,
        no_output=[sample(False) for _ in range(3)],
        xdmf=[sample(True) for _ in range(3)],
    )
    Path(a.output).write_text(json.dumps(data, indent=2) + "\n")
