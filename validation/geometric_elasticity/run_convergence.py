"""Reproducible mesh/increment study; the solver remains in the package."""

import json
import runpy
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
solve = runpy.run_path(str(root / "examples/geometric_elastic_column/case.py"))[
    "solve_column"
]
out = root / "validation/geometric_elasticity/results"
out.mkdir(parents=True, exist_ok=True)
rows = []
for a in (0.1, 0.2):
    for nx, ny, increment in [
        (40, 2, 0.05),
        (80, 4, 0.05),
        (120, 6, 0.05),
        (120, 6, 0.025),
    ]:
        r, c, s = solve(
            nx=nx,
            ny=ny,
            imperfection=a,
            increment=increment,
            output=out / f"column_a{a}_n{nx}_dt{increment}.xdmf",
        )
        name = f"column_a{a}_n{nx}_dt{increment}"
        np.savetxt(
            out / f"{name}.csv",
            np.column_stack(list(c.values())),
            delimiter=",",
            header=",".join(c),
            comments="",
        )
        row = dict(
            imperfection_mm=a,
            nx=nx,
            ny=ny,
            increment=increment,
            tip_mm=float(c["added_tip_displacement_mm"][-1]),
            reaction_N=float(c["support_reaction_N"][-1]),
            maximum_reaction_error_N=float(
                np.max(np.abs(c["support_reaction_N"] - c["load_N"]))
            ),
            minimum_J=min(
                i.checks["minimum_quadrature_J"] for i in s.last_solve_info.increments
            ),
            accepted_increments=len(s.accepted_increments),
            attempted_increments=len(s.attempted_increments),
        )
        rows.append(row)
        (out / "convergence.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
