"""StVK shallow arch: public continuation through a force limit point."""

import json, runpy
from pathlib import Path
import numpy as np
from agentfem import constitutive

root = Path(__file__).resolve().parents[2]
solve = runpy.run_path(str(root / "examples/shallow_arch_arc_length/case.py"))[
    "solve_arch"
]
out = root / "validation/geometric_elasticity/results"
rows = []
for nx, size, steps in [(32, 0.04, 80), (64, 0.02, 160)]:
    r, u, load = solve(
        nx=nx,
        arc_size=size,
        increments=steps,
        material=constitutive.saint_venant_kirchhoff(young=1000, poisson=0.0),
        output=out / f"stvk_arch_{nx}.xdmf",
    )
    np.savetxt(
        out / f"stvk_arch_{nx}.csv",
        np.column_stack((-u, load)),
        delimiter=",",
        header="downward_crown_displacement,load_factor",
        comments="",
    )
    row = dict(
        nx=nx,
        arc_size=size,
        maximum_load_factor=float(max(load)),
        minimum_load_factor=float(min(load)),
        max_equilibrium_residual=float(max(r.quantity("equilibrium_residuals"))),
        passed_limit_point=bool(np.any(np.diff(load) < 0)),
        final_displacement=float(-u[-1]),
    )
    rows.append(row)
    (out / "arc_length.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
