"""Independent inextensible elastica BVP versus 2-D solid StVK solution."""

import runpy, json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp


def reference(alpha):
    # s is arc length/L, theta(0)=0 and free-end moment theta'(1)=0.
    # Force is fixed vertical: theta'' + alpha cos(theta) = 0.
    s = np.linspace(0, 1, 101)

    def rhs(s, z):
        return np.vstack((z[1], -alpha * np.cos(z[0]), np.cos(z[0]), np.sin(z[0])))

    def bc(a, b):
        return np.array([a[0], b[1], a[2], a[3]])

    initial = np.vstack(
        (alpha * (s - s * s / 2), alpha * (1 - s), s, alpha * s * s / 2)
    )
    solution = solve_bvp(rhs, bc, s, initial, tol=1e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)
    return dict(
        x=float(solution.y[2, -1]),
        y=float(solution.y[3, -1]),
        angle=float(solution.y[0, -1]),
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    solve = runpy.run_path(str(root / "examples/geometric_elastic_cantilever/case.py"))[
        "solve_cantilever"
    ]
    out = root / "validation/geometric_elasticity/results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in (1.0, 2.0, 4.0):
        ref = reference(alpha)
        for nx in (40, 80):
            r, c = solve(
                nx=nx, alpha=alpha, output=out / f"elastica_a{alpha}_n{nx}.xdmf"
            )
            row = dict(
                alpha=alpha,
                nx=nx,
                reference=ref,
                x=float(c["tip_x_over_L"][-1]),
                y=float(c["tip_y_over_L"][-1]),
            )
            row["relative_y_error"] = abs(row["y"] / ref["y"] - 1)
            rows.append(row)
            (out / "elastica.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
