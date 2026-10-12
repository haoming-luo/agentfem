"""Actual linear preload, then public eigenbuckling; SI-consistent N/mm/MPa."""

from pathlib import Path
import json
import numpy as np
from agentfem import mesh, models, studies, fields, constitutive

out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
rows = []
for dim, nx, ny in [(2, 40, 2), (2, 80, 4), (2, 120, 6), (3, 40, 3)]:
    d = (
        mesh.rectangle((0.0, -5.0), (200.0, 5.0), (nx, ny))
        if dim == 2
        else mesh.cuboid(
            (0.0, -5.0, -5.0), (200.0, 5.0, 5.0), (nx, ny, ny), cell_type="hexahedron"
        )
    )
    assumption = "plane_stress" if dim == 2 else None
    material = constitutive.isotropic_elastic(young=210000.0, poisson=0.3, density=1.0)
    left = mesh.boundary(d, lambda x: np.isclose(x[0], 0.0), name="clamp")
    right = mesh.boundary(d, lambda x: np.isclose(x[0], 200.0), name="free_end")
    linear = models.create(
        mesh=d, study=studies.static_solid(dimension=dim, assumption=assumption)
    )
    u = linear.field(fields.displacement(d, degree=2))
    linear.material(material)
    linear.clamp(u, on=left)
    linear.traction((-1.0,) + (0.0,) * (dim - 1), on=right)
    linear.step(target=u).solve()
    buckling = models.create(
        mesh=d, study=studies.buckling_solid(dimension=dim, assumption=assumption)
    )
    v = buckling.field(fields.displacement(d, degree=2))
    buckling.material(material)
    buckling.clamp(v, on=left)
    r = buckling.step(
        target=v,
        reference_displacement=u.value,
        reference_name="1 MPa compressive dead traction",
        modes=1,
    ).solve_result()
    force = float(r.quantity("load_factors")[0]) * 100
    expected = np.pi**2 * 210000 * (10 * 10**3 / 12) / (4 * 200**2)
    row = dict(
        dimension=dim,
        nx=nx,
        ny=ny,
        critical_load_N=force,
        Euler_load_N=expected,
        relative_error=abs(force / expected - 1),
        residual=float(r.quantity("relative_residuals")[0]),
    )
    rows.append(row)
    (out / "euler.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
