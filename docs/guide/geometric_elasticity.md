# Geometrically nonlinear elastic solids

AgentFEM now accepts an explicit Saint-Venant–Kirchhoff (StVK) material through
`model.step(...)`. Use it when an elastic solid undergoes appreciable rotation
while its material strain remains small. The same implementation supports 3D,
2D plane strain and an exactly condensed 2D plane-stress law.

## Standard application workflow

```python
from agentfem import constitutive, fields, models, studies, steps, solvers

model = models.create(
    mesh=domain,
    study=studies.static_solid(
        dimension=2, assumption="plane_stress", nonlinear=True,
    ),
)
u = model.field(fields.displacement(domain, degree=2))
model.material(constitutive.saint_venant_kirchhoff(
    young=210000., poisson=.3, plane_stress=True,
))
model.clamp(u, on=left)
model.traction((-107., 0.), on=right)
step = model.step(
    target=u,
    incrementation=steps.automatic(initial=.025, maximum=.025),
    solver_options=solvers.NewtonSolverOptions(
        line_search="basic", absolute_tolerance=1e-7,
    ),
)
result = step.solve_result(output="displacement.xdmf")
```

Here stress is in MPa and length in mm. The nominal traction is a reference-area
load of 107 N/mm², corresponding to 10.7 kN over a 10 × 10 mm face. In 2D the
assembled force is **per unit reference thickness**: multiply the reaction by
10 mm to recover the specimen force. The loaded end remains free to rotate;
prescribing the same axial displacement across the end would define a
different experiment.

Complete executable projects:

- `examples/geometric_elastic_column`: initial imperfection, axial dead load,
  actual displacement field and load–lateral displacement CSV.
- `examples/geometric_elastic_cantilever`: a thin cantilever with large
  rotation under a fixed-direction transverse traction.
- `examples/shallow_arch_arc_length`: accepts an explicit StVK `material=`
  for the existing conservative serial continuation procedure.

The full Newton setting above was tested for these smooth elastic benchmarks.
It is an explicit **example setting**, not a change to platform defaults. Load
cutback remains active. Other problems may benefit from backtracking.

## Material and consistent tangent

With reference gradient `Grad`, define

\[
F=I+\operatorname{Grad}u,\qquad E=\tfrac12(F^T F-I),
\]
\[
\Psi=\tfrac\lambda2(\operatorname{tr}E)^2+\mu E:E,
\quad S_{\mathrm{PK2}}=\lambda\operatorname{tr}(E)I+2\mu E,
\quad P=F S_{\mathrm{PK2}}.
\]

`young` and `poisson` determine the Lamé constants. The residual integrates
`P : Grad(v)` over the reference domain. Its automatic derivative provides the
material and geometric parts of the tangent together; no second geometric
stiffness is added. A rigid rotation produces zero strain and energy.

For plane strain, `F33=1`. For plane stress, the eliminated thickness satisfies

\[
E_{33}=-\frac\lambda{\lambda+2\mu}(E_{11}+E_{22}),\quad
F_{33}=\sqrt{1+2E_{33}},\quad
\bar\lambda=\frac{2\mu\lambda}{\lambda+2\mu}.
\]

The reduced energy uses `bar(lambda)` and the in-plane Green strain. Its
consistent derivative includes the thickness relaxation. A non-positive
thickness radicand is rejected; the implementation never clips it to a small
positive value. Finite-strain acceptance checks sample quadrature points and
are not a proof of positivity at every point of an element.

## Outputs and interpretation

For this material, `results.finite_strain_cell_fields(u, material, ...)` embeds
2D results into 3D, including the relaxed thickness:

| Field | Meaning |
|---|---|
| `F` | Deformation gradient; plane stress includes `F33` |
| `GREEN` | Green–Lagrange strain |
| `E` / `LE` | Logarithmic strain, following the existing output convention |
| `P` | First Piola–Kirchhoff stress |
| `S` | **Cauchy stress**, not second Piola–Kirchhoff stress |
| `J` | Full volume ratio, including plane-stress thickness |
| `SENER` | Energy per reference volume |
| `EVOL` | Integral of `J` over the reference element |

For accepted traction-driven histories, use
`mechanics.load_controlled_response(model, step, support=left, monitor=right)`.
It reconstructs support reactions at each saved state and measures the
reference-boundary-average monitor displacement. The support must not share
nodes with the traction region. Body loads and amplitude-driven loads are
outside this helper's scope; the underlying solver has a broader load API.

## Scope and limits

- This is a distinct material law, not renamed Neo-Hookean elasticity.
  Existing `isotropic_elastic` and `nonlinear=True` semantics do not change.
- StVK can lose physical suitability at large material strains even with
  positive `J`. No general large-strain, rubber or extreme-compression claim
  is made.
- The convenience step currently consumes one material assignment. Initial
  stress, thermal eigenstrain, near-incompressible mixed treatment, plasticity,
  contact and automatic dynamics are not included.
- Reference dead tractions are supported. Pressure/follower loading is
  rejected by the StVK convenience path in this release.
- The standard Newton path was exercised in serial and with two MPI ranks;
  this is not a scalability claim. The existing arc-length adapter is serial.
- Force-controlled failure near a limit point is not an instability proof.
  The separate shallow-arch validation checks continuation past a load maximum.

## Point constraints

`constraints.pin(...)` now checks each requested component using MPI-global
owned degrees of freedom. A point absent from the mesh raises
`AFM-CONSTRAINT-PIN-EMPTY`, including its name and requested coordinate.
A rank with no local hit is valid when another rank owns the matching node.
Constraint summaries include the requested coordinate, actual matched
coordinates, global count and tolerance. A successful point match does not
establish that the complete support system removes all rigid-body modes.

## Verification and reproducibility

See `validation/geometric_elasticity/REPORT.md` for measured values and commands.
Tests cover energy/stress/tangent consistency, rigid rotations, thickness
condensation, small-load convergence, invalid-state rollback, restart, standard
provider selection and serial continuation. Structural evidence includes the
Euler column, two imperfections with mesh/increment studies, and an independent
elastica boundary-value solution.

References:

- [COMSOL: linear elasticity with geometric nonlinearity](https://www.comsol.com/blogs/modeling-linear-elastic-materials-how-difficult-can-it-be/).
- [Bisshopp and Drucker, Large deflection of cantilever beams (1945)](https://doi.org/10.1090/qam/13360).
