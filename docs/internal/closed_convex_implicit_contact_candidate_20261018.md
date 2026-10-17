# Closed-convex sharp rigid tool: bounded implicit contact candidate

Status: source-level, serial finite-J2 candidate. This does not
promote arbitrary STL/CAD, general contact, or the 0.4.3 release contract.

## Geometry and Operator

`convex_polyhedral_rigid_surface(...)` accepts a closed, consistently oriented
triangle manifold only if every supplied vertex lies behind every outward
facet plane. It rejects open, inward-oriented and nonconvex tooling before a
solve. This is a geometry certificate for the exact supplied facets, not a
repair, remesh or convex-hull operation.
`convex_polyhedral_rigid_surface_from_mesh(...)` applies that same certificate
to the reviewed triangle-only mesh import, retaining source hash, format and
coordinate-scale evidence; it does not certify arbitrary STL files as convex.

For a point inside the tool, the signed gap is the maximum of its outward
unit-normal face-plane gaps. A negative value means penetration. The nearest
limiting face is selected by stable facet ID on a tie. Outside, the inactive
branch reports Euclidean distance to the actual triangle surface. Every
Newton trial reprojects at the current prescribed pose. The active residual
is the derivative of the reference-area normal penalty potential on the
selected face; its within-branch tangent has zero geometric Hessian. At a
noncoplanar face switch the potential remains continuous but the force has
distinct one-sided limits. The selected limiting tangent is a semismooth
branch, **not** a classical two-sided derivative or a smoothed normal.

The implementation checks every support plane for interior penetration and
uses the existing triangle BVH for exact closest-point projection of inactive
exterior points. The BVH may prune candidates but never freezes an old gap,
normal or contact force. This is not a general arbitrary-tool search. It shares
the existing rigid pair, finite-J2 Newton, strong-boundary lifting, tool
force/moment work, accepted State, rollback and checkpoint owners. It does not
add a new Step or a geometry-specific material law.

## Evidence and limits

Focused solver-neutral tests check closure/orientation rejection, interior and
exterior signed gaps, a noncoplanar feature switch, fixed-branch energy
gradient, tangent difference, and rigid-motion objectivity. A real DOLFINx
boundary operator checks its current-trial tangent under both fixed and
translated/rotated poses. A finite-J2 `model.step()` case compares the same
planar face supplied as a closed convex tool versus the older open triangle
patch; displacement and accepted tool work agree at the tested resolution.
An independent one-dimensional equilibrium also checks the homogeneous
elastic-regime Hencky bar against
`E log(lambda)/lambda + k(lambda - x_tool) = 0`, with the penalty potential
compared separately. This validates a bounded normal-contact limit, not a
curved pressure field or a plastic structure.

A separate twelve-increment finite-J2 fixture moves a convex tool far enough
that at least one *active* slave quadrature point switches between
noncoplanar face normals. The same path is run in an elastic regime and with
evolving plastic State. Every accepted Newton increment converges, plastic
equivalent strain becomes nonzero in the latter, and an interrupted checkpoint
resumed before the switch reproduces the continuous displacement, material
State and final prescribed-tool work. This demonstrates one selected-branch
transition path; it is not a convergence theorem for arbitrary corners, large
sliding, or chattering active sets.

An isolated, locally built `0.4.3` wheel was installed outside the source
checkout and ran `examples/finite_j2_closed_convex_contact.py` through the
ordinary `model.step()` path. One-cell/eight-increment and two-cell/
sixteen-increment cases report right-end displacements
`-0.0009990007775119401` and `-0.0009990007775003966`; the independently
solved scalar equilibrium gives `-0.0009990007774995924`. The examples use
the same uniform deformation, so this is an installed-package and exact-limit
smoke, not a mesh convergence study for a nonuniform contact patch. The wheel
was built for the local Apple Silicon environment, not released or tested on
Windows/Linux by this evidence.
These are local tests, not industrial contact validation.

### Plastic feature-switch sensitivity screen

The separate one-unit-cube, translating box-tool fixture uses finite-J2 with
yield stress 2, a tool-side lower y edge at 0.15, 0.06 tool travel, and no
applied traction. The table reports accepted end-state quantities; all lengths,
stresses and energies are in this dimensionless fixture's consistent units.
It is a diagnostic screen, not an independent reference solution.

| Changed control | Cases | Final tool work | Contact potential | Peak equivalent plastic strain |
| --- | --- | --- | --- | --- |
| cells per axis, 12 increments, penalty 500 | 1 / 2 / 3 / 4 / 6 / 8 | 0.23136 / 0.21312 / 0.21030 / 0.20709 / 0.20021 / 0.19604 | 0.08733 / 0.06091 / 0.05247 / 0.04999 / 0.04728 / 0.04681 | 0.04151 / 0.04627 / 0.04816 / 0.04828 / 0.05114 / 0.05381 |
| penalty, 4 cells per axis, 12 increments | 250 / 500 / 1000 / 2000 | 0.16686 / 0.20709 / 0.23540 / 0.25299 | 0.06559 / 0.04999 / 0.03261 / 0.01918 | 0.03634 / 0.04828 / 0.05668 / 0.06190 |
| increments, 4 cells per axis, penalty 500 | 6 / 12 / 24 / 48 | 0.20947 / 0.20709 / 0.20738 / 0.20772 | 0.04993 / 0.04999 / 0.05000 / 0.05003 | 0.04818 / 0.04828 / 0.04834 / 0.04836 |

The 6-to-8 spatial change is still about 2.1% in tool work and 5.2% in
peak plastic strain; the penalty change is far larger. Peak local strain near
a sharp edge is not a mesh-independent engineering quantity by assumption.
These observations keep the accuracy gate open. The short local wall times
were also measured, but are not transferable performance claims or a basis
for a general solver default.

### Selected-branch Coulomb candidate: adverse sensitivity evidence

The same dimensionless sharp-tool fixture was screened with Coulomb
coefficient `0.2` and tangential penalty `100`. With **normal-only** tool travel
`(-0.06, 0, 0)`, twelve fixed increments, penalty `500`, and 1/2/3/4/6 cells
per axis, accepted cumulative friction dissipation is approximately
`0.00246 / 0.000422 / 0 / 0.0000645 / 0`. This is not a monotone friction
limit. The motion primarily presses normally; edge-induced sliding can fall
below the stick/slip threshold as the boundary quadrature changes. It was
therefore rejected as a useful independent sliding benchmark.

Adding an explicit tangential tool travel of `0.06` makes every screened
1/2/3/4-cell case dissipative, but dissipation still changes substantially:

| Cells per axis | Tool work | Coulomb dissipation | Contact potential | Peak PEEQ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.26268 | 0.005910 | 0.09950 | 0.04336 |
| 2 | 0.24391 | 0.003161 | 0.07295 | 0.05062 |
| 3 | 0.24284 | 0.001773 | 0.06506 | 0.05697 |
| 4 | 0.23346 | 0.001158 | 0.06120 | 0.06388 |

At six cells a twelve-step fixed path rejects an increment at load factor
`0.75`; it is not silently accepted. Reusing the existing automatic cutback
Procedure accepts that mesh in 16 increments after one failed attempt, with
work `0.21655` and dissipation `0.002163`. At eight cells the same automatic
limits accept twelve increments with work `0.21177` and dissipation `0.000328`.
The irregular friction loss, changing peak PEEQ and contact-area sampling keep
sharp-friction accuracy **open**, even though the selected-branch tangent,
state transaction and one interrupted restart pass. These are local
diagnostics only; no Abaqus-equivalence or production default is inferred.

The candidate still lacks an external structural comparison for a sharp
feature, mesh and penalty studies, MPI assembly, independently validated
friction on sharp faces,
and whole-step performance evidence for larger tools. In an isolated synthetic
996-facet/1000-exterior-query check, the exterior projection stage changed
from 2.055 s to 0.266 s on one Apple Silicon host; this is not a complete
solve speedup or portable benchmark. At a medial-axis tie, one limiting branch is
selected deterministically; general convergence across repeated active-set
switches is not proven. Open or nonconvex triangle surfaces retain their
fail-closed active edge/vertex policy. Release language must distinguish
this narrow convex solid contract from Abaqus/Standard general contact.
