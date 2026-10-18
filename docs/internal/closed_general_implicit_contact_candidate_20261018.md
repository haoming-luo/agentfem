# Reviewed closed-solid implicit contact: bounded nonconvex candidate

Status: experimental source candidate on `feat/implicit-contact-parity`; not a
0.4.3 release capability, general CAD contact, or Abaqus parity claim.

## Why this is a separate geometry contract

An oriented open triangle sheet does not define a solid interior. Averaging
normals at a noncoplanar edge invents a gap law. A certified convex solid has
an efficient support-plane interior law, but that law cannot represent a
reentrant corner. `closed_triangulated_rigid_solid(...)` therefore accepts a
closed, consistently oriented triangle manifold with connected vertex links
and positive oriented volume only when the caller explicitly sets
`reviewed_embedded=True`. Point-touching shells fail before a solve. The
topology check does **not** prove absence of self-intersection, nested
component correctness or CAD fitness; the caller attests embedding after
review. `closed_triangulated_rigid_solid_from_mesh(...)` retains imported
source SHA-256, format and coordinate scale and does not automatically infer
that attestation from an STL file.

At every Newton trial, an exact triangle BVH finds the nearest surface point.
A direct oriented-solid-angle sum classifies the query as outside (winding 0)
or inside (winding 1). Positive signed distance is admissible, negative is
penetration. Distinct equidistant nearest points and ambiguous winding reject
the trial instead of selecting an arbitrary normal. A shared nearest point on
two triangles is allowed even when their facet normals differ: the physical
signed-distance gradient comes from the point-to-feature vector. Coplanar
triangulation seams are not physical edges. A subdivided straight sharp edge
remains an edge, not an artificial vertex.

Within a unique nearest-feature branch, the face Hessian is zero. For an edge
with unit tangent `e` and signed gap `g`, the Hessian is
`sign(g) * (I - e⊗e - n⊗n) / abs(g)`; for a vertex it is
`sign(g) * (I - n⊗n) / abs(g)`. The same residual and tangent enter the existing
reference-area penalty Operator, finite-J2 Newton/line search, prescribed
tool force--moment--work ledger, atomic accepted State, rollback and serial
checkpoint. At a geometrically nonsmooth point on the surface, the Hessian is
undefined and the trial is rejected. No normal is smoothed across materials,
features or facets.

The direct winding sum costs O(number of facets) per query. This is a bounded
serial reference geometry, **not** an optimized large-tool search; replacing
it with a certified hierarchical sign classifier would require a separate
accuracy and performance gate. The BVH still accelerates nearest-feature
search, but does not make the sign computation sublinear.
For a *convex* tool, prefer the separately certified
`convex_polyhedral_rigid_surface(...)` path: its support-plane interior law
has a defined deterministic limiting branch at an interior face-distance
tie. The general nearest-point law deliberately rejects such a tie rather
than claiming equivalent active-set convergence.

## Local evidence and limits

Focused source tests check interior/exterior classification of an L-prism,
reentrant-edge, reentrant-vertex and coplanar-seam responses, subdivided physical edges,
outward-orientation and review requirements, rigid-motion objectivity,
finite-difference tangent and work conjugacy. The edge force, potential and
tangent also match independent quadratic radial-distance oracles. A real
DOLFINx boundary quadrature Operator checks the reentrant-edge tangent at
current displacement. An ordinary serial finite-J2 `model.step()` uses the
reviewed L-prism's flat face, agrees with the equivalent planar tool and
restores a checkpoint. A separate affine-periodic finite-J2 Step checks
`TᵀR≈0` for the **total** material-plus-contact residual and the same planar
limit. Unsupported friction is rejected before assembly; a bounded two-rank
preflight rejects this serial-only implicit path before a solve. A separate
six-increment ordinary finite-J2 Step now has an **active reentrant-edge**
quadrature point at the accepted final state, nonzero equivalent plastic strain,
and positive accepted tool work. Its current contact tangent agrees with a
fixed-branch directional finite difference at that accepted state. The
assembled material-plus-contact Newton matrix also agrees with a finite
difference when the *preceding* accepted J2 state is held fixed. Reloading
from the final accepted yield-surface state is a branch kink and is not that
discrete-update tangent test. A halfway
checkpoint/restart reproduces its
final displacement, committed material state and tool work. An injected
post-energy-record failure restores the previous accepted displacement,
material state, tool pose and work. This is a small
manufactured FE regression; it does **not** establish a mesh, penalty or
increment limit, a same-input Abaqus comparison, or robust convergence across
arbitrary active-set switches.

An exploratory fixed-geometry sensitivity screen used the same unit-cube
finite-J2 fixture, tool travel, material and serial backend. These are
dimensionless fixture values, not a physical reference solution:

| Cells per cube edge | Penalty | Increments | Final x-contact force | Max PEEQ | Tool work | Active reentrant edge points |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 6 | -0.307884 | 0.004631 | 0.010731 | 1 |
| 1 | 20 | 12 | -0.308336 | 0.004577 | 0.010243 | 1 |
| 1 | 20 | 24 | -0.307757 | 0.004755 | 0.010383 | 1 |
| 1 | 20 | 48 | -0.307564 | 0.004770 | 0.010388 | 1 |
| 1 | 40 | 6 | -0.570216 | 0.008949 | 0.019391 | 1 |
| 2 | 20 | 6 | -0.307543 | 0.006715 | 0.008196 | 0 |

The 24→48 increment comparison changes the x-force by about 0.063%, max
PEEQ by 0.32% and tool work by 0.049%; that is only a *single-mesh load-path
screen*. The penalty change materially alters the result, and the refined
mesh samples a different active contact feature. The mechanical energy
residual also is not monotonically reduced by these increments; no complete
energy-closure or spatial-convergence assertion follows from this table.

The present tests are local software evidence, not an independent structural
benchmark. [NAFEMS contact benchmarks](https://www.nafems.org/publications/resource_center/r0094/)
motivate same-input comparison; a reference number from a different model
cannot validate this implementation. Abaqus' own
[contact-control documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm)
distinguishes finite-sliding surface, edge and vertex cases. The exact
solid-angle construction follows the geometric winding-number formulation
described by [Jacobson et al.](https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.html),
while [VTK's polyhedron note](https://docs.vtk.org/en/v9.7.0.rc0/design_documents/PolyhedronAlgorithms.html)
documents the triangle solid-angle primitive. Those references motivate the
algorithm; they do not certify our contact solver.

An isolated native Apple Silicon `0.4.4.dev0` wheel built from Git commit
`ce1dc177` has SHA-256
`116dffa0c6336b982392cc01ba80e24c28e0fea260766a9296bb38ff9e7e2489`.
Installed outside the checkout, it imports both `_hex_native` and
`_body_pipeline`; the ordinary three-increment finite-J2 L-prism Step
completes with accepted load factor 1 and tool work
`0.0016303122370297336` in the dimensionless flat-face fixture. This is an
installed-package smoke, **not** a global active reentrant-edge solve, an
independent reference or Windows/Linux qualification.

The later isolated native wheel from commit `db87ea56` has SHA-256
`2dd1ab6213ecadb459fa10a7b0ac08051f67a3ba6fd8f2cc18c1e7e893d3c08d`.
It was installed outside the checkout and, with the native `_hex_native` and
`_body_pipeline` modules present, ran the six-increment active-edge plastic
fixture: maximum PEEQ `0.004630797762548862`, active edge `True`, accepted
tool work `0.010730556638046503`. The standalone public-API
`examples/finite_j2_closed_concave_contact.py` also ran from that isolated
installation at 24 increments and exported an XDMF field file plus result
manifest. These are **Mac installed-package** checks of a manufactured
problem, not release-platform or industrial accuracy evidence.

### Sign-query performance experiment, not an activated fast path

`tools/benchmark_closed_solid_projection.py` fixes the L-prism geometry,
random seed, inside/outside query distribution and repeated local timings. On
this Apple Silicon host, with 5,000 queries and median of three in-process
trials, it measured:

| Facets | Full direct projection | BVH nearest search | Direct winding | Experimental BVH ray parity |
| ---: | ---: | ---: | ---: | ---: |
| 320 | 0.613 s | 0.435 s | 0.169 s | not screened in the final table |
| 1,280 | 0.963 s | 0.533 s | 0.416 s | 0.635 s |
| 5,120 | 2.115 s | 0.554 s | 1.525 s | 0.684 s |

The [Möller–Trumbore ray/triangle method](https://my.eng.utah.edu/~cs6965/papers/MT97.pdf)
motivated the strict-interior crossing prototype. It agrees with direct
winding on all 5,000 sampled queries at each of the two ray-screened
densities and defers
grazing rays, but a parity sign does not preserve the direct winding path's
query-time evidence for malformed or multiply wound geometry. It is therefore
kept **only in the research benchmark tool**, not in the installed solver.
The 1,280-facet local result is slower, the 5,120-facet local result has not
been shown to improve a whole accepted Newton Step, and no cross-machine
claim is made. This experiment does identify nearest-feature search as another
substantial cost, not just the O(facets) sign calculation.

Open work before broader admission: independent same-input force/displacement
comparison; controlled geometry, mesh, penalty and increment sensitivity;
large-tool sign acceleration with parity proof; distributed ownership and
cross-rank restart. Friction, deformable master, self-contact and arbitrary
unreviewed STL remain unsupported. Do not promote this branch into a release
merely because a small Newton solve completes.
