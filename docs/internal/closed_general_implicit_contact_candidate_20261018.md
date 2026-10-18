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
preflight rejects this serial-only implicit path before a solve. These tests do not
yet demonstrate a converged *global* J2 load path with active reentrant-edge
contact, nor do they establish a mesh/penalty/increment limit for that path.

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

Open work before broader admission: a stable active reentrant-edge global
path with plasticity and restart; independent same-input force/displacement
comparison; controlled geometry, mesh, penalty and increment sensitivity;
large-tool sign acceleration with parity proof; distributed ownership and
cross-rank restart. Friction, deformable master, self-contact and arbitrary
unreviewed STL remain unsupported. Do not promote this branch into a release
merely because a small Newton solve completes.
