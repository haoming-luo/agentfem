# Reviewed closed-solid implicit contact: bounded nonconvex candidate

Status: experimental source candidate on `feat/implicit-contact-parity`; not a
0.4.3 release capability, general CAD contact, or Abaqus parity claim.

## Why this is a separate geometry contract

An oriented open triangle sheet does not define a solid interior. Averaging
normals at a noncoplanar edge invents a gap law. A certified convex solid has
an efficient support-plane interior law, but that law cannot represent a
reentrant corner. `closed_triangulated_rigid_solid(...)` therefore accepts a
closed, consistently oriented triangle manifold with positive oriented
volume only when the caller explicitly sets `reviewed_embedded=True`. The
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
limit. Unsupported friction is rejected before assembly. These tests do not
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

Open work before broader admission: a stable active reentrant-edge global
path with plasticity and restart; independent same-input force/displacement
comparison; controlled geometry, mesh, penalty and increment sensitivity;
large-tool sign acceleration with parity proof; distributed ownership and
cross-rank restart. Friction, deformable master, self-contact and arbitrary
unreviewed STL remain unsupported. Do not promote this branch into a release
merely because a small Newton solve completes.
