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
The vertex-link requirement is scoped to a **closed shell**. An open Q1
reference patch may have two corner-adjacent pieces used only for distance
search; forcing the closed-solid rule on every open triangle surface broke
that existing search case in remote CI. The corrected preflight retains
closed-shell rejection and preserves open-patch search. Open surface summaries
state `manifold_scope="edge_incidence_only"` rather than implying a
closed-solid vertex certificate.

At every Newton trial, an exact triangle BVH finds the nearest surface point.
A point certified beyond the complete tool AABB is exterior. The default
`winding_reference` sign path then uses a direct oriented-solid-angle sum
inside the AABB to classify outside (winding 0) or inside (winding 1).
An explicit `angle_weighted_pseudonormal` option instead uses the nearest
face/edge/vertex normal, with direct winding when the dot product is near
roundoff. It is valid under the separately reviewed embedded-manifold
assumption; it does **not** provide the winding path's per-query malformed-
geometry evidence. The two methods have different scientific checkpoint
identities. Positive signed distance is admissible, negative is
penetration. Distinct equidistant nearest points and ambiguous sign reject
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

The default direct winding sum costs O(number of facets) per query. The
opt-in nearest-feature sign has O(1) query work after O(facets) angle-weighted
normal preparation, but the nearest-point BVH still dominates some cases.
The fast route follows the proof of [Bærentzen and Aanæs (2005)](https://doi.org/10.1109/TVCG.2005.49)
for an embedded, outward-oriented closed two-manifold. The constructor checks
topology and orientation but **does not prove non-self-intersection**; the
caller must review that assumption before opting in. A fast sign on a bad
mesh is not made trustworthy by matching one short test.
A seeded 2,000-query regression now concentrates logarithmically from
`1e-8` to `1e-1` reference lengths around the L-prism's reentrant edge;
opt-in pseudonormal and direct winding agree on validity, status, signed
gap and normal for these sampled points. This is a local near-feature
screen, not an embedding certificate or an exhaustive proof of sign parity.
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
limit. Bounded Coulomb friction on an active locally planar face uses the
existing fixed-old-state tangent and accepted tangential State. An active
physical edge or vertex rejects the trial and rolls back; sharp-feature
friction is not approximated by the planar law. A bounded two-rank preflight
rejects this serial-only implicit path before a solve. A separate
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

With the same active reentrant edge under prescribed translation plus a small
rotation, a separate accepted-state regression independently integrates the
unsmoothed point tractions about the current tool reference point. Its force
and moment agree with the provider dual, and the fixed-displacement contact
potential derivative agrees with their generalized work. This checks a
current smooth branch of the nonsmooth geometry, not work convergence through
a feature switch.

A separate three-dimensional pressure-controlled block uses only tangential
restraints on its opposite face, so no hidden normal support can carry the
imposed force. With the reviewed L-prism's planar face active, the integrated
contact resultant balances the independently prescribed `0.5` unit load,
and the reference-area mean penetration is `0.5 / penalty` across two tetrahedral
mesh densities and penalties 500, 1,000 and 2,000. This follows the load-balance
check used in [Abaqus' finite-sliding deformable/rigid verification](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefrigid.htm),
but our model is a different manufactured input. It validates normal-force
assembly on a flat branch, not the reentrant-edge constitutive response or a
same-deck code comparison.

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

A further *same-geometry, penalty-20, 12-increment* diagnostic exposes the
sharp-feature integration problem rather than closing it:

| Cell divisions per cube edge | Element/face quadrature | x-force | Max PEEQ | Tool work | Active points | Active edge points |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | tetra/default | -0.307580 | 0.006719 | 0.008062 | 10 | 0 |
| 3 | tetra/default | -0.351190 | 0.008529 | 0.008856 | 22 | 0 |
| 4 | tetra/default | -0.338940 | 0.008319 | 0.008825 | 42 | 0 |
| 5 | tetra/default | -0.319805 | 0.008538 | 0.008474 | 65 | 1 |
| 2 | Hex8/2×2 | -0.336987 | 0.007088 | 0.008496 | 6 | 0 |
| 2 | Hex8/3×3 | -0.361257 | 0.007248 | 0.009105 | 14 | 0 |
| 2 | Hex8/4×4 | -0.272692 | 0.005181 | 0.008063 | 31 | 1 |
| 3 | Hex8/2×2 | -0.347405 | 0.007775 | 0.008763 | 14 | 0 |
| 3 | Hex8/3×3 | -0.328525 | 0.007376 | 0.008784 | 36 | 0 |
| 3 | Hex8/4×4 | -0.316077 | 0.007179 | 0.008636 | 60 | 0 |

These are isolated source-checkout screens, not a mesh/face-rule convergence
sequence: the sampled active feature and the force change nonmonotonically.
The accepted `SimulationResult` now reports raw active `face`/`edge`/`vertex`
quadrature counts for this reviewed closed-solid route; it classifies coplanar
triangle seams as faces and does not relabel the point samples as a continuous
pressure map. This makes the failure mode visible without changing the model.
The present point-quadrature Operator is therefore **not promoted** for a
sharp-edge-dominated structural response on this evidence. An optional
positive-weight Basix degree-5/7 triangle rule is now available for
tetrahedral CG1/CG2 traces, in addition to the existing Hex8 face rules.
It preserves the default three-point triangle rule and enters checkpoint
identity. CG1/CG2 reference-area and planar-force checks pass, as does a
two-rank CG1 partition-canonical point-ID/weight check; the **implicit
solver** remains serial. On the same tetrahedral 12-increment fixture,
degree 7 gives x-force
`-0.336957`, `-0.319265`, `-0.320473` for 2, 3, 4 cell divisions; the 3→4
values are close but the 2→3 change is about 5.3%, max PEEQ still changes,
and one physical edge point appears or disappears with sampling. Degree 5
gives `-0.334865` and `-0.328318` for 2 and 3 divisions. These are
diagnostics, not a converged method or a reason to change the default.
The public manufactured example exposes the rule without changing solver code:
`python examples/finite_j2_closed_concave_contact.py --cells 3 --increments 12 --output /tmp/contact-default`
and the same command with `--contact-quadrature-degree 7 --output /tmp/contact-degree7`
write separate XDMF/manifest outputs. This makes the face-rule sensitivity
reproducible without interpreting the option as an accuracy certificate.
Both 3-cell/12-increment public runs completed in this worktree. The default
three-point rule gave x-resultant `-0.3511900373`, zero sampled reentrant-edge
points and maximum PEEQ `0.0085291430`; the positive degree-7 rule gave
`-0.3192648429`, one sampled edge point and maximum PEEQ `0.0082096584`.
The result manifests identify their different face rules. This approximately
9.1% force difference at a fixed mesh is evidence of **unresolved contact
integration sensitivity**, not a degree-7 accuracy gain.
Keeping the 12-increment path, penalty and degree-7 rule fixed, the public
example at 5 and 6 cells per axis gave x-resultants `-0.3248313540` and
`-0.3268716591`, sampled physical-edge counts 1 and 3, and maximum PEEQ
`0.0085586637` and `0.0088500549`. The six-cell default-rule counterpart
gave x-resultant `-0.3431021957`, zero sampled edge points and maximum PEEQ
`0.0089111601`. Thus even the six-cell rules differ by about 4.7% in
x-resultant; closer degree-7 forces between successive meshes do not certify
contact-footprint or plastic-history convergence.
At six cells and degree 7, a fixed-mesh penalty bracket of 10/20/40 gave
x-resultants `-0.1712914511`, `-0.3268716591`, `-0.5945569639`, maximum
penetrations `0.04883127`, `0.04738339`, `0.04491391`, and maximum PEEQ
`0.00452155`, `0.00885005`, `0.01589832`. The prescribed tool translation
is `0.055` in this dimensionless fixture, so these are **large penetration
fractions** and a deliberately soft algorithmic test, not a hard-contact
reference. All three Steps complete; none provides a penalty-limit
certificate. A release benchmark must choose physically meaningful scaling
and show simultaneous penetration, force and history convergence.
This distinction also matters for cross-solver comparison: Abaqus documents
[hard, direct, augmented-Lagrange and penalty enforcement as distinct
Standard choices](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-itnhelppropcontactmechnormal.htm);
matching geometry alone does not make a soft penalty fixture a same-input
hard-contact reference.
At six cells, degree 7 and penalty 20, repeating with 24 rather than 12
increments gave x-resultant `-0.3268696751`, maximum PEEQ `0.0088462369`,
maximum penetration `0.04738336` and tool work `0.0086537641`, compared with
`-0.3268716591`, `0.0088500549`, `0.04738339` and `0.0086516508` at 12.
This narrow path-step screen is much less sensitive than the face rule or
penalty bracket; further increment reduction alone will not cure the exposed
contact-integration and penalty-limit gap.
An exploratory degree-7, six-cell trial at penalty 200 fails at load factor
0.5 with both 12 and 24 fixed increments. After the general finite-J2
failure-diagnostic repair, the reported cause is `line search found no
acceptable trial step at residual 0.00039544` for the 12-increment attempt;
it is not silently counted as an accepted result. A dedicated regression
checks the rejection reason reaches the event/exception and that accepted
state rolls back. This failure does not establish whether a physically
scaled hard-contact formulation would converge under another method.
The existing `line_search="basic"` full-Newton policy, now selectable in the
public research example with `--line-search basic`, does complete the same
six-cell/12-increment degree-7 fixture at penalties 200, 1000 and 10000.
Reducing the backtracking minimum step from `1/128` to `1e-4` in a one-off
diagnostic still rejected near load factor 0.5; the issue is not merely that
the default minimum step is too large.
Their x-resultants are `-1.82327209`, `-3.11990251`, `-3.94146457`;
maximum penetrations are `0.03414494`, `0.01850543`, `0.00543376` and
maximum PEEQ values are `0.04199805`, `0.06374637`, `0.08467591`.
The penalty response has **not plateaued** and plastic history remains
strongly parameter-dependent. This comparison isolates the backtracking
globalization issue; a successful full-Newton path is neither a proof that
`basic` is universally robust nor a hard-contact accuracy certificate.
At penalty 200, repeating the full-Newton run with 24 instead of 12
increments gave x-resultant `-1.82503468`, maximum PEEQ `0.04178001` and
maximum penetration `0.03418063`; it is a narrow path-step check, not spatial
or penalty convergence.
An exact-head installed-wheel control at the same six-cell mesh, 12 increments,
penalty 200 and `basic` Newton policy compared the three supported face rules:

| Triangle rule | x-resultant | max PEEQ | max penetration | active reentrant-edge points |
| --- | ---: | ---: | ---: | ---: |
| default degree 2 | -1.81303789 | 0.04410058 | 0.03238988 | 0 |
| opt-in degree 5 | -1.82587935 | 0.04396510 | 0.03360974 | 1 |
| opt-in degree 7 | -1.82327209 | 0.04199805 | 0.03414494 | 0 |

The x-force changes by about 0.14% from degree 5 to 7 relative to the
degree-7 value, but maximum PEEQ changes by about 4.7%; physical edge-point
count is not monotone with quadrature degree. These are discrete-rule
sensitivity data, **not** quadrature convergence or a resolved contact
boundary. Higher degree alone cannot be the default numerical-method fix.
The same fixture at penalty 200, 12 increments, degree-7 quadrature and
`basic` Newton was refined without changing its physical or algorithmic
inputs. Cells 3--8 came from the installed wheel; cells 10--20 were rerun on
the final candidate source:

For each new row, run
`PYTHONPATH=src python examples/finite_j2_closed_concave_contact.py --cells N --increments 12 --penalty 200 --contact-quadrature-degree 7 --line-search basic --output OUTPUT`
with `N=10,12,14,16,20` on the candidate branch; the fourteen-cell face-rule
control substitutes degree 5 and changes nothing else.

| Cells per axis | x-resultant | max PEEQ | max penetration | active reentrant-edge points |
| ---: | ---: | ---: | ---: | ---: |
| 3 | -2.01339621 | 0.03934018 | 0.03311168 | 0 |
| 4 | -1.85720972 | 0.04151851 | 0.03215686 | 0 |
| 5 | -1.84844562 | 0.04281734 | 0.03307059 | 0 |
| 6 | -1.82327209 | 0.04199805 | 0.03414494 | 0 |
| 8 | -1.77276480 | 0.04609411 | 0.03439900 | 1 |
| 10 | -1.76727530 | 0.04587753 | 0.03447152 | 1 |
| 12 | -1.75368614 | 0.04897816 | 0.03454590 | 2 |
| 14 | -1.73441914 | 0.05089190 | 0.03538630 | 2 |
| 16 | -1.73535783 | 0.05478494 | 0.03550203 | 5 |
| 20 | -1.72154804 | 0.05989208 | 0.03524172 | 8 |

From six to eight cells the force magnitude changes by 2.77% and maximum
PEEQ by 9.75%. The ten-cell maximum PEEQ initially falls, then rises by
6.76% from ten to twelve cells and by another 3.91% from twelve to fourteen;
the active reentrant-edge sample count changes as well. The ten- through
twenty-cell rows were rerun on the candidate source at identical
physical and algorithmic inputs; the only core changes since the earlier
installed-wheel eight-cell row are result-axis unit metadata, not numerical
assembly. This extends the non-convergence evidence; it does not identify the
asymptotic solution, separate spatial error from penalty error, or validate
the full-Newton policy for other models.

The 16→20-cell x-resultant changes by about 0.80%, while maximum PEEQ rises
about 9.3% and the sampled reentrant-edge points increase from five to eight.
Apparent force stabilization alone is therefore not a validated damage or
plastic-history limit.

At fourteen cells, changing only the positive triangle rule from degree 7 to
degree 5 gives x-resultant `-1.73819547` instead of `-1.73441914` (about
0.22% difference), maximum PEEQ `0.05169160` instead of `0.05089190` (about
1.57% difference), and one rather than two active reentrant-edge points.
Both runs completed with twelve accepted increments. This narrows, but does
not close, the face-rule uncertainty at that mesh; it is not a contact-patch
or penalty-convergence certificate.
The exported *raw integration-point* PEEQ field gives a useful countercheck
against over-interpreting the single maximum. Using each point's exported
physical-volume weight, the degree-7 volume-weighted means for cells
8/10/12/14 are `0.01639180/0.01665422/0.01665232/0.01655183`;
the corresponding weighted 99th percentiles are
`0.03980527/0.03964740/0.03891682/0.03953172`.
At fourteen cells the degree-5 mean is `0.01661404` and its weighted 99th
percentile is `0.03959502`. The whole-volume distribution is more stable than
the local maximum in this *fixed-penalty* screen, but the changing resultant,
edge sampling and maximum forbid a general contact or hotspot convergence
claim. All four integration-point archives report total volume one; no
smoothed nodal field was used for these statistics.
Backtracking remains the example default.
Any new globalization should compare the accepted path, total residual,
contact work, plastic history and cutback behavior against both policies;
silently falling back to a full Newton step after rejected backtracking is
not justified by this one fixture.
[PETSc SNES](https://petsc.org/release/manual/snes/) offers line-search and
trust-region Newton variants, but adopting one is a Procedure integration
task: every trial must re-evaluate the current contact branch and material
from fixed accepted State, and only a converged increment may commit either.
Its bound-constrained VI solvers are not a drop-in switch for this
displacement-only penalty formulation; they require a different algebraic
constraint/unknown contract. First compare a bounded trust-region prototype
against the existing backtracking and full-Newton paths without changing the
contact law or silently accepting a different plastic trajectory.
The next numerical-method experiment
must distinguish a contact-boundary/feature-aware integration strategy from a
different physical model, preserve residual--tangent consistency and compare
like-for-like force, moment, plastic history and work. The refined-boundary
quadrature and contact-patch discussion of
[Duong, De Lorenzis and Sauer](https://arxiv.org/abs/1712.01179) motivates
that experiment; it is not validation of an implementation here.
The independent comparison of segment-based, element-based and hybrid
integration by [Farah, Popp and Wall](https://doi.org/10.1007/s00466-014-1093-2)
also identifies contact-zone boundaries as an accuracy and cost decision.
It supports testing a local partition, not claiming that the particular
clipped-penalty algorithm here was validated by their mortar study.
The bounded next algorithmic slice is **frictionless adaptive boundary
integration**, not an implicit change of every model's default rule: locate
the zero-gap and nearest-feature transitions on each candidate slave face,
subdivide only crossing faces with positive reference-area weights, and use
the same frozen trial partition for residual and consistent tangent. A changed
partition requires a fresh Newton trial, and an accepted result must record
its integration identity and coverage diagnostics. Validate a uniform planar
patch first, then a manufactured sharp-feature cut against independent
high-resolution surface integration, then force/moment/work and mesh/penalty
sensitivity. Stateful friction must **not** inherit changing point identities
without an explicit State transfer rule. This is a design gate, not code
claimed by the present candidate.

An independent, **research-only** affine P1 triangle oracle now fixes the
smallest contact-boundary problem before changing the Step. For the unit
reference triangle and planar gap `g(x,y)=x-a`, exact clipping of `g<0`
gives active area `a-a²/2` and reference-area penalty potential
`a³/6-a⁴/24` at unit penalty. `tools/verify_contact_boundary_quadrature.py`
computes that clipped potential, nodal residual and fixed-branch tangent with
positive degree-two subtriangle quadrature; nine tests check analytical
area/potential, potential and residual derivatives, nodal-force conservation,
vertex-order invariance, open/full limits and invalid inputs. At `a=0.05`,
full-face Basix degree-two **and degree-five**
rules both return zero potential, while the analytical and clipped values
are `2.057291666666667e-5`. Degree seven captures only about 1.24% of that
value, and degree thirteen overshoots it by about 5.65%. This manufactured
counterexample explains why raising a whole-face quadrature degree is not a
general answer to narrow contact strips. It does **not** certify a production
adaptive contact Operator: the current oracle has a fixed planar gap and no
curved tool, nearest-feature transition, Newton partition identity, friction
State transfer, mesh convergence or global force/work acceptance. None of
the production defaults or public capability claims change on this evidence.

The present tests are local software evidence, not an independent structural
benchmark. [NAFEMS contact benchmarks](https://www.nafems.org/publications/resource_center/r0094/)
motivate same-input comparison; a reference number from a different model
cannot validate this implementation. [Code_Aster SSNP154](https://code-aster.org/V2/doc/v11/en/man_v/v6/v6.03.154.pdf)
publishes an Abaqus/MARC contact-friction comparison for a **2D plane-strain**
roller and distinguishes extrapolated bulk stress from contact-multiplier
pressure, including mesh-dependent oscillations. It is useful for designing
the future independent benchmark and choosing like-for-like observables,
but its reported displacement/pressure numbers are not a reference for our
3D finite-J2, reference-area penalty fixture. Abaqus' own
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

An isolated native Mac wheel from `f34a926d` has SHA-256
`a2a56e7ccfcb9c024ddc385342207eeaa55ac4aa07b6671414c3e77aef38f873`.
Its installed package path, not the checkout, supplied the public affine-J2
`model.step()` runtime for five moving-friction, reduced-Jacobian and restart
regressions; all five passed. `_hex_native` and `_body_pipeline` imported from
the same isolated installation. This verifies packaging of the bounded serial
entry point, not other operating systems, MPI contact, or full-scale accuracy.
The same wheel ran the standalone public example at six increments with an
active reentrant edge and wrote `fields.xdmf`, its HDF5 payloads and
`result.json`: maximum PEEQ `0.004630797762550675`, tool work
`0.010730556638046358`. Its own result scope remains
`serial_manufactured_fixture_not_external_validation`.
The `2c8be39f` source archive has SHA-256
`b8e666a2edd52e3dbdb3bae16902f4a1e630746b08802da024aef639d839c6df`.
The platform wheel built from that archive has SHA-256
`47fe7c08f87971bfac44e0af2c9e1f3037f78f017aefda610b9b0a3fcbfbd6d8`
and includes `_hex_native`, `_body_pipeline`, and `_q1_native`. Installed in
an isolated environment outside the checkout, its `python -I` import resolved
to that environment's site-packages, and the same six-increment public example
completed with one active reentrant-edge quadrature point, maximum PEEQ
`0.004630797762550675`, tool work `0.010730556638046358`, and
`fields.xdmf`, both HDF5 payloads, and `result.json`. This closes the
`2c8be39f` **Mac installed-package smoke**, not independent physical validation
or cross-platform numerical acceptance. Three selected public-Step tests run
under the isolated interpreter's `-I` mode also passed: opt-in tetrahedral
quadrature with restart identity, globally active reentrant-edge J2 contact,
and two moving tools with atomic restart.
The subsequent `cfec462e` source archive has SHA-256
`1869fc3ef8abcab4c1b8d3dffd462b4ce64bae55d574d68fd4792f10efc234ff`;
its native Mac wheel has SHA-256
`ec6a532a51b22f15eb9a8ed976765f6a8bb8abee86b267a2aecbf649b10fcb1c`.
The wheel was installed in a new isolated environment, all three native
modules imported from its site-packages, and three public-Step tests passed
under `python -I`, including the example's default/degree-7 face-rule
exports. The non-MPI installed-wheel `release_gate.py --dist ... --smoke`
completed, including its representative workflows and project templates;
the report is retained at
`/private/tmp/agentfem-contact-release-smoke-cfec462e.json` on this host.
This is **Mac package/workflow acceptance**; it does not complete the
independent contact physics, cross-platform or distributed implicit gates.
The source-side serial checkpoint policy also passed `every=1, keep_last=1`:
only the final v2 artifact remained and its accepted Coulomb history restored.
`portable=True` is correctly rejected at preflight for an external contact
Operator; a serial v2 file must not be described as MPI-rank portable.
An opposed active frictionless sphere plus moving frictional plane also kept
separate pair States through a v2 restart and reproduced the final field and
accepted sliding dissipation. This is a two-tool composition check, not a
general multi-body or distributed-contact benchmark.
An additional affine-periodic check places an active reentrant edge opposite
an active analytical sphere. Both contact duals remain named separately from
the periodic dual, the total material-plus-both-contacts residual reduces to
zero, and a serial checkpoint reproduces displacement and committed J2 State.
It does not establish convergence across an edge transition or distributed
multi-tool assembly.
Changing the Coulomb coefficient invalidates the scientific restart identity;
corrupting the second pair's checkpoint data rejects the entire load before
the first pair's friction State can remain half-restored. An injected late
second-pair commit failure likewise restores the previous material, field,
pose and friction boundary.
The nonsymmetric affine Coulomb tangent rejects CG at construction; a serial
GMRES/ILU solve matches direct LU in displacement, accepted friction
dissipation and rigid-tool path work on the same fixed-input Step.
The public affine friction `solve_result(output=..., strict_output=True)`
path also writes XDMF/HDF5 field artifacts while retaining separate raw
contact-point diagnostics; this is a final accepted field view, not a
time-history export or a first-law verification.
The affine Procedure now checks an external Operator's in-memory State
snapshot/restore pair before allowing `commit_accepted`, and checks its durable
accepted snapshot/restore pair before checkpoint publication. A deliberately
incomplete provider fails before numerical work or an output artifact, rather
than producing an unrollable accepted increment.

The next sharp-feature friction gate is not a normal-averaging patch.
[Abaqus' contact overview](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactoverview.htm)
describes distinct finite-sliding surface, edge and vertex formulations,
while its [contact-controls guide](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm)
distinguishes force redistribution smoothness at sliding transitions. We
therefore retain the current fail-closed physical-edge friction boundary until
an explicit transition law, generalized tangent, force/work audit and
cross-feature convergence reference are available. Smoothing a geometric
normal alone would change the model without establishing those contracts.

### Sign-query performance experiment, not a whole-Step speed claim

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

An exact, cheaper prefilter now classifies a query as exterior only when its
reference-frame point lies **strictly beyond the complete tool AABB with a
tolerance margin**. It still computes the current nearest point and normal;
inside the AABB, direct winding and ambiguous-sign rejection remain mandatory.
Fixed and moving-tool tests verify identical signed gaps for certified
exterior points. A frozen-wheel versus candidate-source local comparison on
this Mac, 5,120 facets, 5,000 fixed-seed queries and three medians yielded:

| Query distribution | Pre-shortcut full projection | Exact-AABB shortcut full projection |
| --- | ---: | ---: |
| Half strictly outside the tool AABB | 2.085 s | 1.334 s |
| All within the AABB | 2.063 s | 2.092 s |

The first row is about 36% faster for **that local geometry query**, not a
whole Newton step or application. The second row is a control showing no
useful gain when the certificate cannot apply (the small difference is timing
noise). This safe screen does not replace the unpromoted experimental ray
classifier.

The angle-weighted pseudonormal theorem offers a different route: its sign
can be computed from the same nearest feature already found by the BVH.
`sign_method="angle_weighted_pseudonormal"` is an **explicit opt-in** on the
reviewed closed solid; direct winding stays default. The option preserves
identical signed gaps and final material/tool-work responses in the 2,000
fixed-seed concave/reentrant-vertex queries per fixture and the six-increment
active-edge finite-J2 Step. A 5,120-facet, 5,000-query, three-median local
run on this Mac measured default full projection 2.062 s versus opt-in
0.584 s (3.53× **geometry-query** speedup). These are fixed-input local
measurements, not a whole accepted Newton-Step or cross-machine claim.
Neither option certifies the input mesh is free of self-intersection, but
direct winding retains stronger query-time evidence on malformed geometry.
An additional **whole-Step** screen, including finite-J2 material updates,
Newton and contact but excluding export, used the same 5,120-facet tool and
24 accepted increments. Three in-process repetitions on this Mac gave:

| FE cells per axis | Contact-active points | Default winding Step | Opt-in pseudonormal Step | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 1.323 s | 0.684 s | 1.94× |
| 2 | 10 | 3.719 s | 1.567 s | 2.37× |
| 3 | 22 | 8.169 s | 3.278 s | 2.49× |

`tools/benchmark_closed_j2_step.py` recreates the model for each repetition,
times `solve_result()` only, and rejects any difference in final displacement,
material state, accepted count, active count or tool work. This is evidence
that the local sign cost can matter at Step level on this manufactured
contact-heavy problem, not a general 2× promise: a real tool, larger FE
mesh, different hardware or different contact density can change the ratio.
The three-cell row was run later with the same 5,120-facet tool, 24 accepted
increments and three repetitions; its final tool work was 0.008853793023137713
for both modes. A 1,280-facet control at that FE size measured 4.482 versus
3.090 seconds (1.45×), showing the benefit depends on tool facet count.
After the affine result-diagnostic extension, the same three-cell source-side
screen repeated with 5,120 facets and three medians at 8.241 versus 3.292
seconds (2.50×); both modes again gave tool work
`0.008853793023137713`. The small absolute-time drift from the prior run is
not evidence of a portable regression or speed gain.
The fast option's numerical reproducibility on a large independent model
and malformed-mesh robustness remain open release gates.

Open work before broader admission: independent same-input force/displacement
comparison; controlled geometry, mesh, penalty and increment sensitivity;
whole-Step speed and malformed-mesh robustness of the opt-in sign path;
distributed ownership and
cross-rank restart. Sharp-feature friction, deformable master, self-contact and arbitrary
unreviewed STL remain unsupported. Do not promote this branch into a release
merely because a small Newton solve completes.

## Nine-gate local-parity map

This candidate advances the previously agreed *bounded* contact roadmap; the
gates are not interchangeable. Status refers to demonstrated evidence, not
planned code:

| Gate | Current evidence | Remaining boundary |
| --- | --- | --- |
| 1. Same-input benchmark contract | The manufactured L-prism fixture and outputs are frozen in test/example code. | No independent Abaqus/Code_Aster same-input result or agreed dimensional acceptance tolerance. |
| 2. Existing candidate audit | Ordinary Step, tool work, restart and current-trial geometry are reused; no parallel solver path was added. | Broader scientific promotion review remains separate. |
| 3. Geometry and feature transitions | Unique nearest face/edge/vertex branches, coplanar seams, non-manifold vertex rejection and a globally active reentrant edge pass. | Self-intersection is caller-attested; medial-axis ties reject; no arbitrary CAD guarantee. |
| 4. Implicit total operator and State | Full J2-plus-contact tangent passes fixed-old-state difference; an active sharp edge passes periodic MPC total-residual, reduced contact tangent and fixed-old-state material-plus-contact reduced Jacobian differences; accepted failure rolls back. Active-edge translation/rotation has an independently integrated force/moment and a matching fixed-displacement work derivative. | This remains a manufactured serial fixture; feature-switch and larger mesh convergence are separate gates. |
| 5. Multiple rigid pairs | Named planar/convex tool work and atomic restart tests pass; mixed active concave-plus-moving-plane J2 pair keeps separate work, summed tangent and checkpoint restart. An active reentrant edge opposite an active sphere also passes periodic total-residual reduction and serial J2 restart. | Large mixed contact-set scaling and independent multi-tool reference remain untested. |
| 6. Friction | A finite-J2 strong-BC Step with a reviewed closed concave solid on a planar face matches the convex limit in displacement and friction dissipation for translated and slightly rotated tools under both sign modes; the fixed-old-state tangent and interrupted restart pass. The serial affine-periodic J2 path now commits a moving-plane Coulomb State, reduces its contact tangent by `TᵀKT`, checkpoints it in v2 and atomically rejects corrupt State; reviewed triangle and closed-concave planar faces match the same plane limit. An active reentrant edge rejects without partial State. | The planar law does not include changing-normal terms at a physical edge or vertex; general sharp-feature friction, distributed affine contact and a full thermodynamic path audit remain unsupported. |
| 7. Material/time composition | Active-edge finite-J2 plastic path passes; explicit contact/material lifecycle is a separate existing route. | No generic historical implicit material or explicit general-closed-tool composition. |
| 8. Independent acceptance and speed | Directional analytical checks, fixed-seed geometry timing and a manufactured whole-Step opt-in/default comparison are recorded; 24→48 increment screen is narrow. A fixed-geometry mesh/face-rule screen exposes nonmonotone active-feature sampling. The six-cell 10/20/40 penalty bracket confirms large penetration and changing force/plastic history; at penalty 200 and full Newton, the degree-7 mesh screen through twenty cells still changes active-edge sampling and maximum PEEQ. The independent clipped-face oracle proves that ordinary whole-face rules can miss a narrow active contact strip entirely. | No independent full-geometry comparison, sharp-feature spatial/penalty convergence, industrial whole-step speed claim or release-grade energy closure. |
| 9. Product boundary | Public-API example exports XDMF/manifest from an isolated native Mac wheel; the pre-metadata numerical candidate passed 177 installed-wheel serial tests (8 MPI-only skips). Source archive and native Mac wheel built from numerical source `5ff2aa0b` pass isolated non-MPI installed-wheel release smoke. Unsupported MPI/friction preflight rejects. | Linux/Windows installed numerical acceptance and final remote CI remain version gates; a local Mac smoke is not cross-platform release approval. |

The table intentionally refuses to turn one successful six- or 24-increment
fixture into an Abaqus General Contact equivalence claim.

### Trial active-boundary integration (bounded source candidate)

The first production-path slice of the boundary-integration repair is an
explicit `contact_integration="planar_active_partition"` option.  It is admitted
only for serial, frictionless, scalar-penalty analytical-plane contact on an affine CG1
tetrahedral slave triangle with the default positive degree-two rule.  At each
Newton trial the current planar gap is evaluated at the three slave vertices;
only a sign-crossing face is clipped in *reference* barycentric coordinates.
Positive degree-two quadrature on each active subtriangle then uses the same
trial trace for the contact residual, tangent, force, moment, potential and
accepted-point output.  Open/full faces retain their ordinary trace.  The
friction State and its stable point IDs are deliberately excluded.  The
accepted checkpoint identity includes this selected integration mode.

The elementary narrow-strip oracle, `g=x-a` on a unit triangle, has exact
active area `a-a²/2` and potential `k(a³/6-a⁴/24)`; the new path matches it
through widths `0.0001–0.2`.  A real DOLFINx unit-cube boundary test with
`a=0.05` catches contact that whole-face quadrature reports as zero, and its
finite-difference tangent agrees with the moving trial partition.  A small
ordinary finite-J2 Step and checkpoint rule-identity test pass.  This repairs
one reproducible blind spot, not the L-tool or hard-contact convergence gate.
Curved tools, nearest-feature transitions, Q1/CG2 faces, friction and MPI remain
outside this method; do not infer general coverage from the affine proof.

`tools/verify_contact_feature_transition.py` records why the next slice is
not a mechanical reuse of the zero-gap clipper.  For the active two-plane gap
`g=|x-c|-a` on a unit reference triangle, moving the sharp feature leaves
the exact integrated tangent zero, while a fixed-branch bulk tangent alone is
`2ka(1-c)`.  An equal negative line contribution at the still-compressed
feature switch is necessary.  At the zero-gap boundary that line term instead
vanishes.  A sharp-tool partition must therefore integrate **both** positive
area subcells and the consistent feature-interface tangent, or explicitly use
and validate a different nonlinear method; simply cutting at the feature and
reusing the planar tangent would claim a false Jacobian.

`tools/verify_two_plane_contact_integration.py` extends that scalar
counterexample to the full three-node vector residual and Hessian on a
reference triangle with two affine competing plane gaps. It partitions the
negative-gap area by the dominant feature, integrates each subpolygon with
positive degree-two quadrature, and adds a separate Gauss line integral on
the compressed feature switch. The line term is proportional to the common
negative gap and the outer product of the two normal jumps. It vanishes on
the zero-gap contour but not on an interior compressed feature transition.
Independent analytic potential/translation derivatives, directional finite
differences, symmetry, feature-order and vertex-order checks pass. A
manufactured nine-DOF spring-supported triangle reaches a global stationary
state with this full tangent; its energy and nodal force match separately
split one-dimensional surface integration, and its assembled Jacobian matches
a directional difference. A further manufactured one-tetrahedron *linear*
elasticity fixture reaches equilibrium with the same contact integral. Its
hand-derived bulk stiffness agrees with an independent DOLFINx UFL/PETSc
assembly. These are not finite-J2, reviewed L-prism or external-solver
comparisons. The oracle rejects a nonunique boundary-aligned feature switch
rather than claiming a two-sided tangent. It remains an independent
**research-only two-plane reference**, separate from the bounded convex Step
mode below: the reviewed L-prism's nearest-edge and nearest-vertex distances
are nonlinear, and its full feature graph, coverage, friction State and
global convergence remain unproved.

### Certified convex bounded-feature trial integration

The next bounded production-path slice is an **opt-in**
`contact_integration="convex_feature_partition"` for frictionless scalar-penalty
contact against an already certified closed-convex tool. It supports affine
CG1 tetrahedral slave triangles with the default reference-area rule and one
MPI rank. At each current trial the tool's distinct support-plane gaps are
evaluated at the three slave vertices. Each positive-area dominant plane
receives a polygonal active subcell and positive degree-two quadrature; the
sum of those areas is checked against an independently clipped convex active
region. A face with more than three positive-area dominant planes rejects
instead of guessing a subset. Each pair of active planes switching across a
compressed dominant line contributes the affine feature term to the same
contact Newton matrix. An independent symmetric convex-corner fixture checks
exact potential and total force as well as five directional tangent differences
with three positive-area dominant planes. Five perturbed corner states also
pass directional tangent differences. Four such planes still reject.
A boundary-aligned undefined
two-sided switch also rejects. The default Gauss route is unchanged.

An axis-aligned convex box with a narrow interior slab gives the independent
two-plane oracle's potential, nodal force and **full** tangent. A real DOLFINx
boundary Operator catches a strip missed by whole-face Gauss and its
directional tangent matches the moving trial residual. A two-increment
ordinary finite-J2 Step has an active feature line; its full material-plus-
contact PETSc matrix matches a fixed-old-state residual difference. The same
option composes with the serial affine-periodic J2 route: the full
material-plus-contact residual is reduced by the existing MPC and its total
reduced Jacobian passes a fixed-old-state directional difference. Prescribed
rotation, force--moment work derivative, and interrupted serial checkpoint
match their continuous controls. The fixed tool's distinct support planes are
prepared once per Operator, while the current trial partition and projection
are recomputed; a disagreement between their selected active features rejects
the trial. The chosen integration enters checkpoint identity and exported point
IDs are accepted-snapshot-local. These are bounded
software/numerical tests: they do not establish general convex-tool
multi-feature coverage, nonconvex L-prism accuracy, spatial/penalty
convergence, sharp-edge friction or Abaqus parity.

Methodological context: Farah, Popp & Wall (2015),
`doi:10.1007/s00466-014-1093-2`, compares segment- and element-based mortar
integration; Duong, De Lorenzis & Sauer (2017), `arXiv:1712.01179`, discusses
refined boundary quadrature for more general contact.  Neither paper validates
the present implementation, whose scope is narrower and independently tested.

The committed `ede47723` two-feature source candidate passed 212 targeted real-FEniCSx
serial regressions, strict documentation build and targeted Ruff checks. Its
native Apple Silicon wheel has SHA-256
`be5871036ca1b23cf0942ecdc79244d8bd8071554b107448dcf06225c7f16bbb`;
the matching source archive has SHA-256
`7d1789b88549ecb7924b8af1b1739c19a5c23043b21ef1ba99c3db8a1bf7a197`.
An isolated `python -I` import resolved to the installed wheel, its native
Hex/body modules imported, the new example completed two increments with two
active feature lines and emitted XDMF, quadrature payloads, contact samples
and a result manifest. All 13 new-path tests also passed from the isolated
wheel, without source-path injection. The existing non-MPI `release_gate.py`
installed-wheel smoke passed with all four project templates; its report is
`/private/tmp/agentfem-feature-release-smoke-ede47723.json` on this host.
These are package and local workflow checks, not Linux/Windows evidence,
external solver parity, a converged sharp-feature solution or permission to
promote this experiment as a stable release.

The subsequent three-feature extension passed 183 focused real-FEniCSx serial
tests spanning the contact Operator, finite-J2 Step, periodic MPC, work and
existing contact modes, along with targeted Ruff and a strict docs build.
Its installed-package evidence must be regenerated from its exact commit;
the hashes above identify only `ede47723`, not this extension.
