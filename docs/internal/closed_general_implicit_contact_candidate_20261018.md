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
| 4. Implicit total operator and State | Full J2-plus-contact tangent passes fixed-old-state difference; an active sharp edge now passes periodic MPC total-residual, reduced contact tangent, and fixed-old-state material-plus-contact reduced Jacobian differences; accepted failure rolls back. | This remains a manufactured serial fixture; cross-branch and larger mesh convergence are separate gates. |
| 5. Multiple rigid pairs | Named planar/convex tool work and atomic restart tests pass; mixed active concave-plus-moving-plane J2 pair keeps separate work, summed tangent and checkpoint restart. | Large mixed contact-set scaling and independent multi-tool reference remain untested. |
| 6. Friction | A finite-J2 strong-BC Step with a reviewed closed concave solid on a planar face matches the convex limit in displacement and friction dissipation for translated and slightly rotated tools under both sign modes; the fixed-old-state tangent and interrupted restart pass. The serial affine-periodic J2 path now commits a moving-plane Coulomb State, reduces its contact tangent by `TᵀKT`, checkpoints it in v2 and atomically rejects corrupt State; reviewed triangle and closed-concave planar faces match the same plane limit. An active reentrant edge rejects without partial State. | The planar law does not include changing-normal terms at a physical edge or vertex; general sharp-feature friction, distributed affine contact and a full thermodynamic path audit remain unsupported. |
| 7. Material/time composition | Active-edge finite-J2 plastic path passes; explicit contact/material lifecycle is a separate existing route. | No generic historical implicit material or explicit general-closed-tool composition. |
| 8. Independent acceptance and speed | Directional analytical checks, fixed-seed geometry timing and a manufactured whole-Step opt-in/default comparison are recorded; 24→48 increment screen is narrow. | No independent full-geometry comparison, spatial/penalty convergence, industrial whole-step speed claim or release-grade energy closure. |
| 9. Product boundary | Public-API example exports XDMF/manifest from an isolated native Mac wheel; unsupported MPI/friction preflight rejects. | Linux/Windows installed acceptance and remote final-head CI remain version gates. |

The table intentionally refuses to turn one successful six- or 24-increment
fixture into an Abaqus General Contact equivalence claim.
