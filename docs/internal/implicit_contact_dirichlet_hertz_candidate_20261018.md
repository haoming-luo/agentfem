# Implicit contact on prescribed DOFs and sphere-indentation screen

Status: experimental source candidate, not a Hertz validation or a general
contact promotion. The source branch follows merged PR #116.

## Numerical correction

The serial finite-J2 rigid-pair route previously rejected contact trace DOFs
shared with strong Dirichlet constraints. It now lifts the native contact
tangent by the same prescribed correction `g_bc - u_trial` used by DOLFINx for
the UFL material tangent. Native COO entries on constrained rows and columns
are omitted after the material matrix has had its strong rows/columns set by
DOLFINx. This preserves the total-operator action on free rows and the
identity rows on constrained DOFs. It does not change the frictionless
penalty law, rigid projection or material algorithm.

One active, tilted triangular-tool/J2 test imposes a nonzero displacement on
the contact face, completes, and checks every shared strong row remains an
identity row. Another fixed-old-state directional test compares the assembled
total J2-plus-contact Newton tangent against a central difference of the
complete correction residual, with the same trial material state and zero
perturbations on constrained DOFs. A separate sphere-indentation test compares a full block with
the symmetry-reduced quarter block at identical physical resolution: 256
versus 64 hexahedra give the same full-body force to better than `1e-9`
relative difference, and four times the quarter contact potential equals
the full-block potential. These are structural equivalence checks, not
independent physical validation. Existing contact/MPC regression remains
required.

## Independent reference screen

`examples/finite_j2_hertz_contact.py` solves a prescribed rigid-sphere
indentation of a finite, bottom-clamped block. Its very high yield threshold
keeps the finite-strain J2 material in its elastic regime. The report keeps
the model separate from the classical small-strain, elastic-half-space,
exact-nonpenetration Hertz limit
`F = 4 E sqrt(R) delta^(3/2) / (3 (1-nu^2))`.
The [published sphere/half-space comparison](https://www.sciencedirect.com/science/article/pii/S0301679X15006027)
states that reference relation and its domain; it does not certify the finite
block or penalty contact used here.

Commands use the current source checkout and the reviewed FEniCSx runtime:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/finite_j2_hertz_contact.py --quarter --cells 8
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/finite_j2_hertz_contact.py --quarter --cells 8 --increments 16
```

At `E=1000`, `nu=0.3`, `R=2`, indentation `0.02`, penalty `1e5`, four
increments and unit half-width/depth, the half-space reference force is
`5.86080586`. Full-block mesh levels with 256, 2048, 6912 and 16384
hexahedra produce forces `7.99194`, `7.32959`, `7.16618`, and `7.09758`.
The finest screened ratio is still `1.2110`, so this is **not** a passed
Hertz accuracy gate. Finite boundaries, contact quadrature, mesh resolution
and finite penalty remain confounded. A deeper quarter block (half-width and
depth `2`, 4096 hexahedra) gives `6.65969` but changes both boundary distance
and resolution, so it cannot independently identify the discrepancy.

At the same extent-3, 13,824-cell quarter-block resolution, separable
grading `2.5` reduces the top-center reference edge spacing from `0.125`
to `0.02945` without changing the cell count. The active quadrature count
increases from `8` to `102`, the maximum sampled active radius is `0.20194`
versus the Hertz radius `0.2`, and the full-body force changes from
`6.45723` to `6.20730` (ratio to the half-space limit `1.1018` to `1.0591`).
This is evidence that under-resolving the contact patch materially biases
the earlier comparison. It is **not** a pass: the mesh map changes the
discrete problem, the result still differs by `5.9%`, and penalty/domain
limits remain unseparated. The peak value `69.737` is nominal force per
reference area at a sampled quadrature point, not a smoothed nodal pressure.
The Hertz peak for these inputs is about `69.96`, so the sampled center peak
is close while the integrated force is still high; pressure shape and
reference-area activity therefore matter. The active reference area
`0.13521` exceeds the Hertz circle area `0.12566`; neither a matching peak
nor a near-matching sampled radius establishes pressure-profile agreement.
At fixed extent `3`, grading `2.5` and penalty `1e5`, a 4,096-cell quarter
block gives full force `6.25502`, versus `6.20730` for 13,824 cells; active
quadrature points rise from `43` to `102`. At the 4,096-cell level, raising
the penalty to `2e5` changes full force to `6.34303` and maximum penetration
from `0.0007015` to `0.0003581`. Thus the penalty limit does **not** explain
away the positive force mismatch, and the spatial sequence is not yet flat.

At 512 quarter-block cells, penalties `5e4`, `1e5`, `2e5` give full-body forces
`7.10580`, `7.32959`, `7.38030` and maximum penetrations `0.0015424`,
`0.0010021`, `0.00069635`. At fixed mesh/penalty, 4, 8 and 16 increments
leave the final force unchanged within the nonlinear tolerance. Their
accepted trapezoidal work residual declines `1.8809e-4`, `5.8245e-5`,
`2.0718e-5`; this is a path-discretization diagnostic, not a general first
law for plastic contact.

### Linear-solver screen, not a new global default

The example now accepts an explicit `--linear-policy` so the *same* model can
compare the existing direct LU route with PETSc GMRES/GAMG or GMRES/ILU. On
this Apple Silicon FEniCSx environment, the 4,096-cell, extent-3 graded
quarter model takes `32.84 s` with direct LU and `11.65 s` with GMRES/GAMG;
the full-body force differs by about `1.3e-11` and both complete 17 Newton
linear solves. Accepted linear-solve phases are `22.43 s` versus `1.52 s`;
material update and line search are not materially accelerated. The 13,824-
cell GMRES/GAMG case completes in `48.81 s` with full force `6.20730`,
matching the earlier direct-route value at reported precision. All its 21
linear solves have positive convergence reasons. The 32,768-cell GMRES/GAMG
case completes in `113.34 s` with full force `6.18308`, active reference
area `0.13663` and peak nominal pressure `69.65`; all 20 linear solves
converge. Its force ratio to Hertz is still `1.0550`, so the independent
accuracy gate is **not** passed by mesh refinement alone. A 32,768-cell
direct-LU screen was stopped after 20 minutes while PETSc was in numeric LU
factorization; it produced **no** scientific result and is not a measured
direct-versus-iterative speed ratio. These timings are case/hardware evidence,
not a portable choice for every contact model.

One far-boundary screen increases the quarter-block half-width/depth from `3`
to `4`, retaining 13,824 cells and adjusting the separable grading from `2.5`
to `3.0` so the top-center spacing remains of similar order (`0.02945` versus
`0.02791`). With GMRES/GAMG it completes in `47.60 s`, gives force `6.12050`
(Hertz ratio `1.04431`), active area `0.13266`, and 102 active quadrature
points. The decline from `6.20730` is consistent with finite-domain influence,
but spacing and element distortion are not exactly identical; this is **not**
an isolated domain-size convergence or a certified Hertz result.

## Remaining gate

Run independently controlled mesh **and** domain-size refinement with a
resolved contact patch, then penalty and increment sensitivity. Compare
force, pressure footprint, penetration, reactions, tool work and plastic
state against either the valid Hertz limit or a frozen external FE result.
Do not turn the current force ratio into a release tolerance. Active
noncoplanar triangle seams/free edges, multiple implicit pairs, implicit
friction and distributed implicit assembly remain separate work.

The benchmark now also reports reference-area activity, maximum sampled
contact radius, maximum nominal penalty pressure, and the actual near-contact
mesh spacing. An optional separable coordinate grading resolves the patch
without uniformly refining the remote block; it changes only the example
mesh, never the solver's geometry or quadrature. The independent force limit
must be judged together with these spatial diagnostics, rather than as a
single scalar. [COMSOL's contact-fatigue example](https://www.comsol.com/blogs/how-to-model-contact-fatigue-in-comsol-multiphysics)
likewise identifies the contact zone and nearby subsurface as requiring much
finer resolution than the remote structure.

## Local-parity acceptance map

The deliberately bounded target is 3D solid versus prescribed rigid tools,
not general contact. This record separates code gates from evidence gates:

| Gate | Current evidence | Outstanding acceptance |
| --- | --- | --- |
| Comparable problem | Finite-block sphere setup is fixed and reproducible in source | Freeze an independently run Abaqus deck with the *same* block, law, mesh, boundary path and penalty; no numerical Abaqus reference has been supplied |
| Existing contact composition | Merged PR #116 has reviewed smooth/facet rigid geometry, affine MPC, moving-tool work and rollback; serial focused regressions pass | Installed-package and post-merge release-gate run for the new changes |
| Facet features | Reviewed topology identifies true free edges and noncoplanar neighbors; ambiguous active projection fails closed | Feature-specific edge/vertex contact potential, generalized derivative, branch switching and force/moment tests; no normal averaging |
| Total implicit Operator | Contact/strong-BC overlap now uses native contact lifting, identity rows, a complete fixed-old-state total-tangent difference check, failed-increment rollback and same-DOF checkpoint recovery | Add an independently constructed residual/force check and expand beyond the current serial frictionless discretization |
| Multi-tool implicit | Ordinary serial strong-BC J2 now composes distinct frictionless rigid pairs: summed residual/tangent, elastic-regime total derivative, independent moving-pair work and atomic failed-increment/checkpoint recovery. An opposing moving-plane fixture checks both tool forces, support-inclusive force balance and result records | Mixed curved/faceted tools, plastic branch switching, external structural comparison and finer mesh/penalty checks; affine MPC still admits one pair only |
| Implicit friction | Not admitted | Stick/slip internal state, consistent linearization, dissipation, rollback and independent path benchmarks |
| Material/time composition | Elastic-regime J2 sphere and plastic J2 facet cases complete; explicit body contact has its own stable-step contract | A history material on the *same implicit* seam and explicit contact stability under evolving state |
| Independent accuracy and speed | Full/quarter symmetry, mesh, penalty and increment screens exist; half-space mismatch remains visible | Controlled far-boundary, graded mesh, pressure-footprint and penalty limits; external FE comparison and whole-step timings |
| Product route | Ordinary `model.step()` is used; final unsmoothed reference-trace pressure/penetration scalars are now available | Installed-wheel example, pressure/slip field format, capability matrix and versioned release gate |

The [Abaqus/Standard general-contact description](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm)
uses separate surface-, edge- and vertex-contact formulations. This supports
treating sharp features as a separate numerical law, not treating the current
selected-facet normal as a differentiable smooth surface. No row in this map
is promoted merely because a single visual plot or Newton solve looks plausible.

## Next implementation decisions

The immediate implicit friction route should reuse the accepted, stable-ID
`TangentialContactState` and return map already used by explicit contact. It
must add a *fixed-old-state* trial tangent, first on a planar rigid branch:
stick has tangential tangent `k_t (I - n n^T)`; slip also differentiates the
pressure cap `mu p` and slip direction. The opening, stick/slip switching and
zero-slip branches require directional checks and cutback. Extending the same
formula to curved or faceted moving tools additionally differentiates normal
transport and master-point motion; those terms cannot be silently dropped.
The private `implicit_friction.py` candidate now implements that fixed-plane
point tangent and its reference-trace COO mapping. Six solver-neutral tests
compare stick and slip directional differences on axis-aligned and rotated
planes, and reject stale trials and the nonsmooth cap/opening branches. It
has **no** accepted state owner or global Step admission; frictional implicit
`model.step()` still reports the missing lifecycle gate.
The sliding tangent includes a pressure-cap cross term and is generally
non-symmetric. Public admission must therefore reject symmetry-only linear
solver selections (for example CG) or select a reviewed nonsymmetric route;
the current default direct solve does not by itself validate the friction
Procedure.
The [FEBio frictional-contact formulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/)
uses an explicit stick/slip return map and verifies patch, Hertz and
large-sliding cases; its numerical results are not AgentFEM evidence.

For multiple rigid tools, the ordinary serial strong-BC J2 Step now adds
independently named frictionless pair Operators. Each pair retains its own
projection identity and prescribed pose, while one owning Procedure accepts or
rolls back the increment. `SimulationResult` carries separate duals, nodal
reaction distributions, quadrature diagnostics and accepted cumulative tool
work. The full-space contact tangent sum is compared with a central
difference in an elastic-regime J2 fixture. Two independently moving tools
also pass interrupted/restarted path equivalence; malformed per-tool work is
rejected on checkpoint load. A separate opposed planar-die test checks that
both tools remain active, force directions oppose, support-inclusive force
balance closes and result metadata retains both identities. This is **not**
yet evidence for arbitrary shaped dies, mixed sharp/curved branch switches, implicit friction, affine
MPC multi-pair reduction or MPI.

For a truly sharp noncoplanar tool edge, a bounded feature-contact law needs
face/edge/vertex classification plus an admissible-side rule and a
semismooth/generalized derivative. A declared physical fillet instead uses
the smooth-surface path and must keep its radius in the model identity. The
FEBio paper reports that a slight radius was needed to solve one sharp-corner
example; this is motivation for an explicit geometry choice, not permission
to round an imported mesh behind the user's back.

For an *ideal* noncoplanar sharp edge, one selected-facet signed plane gap
does not have a continuous normal derivative across the edge at finite
penetration. Requiring a continuously rotating force there **without** a
declared fillet or a feature-specific contact potential would silently
change the geometry or the law. The proper sharp-feature gate is: identify
face/edge/vertex ownership, define one potential with the correct admissible
side, check potential continuity and one-sided or generalized residual
derivatives on each branch, then verify stable branch switching and work.
For a physically rounded die, continuity should instead be tested against
the declared smooth radius. The current fail-closed error is preferable to
choosing an arbitrary adjacent facet normal, but it is not the finished
feature-contact algorithm.
