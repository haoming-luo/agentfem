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
The [Abaqus analytical Hertz relation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-demanalysis.htm)
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
At extent `5`, 13,824 cells and grading `3.35`, the top-center spacing is
`0.02723`; the force is `6.07176` (ratio `1.03599`), active area `0.13113`,
and 100 quadrature points are active. This strengthens the diagnosis that
the earlier small finite block elevated the force, while still leaving a
`3.6%` gap and confounded mesh-map/penalty/finite-strain effects.

An additional *fixed extent-8 and grading-4* spatial sequence with penalty
`1e6` and four increments gives the following source-branch diagnostics:

| Quarter cells | Top-center spacing | Active points | Full force | Hertz ratio | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.04239 | 32 | 6.28701 | 1.07272 | 15.98 s |
| 13,824 | 0.02707 | 86 | 6.16946 | 1.05266 | 56.07 s |
| 32,768 | 0.01987 | 156 | 6.11118 | 1.04272 | 143.23 s |

The trend is monotone under this mesh map, and the last sampled active
radius is `0.19840` versus the Hertz `0.2`. However, the force still differs
by `4.27%`; the active reference area is `0.12044` versus the Hertz circle's
`0.12566`. This sequence does not separate penalty, far-boundary and
finite-strain errors. It is therefore diagnostic evidence, **not** an
independent pass or a solver speedup claim. These runs used the source branch
with one process and PETSc GMRES/GAMG, not an installed release.

## Remaining gate

Run independently controlled mesh **and** domain-size refinement with a
resolved contact patch, then penalty and increment sensitivity. Compare
force, pressure footprint, penetration, reactions, tool work and plastic
state against either the valid Hertz limit or a frozen external FE result.
Do not turn the current force ratio into a release tolerance. Active
noncoplanar triangle seams/free edges, distributed implicit assembly and
broader friction remain separate work; the bounded serial multi-tool and
fixed-plane friction candidates below do not close those general gates.

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
| Facet features | Open/nonconvex triangle tools still reject active noncoplanar ambiguity. A separately certified closed-convex tool now has a continuous interior support-plane gap, one deterministic limiting derivative at a crease, and an elastic/plastic finite-J2 feature-switch path with restart | General nonconvex edge/vertex laws, external sharp-feature comparison, scalable search and mesh/penalty convergence; no normal averaging |
| Total implicit Operator | Contact/strong-BC overlap now uses native contact lifting, identity rows, a complete fixed-old-state total-tangent difference check, failed-increment rollback and same-DOF checkpoint recovery | Add an independently constructed residual/force check and expand beyond the current serial frictionless discretization |
| Multi-tool implicit | Ordinary serial strong-BC J2 composes distinct rigid pairs: summed residual/tangent, independent moving-pair work, and atomic failed-increment/checkpoint recovery. Opposing moving frictionless planes check force balance. Two fixed frictional planes recover the equivalent single-law response while preserving separate State/dual/checkpoint identities; spatially opposed fixed frictional planes also check both force resultants and interrupted restart. Failure of the second commit rolls both back. The separate affine-periodic J2 route now admits named frictionless pairs: equivalent split-tool and spatially opposed tools check total MPC reduction, independent moving-tool work, failed projection rollback and interrupted restart | Mixed curved/faceted friction, affine-periodic friction, broader plastic branch switching, external structural comparison and finer mesh/penalty checks |
| Implicit friction | Fixed, translating and rotating analytical planes are admitted on serial strong-BC finite-J2. A certified closed-convex triangle solid now shares the selected planar-branch tangent: a rotating single-face path matches the equivalent analytical plane, and one plastic noncoplanar feature-switch path preserves tangential State and restart. Stable-ID State, rigid pose, separate Coulomb dissipation and numerical separation release, failed-finalization rollback, serial checkpoint and result histories remain common | Mesh/penalty/increment and external sharp-friction comparisons; curved/open/nonconvex/MPC/MPI friction remain unsupported; no general friction claim |
| Material/time composition | Elastic-regime J2 sphere and plastic J2 facet cases complete; the convex sharp-feature path also evolves plastic State and restarts before a feature switch. Explicit body contact has its own stable-step contract | A second history material on the *same implicit* seam and explicit contact stability under evolving state |
| Independent accuracy and speed | Full/quarter symmetry, mesh, penalty and increment screens exist; half-space mismatch remains visible. The sharp plastic screen below shows substantial mesh and penalty sensitivity, so it is not convergence evidence | Controlled far-boundary, graded mesh, pressure-footprint and penalty limits; external FE comparison and whole-step timings |
| Product route | Ordinary `model.step()` and isolated installed-wheel closed-convex and rotating-plane-friction examples have run; final unsmoothed reference-trace pressure/penetration scalars are available; a serial accepted-point NPZ/JSON archive now keeps pair identity, signed gap, reference weights, pressure, friction traction and accepted tangential state without smoothing; the public capability matrix is explicit | ParaView/VTK view for point samples, broad installed-platform and candidate-bound release gate |

The output candidate at commit `e054df76` was built as an isolated native
Apple Silicon wheel, installed under a temporary path and imported from that
path while the process ran outside the source checkout. The ordinary
moving closed-convex-friction example completed four increments, reported
tool work `2.2357364364` and friction dissipation `1.1808584038`, and wrote
`accepted_contact_points.npz` plus its schema JSON. The archive opens with
`numpy.load(..., allow_pickle=False)` and contains six raw quadrature points
for this tiny fixture. This is an installed-path/format smoke, not a
Windows/Linux test or external contact validation.

The [Abaqus/Standard general-contact description](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm)
uses separate surface-, edge- and vertex-contact formulations. This supports
treating sharp features as a separate numerical law, not treating the current
selected-facet normal as a differentiable smooth surface. No row in this map
is promoted merely because a single visual plot or Newton solve looks plausible.

## Next implementation decisions

The implicit friction route reuses the accepted, stable-ID
`TangentialContactState` and return map used by explicit contact. At fixed old
State and a current prescribed analytical-plane pose, the tangent must include
both the pressure cap and the tangent projection of the transported master
material point. Writing `P=I-n⊗n` and old/current rigid rotations as
`R_old`, `R_new`, that relative tangential-increment derivative is
`A=P R_old R_new^T P`. Thus a stationary or purely translating plane has
`A=P`; replacing `A` with `P` for a rotating tool would give the wrong Newton
matrix even though the return-map value looked plausible. The current-pose
projection is checked against the supplied pose to reject a stale trial.
The slip branch also differentiates `mu p` and slip direction. At opening
and the exact stick/slip cap the chosen semismooth derivative is not a
classical two-sided derivative.

`implicit_friction.py` maps this point derivative through the reference trace
to the normal-plus-friction Newton Operator. Solver-neutral and real DOLFINx
stick/slip direction checks pass for fixed, translating and rotating planar
tools, including a nonzero accepted old rotation. A serial strong-BC finite-J2
rotation path has nonzero generalized moment, accepted tool work, positive
Coulomb dissipation and interrupted restart agreement. Multiple independently
named planar pairs share the same Procedure transaction; a second-pair commit
fault restores both accepted states. Curved, faceted, affine-MPC and MPI
friction remain rejected. The accepted energy record
reports Coulomb dissipation separately from recoverable tangential penalty
energy and numerical separation release; its trapezoidal residual is a
diagnostic, not a universal first-law proof.
For the dimensionless prescribed 0.15-radian plane rotation and 0.4 tangent
translation, a 2/4/8/16-increment screen gives tool work
2.3731/2.2357/2.2688/2.2749 and friction loss
1.7735/1.1809/1.0495/0.9835. The signed mechanical-energy residual changes
from -0.5847 to -0.0628 over that scan. The finite-increment values are not
yet converged and no physical-unit accuracy claim follows from this screen.
The sliding tangent includes a pressure-cap cross term and is generally
non-symmetric. Public admission rejects symmetry-only linear solver
selections such as CG or Cholesky; the default direct LU route is accepted.
At the exact stick/slip cap or opening the selected semismooth branch is not
a classical two-sided derivative. This narrow global solve and return-map
agreement do not certify a broad industrial friction capability.
The [FEBio frictional-contact formulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/)
uses an explicit stick/slip return map and verifies patch, Hertz and
large-sliding cases; its numerical results are not AgentFEM evidence.

The source example `finite_j2_fixed_plane_friction.py` deliberately crosses
the slip cap under combined normal and shear loading. For 4, 16, 64 and 128
accepted load increments on its one-cell-per-axis fixture, cumulative
Coulomb dissipation is `0.18695`, `0.12040`, `0.10915`, `0.10762`; the
trapezoidal energy residual is `-0.07612`, `-0.01356`, `-0.00305`, `-0.00149`.
The full contact force changes by less than `0.05%` between 16 and 128
increments. These dimensionless numbers document strong coarse-step
dissipation sensitivity and apparent path refinement, not mesh or external
benchmark convergence. A weaker synchronous shear load of 15 showed
vanishing slip upon refinement; it was discarded as a misleading sliding
fixture. No exact first-law or general friction claim follows from this screen.

An isolated native `0.4.3` wheel built from source commit `60bb91e1` was
installed into a temporary directory and imported from that directory, outside
the source checkout. The ordinary `model.step()` examples for the moving,
rotating Coulomb plane (four increments) and closed convex tool (two cells,
sixteen increments) both completed. The plane reported tool work `2.23573644`
and Coulomb dissipation `1.18085840`; the convex example reported right-end
displacement `-0.0009990007775003966` versus its narrow analytical bar value
`-0.0009990007774995924`. This proves the local installed-package path for
these examples, not Windows/Linux portability, curved friction accuracy or a
release-grade scientific validation.

### Purely translating planar Coulomb tool

The same serial strong-BC J2 Operator now accepts a prescribed **translation**
of an analytical frictional plane on the shared normalized Step coordinate;
rotation still fails closed. `relative_contact_displacement_increment` tracks
the current projected master material point back to the accepted rigid pose,
so tangential tool motion is not erased by subtracting gap vectors. With old
State and current pose fixed, the planar stick/slip tangent remains the same
derivative with respect to slave displacement. The accepted pose factor is
stored in point-keyed friction kinematics; corrupted pose-factor checkpoints
are rejected before replacing State. Tool work uses the total normal-plus-
friction resultant and moment, while Coulomb dissipation and recoverable
tangential penalty remain separate channels.

Focused DOLFINx tests check translated normal and tangential forces,
fixed-old-State directional tangents, rotating-tool rejection, an ordinary
moving-tool J2 Step, accepted tool work and interrupted restart. The public
two-increment fixture deliberately crosses the Coulomb cap; a weaker path
reached the cap without sliding and was not used to claim dissipative
validation. This bounded translation path is not a curved-tool, rotating-
frame, general finite-sliding or affine-MPC friction solver.

The same 16-increment fixture completed from an isolated locally built macOS
wheel with no source checkout on `PYTHONPATH`; `agentfem.__file__` resolved
inside the isolated wheel target. This is an installed-package smoke, not a
Windows/Linux or release qualification.
An independently calculated, displacement-controlled FE patch checks the
complete contact integration and public reaction field on an area-one face:
with normal penalty `500`, penetration `0.05`, tangential penalty `100`, and
`mu=0.4`, the normal resultant is `-25`. At tangential displacement `0.02`
the force is `-2` and sliding loss zero; at `0.20` the force is capped at
`-10`, cumulative Coulomb loss is `1.0`, and recoverable total contact
potential is `1.125`. Both 3D J2/strong-BC solves match these constructed
values within `1e-10` absolute tolerance. This is a bounded patch identity,
not an independent large-sliding or forming validation.
This patch also exposed and fixed a result-contract defect: the public nodal
`RF` field previously assembled only the UFL material/external form while a
native rigid-pair contact Operator was added separately to Newton. It now
uses the same full-residual helper as prescribed-work integration. The patch
checks that strong reactions plus the independently reported contact
resultant vanish without external traction; native contact is no longer
missing from `RF`. Legacy UFL contact remains assembled once.

In-memory increment rollback now retains the existing read-only, point-keyed
friction records instead of converting every field to nested Python lists.
Only a written checkpoint serializes JSON-safe values. A targeted conversion
microbenchmark with 50,000 points measured about `8.0 ms` per former
list-producing snapshot on this machine; the new runtime snapshot stores two
object references. This removes that conversion from each accepted attempt,
but is **not** a measured whole-solve acceleration and must not be marketed as
one. Single- and multi-pair failure rollback still pass after the change.

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
yet evidence for arbitrary shaped dies, mixed sharp/curved branch switches, moving/curved friction, affine
MPC multi-pair reduction or MPI.

`examples/finite_j2_opposing_rigid_tools.py` is a compact ordinary
`model.step()` example. A locally built candidate wheel was installed with
`--no-deps --no-index --target` into an isolated directory and that example
completed from outside the source checkout; the imported package resolved to
the isolated wheel target. The result carried four accepted increments, two
separate tool resultants and accepted works, and an available constraint work
balance. This is a local Apple Silicon installed-wheel smoke, **not** a
Windows/Linux platform or final release acceptance.

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
