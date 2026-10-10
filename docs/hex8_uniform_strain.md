# Uniform-gradient Hex8 development boundary

The experimental `elements.uniform_strain_hex8(...)` policy lowers ordinary
`model.step()` to small-strain elasticity or a bounded finite-strain history
provider, both with owned-cell MPI assembly and same-partition restart.
Nonmatching interface composition remains serial. The private
`elements._uniform_hex.UniformHex8` remains its local verification oracle.
This is not advertised as an Abaqus C3D8R reproduction.

## Ordinary workflow

```python
policy = elements.uniform_strain_hex8(
    hourglass_modulus=material_shear_scale,
    hourglass_scale=0.1,
)
step = model.step(target=u, element_policy=policy, dt="auto", steps=100)
result = step.solve_result(
    output="fields.xdmf", field_variables=("S", "E", "MISES", "SENER")
)
```

Optional cell fields come directly from the uniform-gradient constitutive
response: `S` is physical stress, `E` is small strain (both 3-by-3 tensors),
`MISES` is von Mises equivalent stress and `SENER` is physical strain-energy
density. They use DG0 without smoothing across cells or material interfaces.
Artificial hourglass energy remains separate; no artificial stress is added to
`S`. Von Mises output is not an anisotropic failure criterion. Units follow the
model's consistent unit system and are not inferred. Processing provenance and
cell location remain in `SimulationResult` with or without file output. Omitting
`field_variables` preserves the existing primary-field default; an empty tuple
requests no derived fields. Unsupported fields are rejected before stepping.

### Bounded finite-strain workflow

Set `kinematics="finite_strain"` on the same element policy. Register one
full-domain material with positive density, the existing `dP/dF` convention,
stored energy/dissipation and an explicit `initial_array_response(count)`.
The native logarithmic J2 material accepts optional `density=`; density affects
inertia, not the constitutive equations. For example:

```python
material = constitutive.finite_strain_j2_logarithmic(
    young=100.0, poisson=0.3, yield_stress=1.0,
    hardening_modulus=5.0, density=2.0,
)
model.material(material)
policy = elements.uniform_strain_hex8(
    hourglass_modulus=40.0, hourglass_scale=0.1,
    kinematics="finite_strain",
)
step = model.step(
    target=u, element_policy=policy, dt=1e-4, steps=100,
    omega_squared_bound=1e8,
    maximum_negative_growth_per_increment=0.1,
)
result = step.solve_result(field_variables=("S", "P", "F", "SENER", "PEEQ"))
```

The numerical values above are demonstration choices, **not universal stable
defaults**. `omega_squared_bound` is a caller-declared complete-path bulk and
hourglass ceiling. The current signed tangent is screened at each endpoint;
`dt="auto"` only selects from the declared ceiling, not from an automatic
nonlinear wave-speed estimate. Negative curvature is rejected unless its
growth-resolution policy is explicitly supplied, as described below.

`S` is accepted Cauchy stress, `P` first Piola stress, `F` the accepted mean
deformation gradient, and `SENER` material stored energy per reference volume.
Declared material state output names are also available (`FP`, `PEEQ`, `PDENER`
for native J2). All derived fields are unsmoothed DG0 views of accepted state;
output never advances the material. Artificial energy stays in its own ledger.
Strong prescribed motion and displacement-independent reference body/traction
loads are supported. Starts are undeformed with virgin history; restarts reuse
the accepted state. The admitted interface special case remains isotropic,
undamaged and serial. Regional history materials, follower loads, contact,
damage/deletion, mass scaling and cross-partition restart are not admitted.

`examples/finite_hex_extension.py` demonstrates the installed-use workflow and
its same-partition checkpoint. An independent test material (compressible
Neo-Hookean with a named peak-energy history) exercises the same entry point,
array protocol, fields and restart without changing the core or invoking a
per-point Python update. This is protocol integration evidence, not validation
of arbitrary external materials. A separate ordinary-Step regression combines
finite bulk and unequal-mesh Q1 elastic bonding and verifies interrupted
recovery against continuous execution.

### Small-strain elastic workflow

The model uses a 3D `studies.dynamic_solid()` Study, continuous Q1 hexahedra,
registered constant isotropic or anisotropic elasticity, positive density and
ordinary strong constraints. Fixed material orientation and complete disjoint
material regions are supported. The engineering-shear order is shared with
the existing elasticity module. A prescribed time increment above the
conservative bulk-plus-hourglass bound is rejected. `dt="auto"` uses that
bound with safety 0.8; it does not tune an adaptive nonlinear increment.

The bound uses small Gram spectra rather than a coarse trace estimate:
with `C=L L^T`, physical eigenvalues are those of
`V L^T B M^-1 B^T L` (6-by-6); hourglass eigenvalues are obtained from
`coefficient * gamma^T M_node^-1 gamma` (4-by-4). Adding the two largest
eigenvalues bounds their combined cell stiffness. Positive assembled mass
then makes the maximum cell bound conservative for the body. Optional interface
stiffness is added separately. All temporary arrays are chunked.

In regular and perturbed single-cell checks this permits about 2.97x and 2.95x
larger stable steps than the former trace bound, with the same safety factor.
Twenty independently distorted, positive-definite anisotropic cell checks
compare the bound against full 24-by-24 spectra. Stability is not temporal
accuracy: wave/load time resolution can still require a smaller user-selected
`dt`; no physical time-step convergence claim follows from increasing the bound.

The result distinguishes `strain_energy`, `kinetic_energy`, `hourglass_energy`,
physical `total_mechanical_energy` and `total_discrete_energy`. These components
feed the existing accepted-path `DynamicEnergyLedger`: natural-load and strong
prescribed-motion work are integrated every accepted increment, independent of
history output cadence. The discrete balance explicitly includes artificial
hourglass energy, not just material storage. Rigid acceleration and a deforming
single-cell oscillator supply independent analytic/discrete references; this
is not a blanket accuracy guarantee for arbitrary loads. Serial and distributed
same-partition interrupted/continuous runs agree; changed operator identity
rejects a checkpoint atomically. Public portable restart, finite deformation,
evolving material orientation,
damage/deletion, eigenstrain, additional contact operators and
operator-changing time inputs are not admitted by this provider.

An optional `cohesive_force` now composes the fixed-reference, undamaged elastic
Q1 interface. Its independent nonmatching traces retain original bilinear
interpolation. The conservative stable step includes both interface and bulk
stiffness; interface stored energy and failed-increment/restart state share
the existing lifecycle. See [nonmatching interfaces](nonmatching_interfaces.md).

See `examples/uniform_hex_wave.py` for a runnable ordinary-workflow example.
On 2026-10-09 a candidate wheel was built without isolation downloads, installed
into a separate environment and used outside the source directory to execute
this 100-step example. Field output and the result manifest were written;
the original installed package was not replaced. This was a local candidate
acceptance, not a public version release.

The subsequent owned-cell MPI candidate (commit `f4f6f79f`) was also installed
outside the source tree. Its two-rank 100-step wave publishes one collective
manifest, which passes integrity verification; 13 distributed Hex8 tests pass
against the installed wheel with source-path injection disabled. Collective
publication uncovered and fixed a duplicate rank-local timing record and an
unstable Python-object representation in Step metadata. Timing now uses the
existing min/mean/max performance evidence, while residual identity is explicit.

Candidate `8ab39cd3` additionally passes installed ordinary-Step/interface tests
(14 selected tests), plus a two-rank 100-step wave with constitutive DG0 fields
and a collectively published, integrity-verified manifest. This checks packaging
and publication, not independent physical validation. Serial field regressions
separately verify tensor shear, material stress jumps and the integral of physical
energy density against the accepted energy history.

For imported `C3D8R`, the migration report continues to say `topology_only`.
It points to this explicitly selected alternative but never silently substitutes
it for the source formulation, material updates or section controls.

## Formulation

### Finite-deformation contribution and staged evidence

`elements._finite_uniform_hex.FiniteUniformHexBatch` separates total-Lagrangian
kinematics from material updates. It computes `F = I + sum(u_a outer grad_X N_a)`
using reference-volume-average gradients, consumes first Piola stress and
reference-volume stored-energy density, and maps `dP/dF` to a matrix-free nodal
tangent action. It does not treat Cauchy stress as Piola stress or embed a new
material law. This mean-gradient approximation is not a selective volumetric
F-bar correction.

The fixed-reference artificial potential uses the existing affine-annihilating
mode vectors. Since their contraction with reference coordinates is zero,
their displacement contraction rotates with the current geometry; its squared
norm is objective. The coefficient remains a declared fixed reference value,
not a damage-updated modulus. Current-cell Bernstein Jacobian admission and
positive mean deformation determinant are checked separately.

Local tests cover finite affine deformation, large rigid rotations, superposed
rotation of already deformed/hourglassed cells, resultant force and current
moment, energy derivatives and tangent finite differences. An independent
compressible Neo-Hookean oracle checks the conservative case. The existing
finite-strain J2 batch protocol separately supplies Cauchy stress converted as
`P = J sigma F^-T` and its declared `dP/dF`; the resulting element tangent agrees
with fixed-old-state force differences. Plastic stored energy is **not** treated
as an incremental potential or a complete dissipation balance.

The private DOLFINx bridge now reuses `MaterialQuadratureResponse` at one
material point per cell, with fixed committed history during trial evaluations.
Downstream element/scatter failures discard trial state and restore scratch
stress/tangent fields. The caller, not the element, commits accepted material
state. A 12-cell global J2 patch checks interior equilibrium, boundary Piola
traction, current-configuration moment, global tangent differences, and four
loading/unloading increments. Its tiny dense Newton loop is a test oracle, not
a second production solver. Shared Q1 layout preparation is also consumed by
the existing small-strain operator.

Private serial material checkpoints use an optional typed numeric-array payload
instead of expanding integration-point arrays into JSON lists. The v6 manifest
binds its payload by size and SHA-256; shape, dtype and finite values are checked
before nodal assignment, and material identity is checked within joint rollback.
Reading uses `allow_pickle=False`. The previous v5 JSON auxiliary representation
remains readable. Failed payload/manifest publication preserves the previous
checkpoint and removes the new generation's unpublished files when possible.
The composed Procedure capability explicitly refuses rank-count portability:
portable nodal fields alone do not make material or interface history portable.
The subsequent v7 encoding adds rank-local arrays for same-partition MPI
restart; it does not make these arrays portable across partitions.

The private assembly exposes one open trial scope through material evaluation,
force scatter and the Procedure's final checks, avoiding three nested copies of
the same scratch fields. Downstream exceptions (including interruption) restore
those fields and discard material trial state. Fixed-reference spectral geometry
is cached compactly per cell; material-dependent spectra are still
recomputed. No stale tangent or unchecked current-cell geometry is reused.

Stored-energy availability is explicit in the shared material batch result:
missing optional energy is not a physically defined zero. This route rejects
providers without stored energy, and rejects a material-requested increment
reduction instead of ignoring it. Empty MPI material partitions preserve their
component schema and neutral time-scale summary; owned-cell MPI assembly and
empty-rank restoration are checked separately.

An instantaneous spectral screen bounds the declared nodal tangent through
small Gram matrices, checked against an explicit 24-by-24 matrix. It requires a
symmetric material tangent and rejects unsupported asymmetry. The default
rejects negative curvature; an explicit signed growth-resolution option is
described below. This is not a general elastoplastic wave-speed
policy or a guarantee over a future increment.

The signed report separately encloses positive and negative
eigenvalues using the positive/negative parts of the symmetric material
tangent. It does not replace that tangent by its absolute value. A regression
demonstrates that an isochoric J2 patch can have negative material curvature
while its assembled, constrained free-DOF matrix is positive definite. Thus a
local curvature finding is not reported as a structural instability verdict.

The private residual lifecycle now runs under the existing central-
difference Procedure with joint nodal/material rejection, accepted-time guards,
and same-partition checkpoint recovery. Tests inject failure even after material
commit and reproduce the uninterrupted trajectory after retry. A serial user
interruption likewise restores the accepted nodal, material and
interface station before propagating cancellation; this is not coordinated
recovery from an arbitrary MPI process failure. Cancellation after auxiliary
checkpoint restoration also restores the complete pre-load station before
propagating, and a subsequent continuation reproduces an uninterrupted run.
A constrained
one-cell elastic Hencky bar is checked against an independently integrated ODE,
`m_eff u'' + C log(1+u)/(1+u) = 0`, with extension exceeding 10% and second-order
time refinement over three increment sizes. This checks temporal integration
of the one-cell model, not spatial continuum convergence or plastic dynamics.

The finite route requires a caller-declared complete-path spectral ceiling
and checks each endpoint against the signed symmetric-tangent screen.
Ordinary isochoric J2 extension can have negative material
curvature; this is not by itself a constitutive bug or proof of global instability.
General wave-speed/curvature treatment remains a gate, not an absolute-value
workaround. A private composition now admits the already existing elastic
nonmatching interface **only when all three separation stiffnesses are equal**.
Its reference-area potential is `K * |jump|^2 / 2`; traction is parallel to the
jump. Checked coplanar common refinement and Q1 trace interpolation are retained.
Tests rotate an already opened interface, check current force/moment balance,
and exercise joint explicit commit, injected failure and interrupted recovery.
The combined stable bound adds the interface contribution; it is not the
minimum of the independent bulk/interface time limits. This is a bounded
isotropic elastic special case, not a convected anisotropic/damaging interface.
Unequal normal/tangential stiffness is explicitly rejected. The restriction is
consistent with the frame-indifference/angular-momentum analysis of
[Ottosen, Ristinmaa and Mosler (2016)](https://doi.org/10.1016/j.jmps.2016.02.034);
their general surface-deformation-gradient extension is not implemented here.
Finite-strain public Step lowering and same-partition distributed bulk restart
are implemented within the bounded workflow above. Cross-partition restart,
general finite-deformation interface kinematics and contact composition remain
unimplemented. The default small-strain policy is unchanged.
The batch stores compact geometry and evaluates forces/tangent actions in
bounded chunks without retaining dense 24-by-24 element matrices. The measured
columnar-history speedup below applies to its stated workload; newer spectral
and kinematics optimizations require their own controlled measurements.

Reference: [Sierra/SM Theory Manual, §15.1.4](https://www.sandia.gov/files/sierra/SM_Theory_5_20/main/element_formulations.html)
motivates an objective reference-configuration hourglass potential; our existing
normalization and coefficient remain explicitly declared, not Sierra defaults.

### Public small-strain formulation

Basix supplies Hex8 node order, trilinear interpolation and geometric
quadrature. Reference preparation integrates volume and shape gradients once.
The material subsequently sees one volume-average engineering strain vector
in the existing elasticity order `(xx, yy, zz, 2yz, 2xz, 2xy)`. A symmetric positive-definite
six-by-six elasticity tensor maps that vector to physical stress components.
The preparation quadrature does not imply multiple constitutive updates.

Four reference hourglass vectors are normalized by `sqrt(8)`. The corrected
vectors are `gamma = Gamma - grad_mean @ (X_centered.T @ Gamma)`, annihilating
affine displacement. The candidate's explicitly chosen stiffness energy is
`scale * modulus * V^(1/3) * sum((gamma.T @ u)^2) / 2`.
This normalization and coefficient are declared choices, not commercial code
defaults; bending and wave evidence must constrain their appropriate use.
Physical and artificial energies remain distinct. Consistent row-sum masses
are integrated rather than assuming distorted cells have equal nodal mass.

Geometry admission expands the triquadratic Jacobian determinant in a tensor
Bernstein basis, using 27 Basix evaluations. Positive control coefficients bound
the determinant throughout the reference cell. Ambiguous cells undergo adaptive
half-subdivision, limited to depth 5 and 4,096 visited boxes per cell. Unresolved
cells are rejected explicitly, not mislabeled as inverted. Preparation remains
chunked; ordinary affine cells take the vectorized fast path.

The check includes a scale-relative floating-point margin; it is not an
interval-arithmetic certificate, a guarantee of good approximation quality, or
a test for overlaps between separate cells. A regression fixture passes the old
corner/Gauss samples but has a negative determinant elsewhere and is now refused.
Distortion-dependent accuracy still requires mesh convergence evidence.

## Local evidence

Targeted tests verify regular/distorted affine response,
infinitesimal rigid translation/rotation, six physical stiffness modes,
twelve controlled hourglass modes, six remaining rigid null modes,
energy-gradient/tangent consistency, positive mass, geometric scaling, and
independent higher-order integration of mean gradients and mass. General
positive-definite orthotropic elasticity is exercised. The local oracle retains
a mass-scaled stiffness row-sum bound; compact execution uses the sharper Gram
bound described above and composes it into the whole-model time-step policy.

No finite-rotation objectivity, material damage, deletion, viscous/relaxation
control or MPI capability follows from these local checks. Distributed assembly
and same-partition restart are tested separately through the ordinary Step.

## Compact execution and serial global evidence

`UniformHexBatch` retains mean gradients, four corrected hourglass vectors,
positive nodal masses and coefficients. Shared material stiffness is stored
once; heterogeneous tensors are supported. Geometry preparation and response
evaluation are chunked. No per-cell 24-by-24 matrices are retained. Response
iteration can consume chunks without storing all output fields. Global residual
evaluation also gathers displacement through the node map one chunk at a time;
it no longer creates a full `(cells, 8, 3)` displacement copy each step.

A 2,048-cell, single-thread local microbenchmark on 2026-10-09 compared five
response repetitions against the scalar reference operator. Median response
times were 0.00192 s (batch) and 0.01785 s (scalar), approximately 9.3x. Compact
stored arrays occupied 1,081,632 bytes versus 18,874,368 bytes for two dense
cell matrices alone. This is not a complete solver speedup or peak-memory
measurement; global DOFs, connectivity and temporary gathers add storage.

A separate 20,000-cell gather comparison (three repetitions, same thread limit)
measured traced transient allocations of 5,155,736 bytes for the full displacement
gather versus 1,512,088 bytes for chunk gathering. Median response time changed
from 0.02053 s to 0.02119 s; energies were identical. This optimization reduces
temporary allocation, not measured CPU time, and does not represent process RSS.

Energy monitoring now uses an energy-only evaluation of the same cell kinematics;
it does not recompute unused nodal forces or skip accepted increments. Residual
assembly reuses bounded, chunk-local reduction maps, with PETSc still responsible
for ghost accumulation. An alternating five-repeat comparison on 8,192 cells
and 50 steps measured median run time 0.85270 s before these two changes and
0.62880 s after (1.36x throughput, about 26% less run time). Physical energy and
kinetic energy agree to roundoff. This is one local no-field-I/O elastic workload,
not a claim about nonlinear material, contact or industrial models.

`tools/benchmark_uniform_hex.py` measures ordinary-Step preparation, run time,
peak process RSS, and the existing collective performance evidence. A bounded
262,144-cell/20-step single-thread run (energy-only path before scatter reduction)
took 7.04 s preparation and 8.75 s advancement, with 1,258,487,808 bytes peak RSS.
RSS includes mesh/runtime/preparation storage and is not the compact kernel size.
The automatically selected step is stable for this declared linear operator,
not an accuracy criterion. No field I/O, contact or material history is included.

The private `UniformHexResidual` uses DOLFINx blocked Q1 DOF maps and the
existing central-difference integrator. Serial affine internal forces agree
with independent fully integrated UFL assembly. Separate resultant and moment
checks pass. An analytic longitudinal standing wave converges under mesh
refinement, with a bounded energy error. The conservative time-step bound uses
positive cell mass and stiffness Rayleigh bounds; additional interface/contact
stiffness is not included and must be composed separately.

The operator has two-/four-rank owned-cell assembly evidence:
reverse ghost accumulation for mass and internal force, global physical and
artificial energy, a global maximum stability bound, empty local partitions,
and collective rejection of rank-local invalid material/non-finite fields.
Distributed nodal wave response agrees with the serial formulation. Ordinary
`model.step` also passes distributed natural-load and prescribed-motion work,
field output, and same-partition interrupted/continuous runs. A two-rank test
includes an empty owned-cell partition. Rank-local time-input/kinematic failures
are delivered before field communication and accepted state is restored.
These tests do not enable nonmatching interface MPI or cross-partition restart.

A slender cantilever (length 10, unit square section, E=100, nu=0, uniformly
distributed end shear) was checked against Euler-Bernoulli deflection with
the rectangular-section Timoshenko shear correction. This is a beam-theory
reference, not an exact three-dimensional solution or a commercial-code run.

| Hourglass scale | 20x2x4 relative tip error | 40x4x8 relative tip error | Fine-mesh artificial energy / total |
| --- | ---: | ---: | ---: |
| 0.05 | 4.883% | 1.177% | 0.375% |
| 0.10 | 3.275% | 0.799% | 0.747% |
| 0.20 | 0.205% | 0.051% | 1.484% |

All three choices improve under refinement. This does not select a universal
default coefficient or establish accuracy for a single element through the
thickness. Tests are in `test_uniform_hex.py` and `test_uniform_hex_global.py`.

With deterministic interior-node perturbations up to 20% of each grid spacing
(seed 1729; the box and boundary planes retained), corresponding fine-mesh tip
errors are 0.995%, 0.532%, and 0.341%; artificial-energy fractions are 0.475%,
0.897%, and 1.707%. Every error decreases from the coarse mesh. This is bounded
distortion evidence, not admission of arbitrary poor-quality meshes.

### Nonuniform finite-deformation spatial check

`tools/verify_finite_hex_manufactured.py` uses an independent compressible
Neo-Hookean test provider (mu=30, lambda=40) and a manufactured displacement
`0.08*sin(pi*x)*sin(pi*y)*sin(pi*z)*(1, 0.4, -0.2)` on the unit cube.
UFL differentiates the exact first-Piola field to generate the body force;
degree-ten integration assembles that load and the displacement error.
All boundary displacements are zero. A test-only Newton/CG oracle solves the
private contribution; it is not a new public nonlinear Procedure.

| Cells per edge | Relative displacement L2 error | Hourglass / physical energy |
| --- | ---: | ---: |
| 2 | 34.46% | 14.92% |
| 4 | 5.868% | 2.019% |
| 8 | 1.340% | 0.447% |
| 12 | 0.586% | 0.194% |
| 16 | 0.327% | 0.109% |

The refined displacement error approaches second-order convergence, while
artificial energy decreases. Equilibrium residuals are below 1e-9 relative.
The coarse mesh is visibly inadequate: patch-test success must not be confused
with accuracy for nonuniform deformation. This is regular-mesh hyperelastic
verification, not plastic localization, industrial forming, or Abaqus agreement.

With deterministic interior-node perturbations of up to 20% of grid spacing
(seed 1729; boundary retained), errors on 4/8/12/16 cells per edge are
6.483% / 1.596% / 0.732% / 0.417%; corresponding artificial/physical energy
ratios are 2.052% / 0.488% / 0.221% / 0.126%. Full-cell geometric admission is
still required. This bounded perturbation test does not admit arbitrary distortion.
Raw regular/distorted records, including clean source revision `341060e3`, are
retained under `evidence/hex8/2026-10-09-finite-spatial-*.json`. Their wall times
overlap a separate endurance run and are not used as performance evidence.

## Remaining implementation gates

### Bounded serial capacity evidence

The private affine-dilation oracle completed 262,144 Hex8 cells and 500 accepted
increments at clean source `aca01be0`. Peak process RSS was 3,074,506,752 bytes
(about 3.07 GB); final absolute displacement/stress/stored-energy errors were
6.38e-16, 1.50e-12 and 1.13e-12 against the independent homogeneous solution.
The native finite-strain J2 provider remained elastic. The record is
`evidence/hex8/2026-10-09-finite-affine-capacity.json`.

This establishes bounded serial capacity and repeated state advancement, not
plastic dynamics, spatial convergence, a complete energy balance or industrial
forming. The run predates columnar transport; its elapsed time overlapped other
checks and is not used to quantify that optimization. Its explicit 8 GiB budget
was an observed-RSS stop at reporting stations, not an operating-system limit.

### Public and private promotion boundaries

Compact preparation and serial affine/bending/wave checks are implemented.
DOLFINx retains mesh/DOF ownership; Procedure retains time integration and
stability composition. The experimental policy, ordinary Step, same-partition restart
and separate artificial energy are implemented. Bounded distortion is tested;
remaining work includes more severe and application-specific mesh-quality
accuracy limits, richer loading
evidence and distributed nonmatching interface composition. No implicit numerical
equivalence to imported commercial reduced-integration elements is assumed.

### Accepted finite-explicit energy and signed-curvature policy

The finite route now reuses `DynamicEnergyLedger` with cached accepted
force and material energy. Initial energy is declared explicitly by the material
(the native logarithmic J2 law supplies a virgin response); no artificial time
increment is used to obtain an initial stress. The same declaration populates
the initial quadrature response fields. Sampling and restart do not reintegrate
the material. Bulk stored energy, artificial hourglass energy, isotropic
reference-interface energy, cumulative material dissipation and kinetic energy
remain separate. Prescribed-motion work uses the existing reaction-path ledger.
Accepted values participate in the same atomic rollback/checkpoint transaction.

Independent proportional finite-stretch checks cover elastic and plastically
yielding logarithmic J2 response, stored energy and yield-stress-times-PEEQ
dissipation. Time refinement reduces the work/energy residual. The interface
composition has interrupted/continuous restart and cached-energy checks. These
tests do not establish arbitrary plastic loading, localization or forming.

The default still rejects negative material curvature. An explicit
option may instead require `dt * sqrt(negative_bound) <= eta`, with
`0 < eta <= 0.25`, while retaining the positive-frequency ceiling separately.
For a frozen scalar negative mode, central difference has growth rate
`2*asinh(omega*dt/2)/dt`, rather than the exact `omega`; this motivates a growth
**resolution** limit, not a physical stability certificate. Negative material
curvature is not sufficient to diagnose a constrained global instability.
No negative eigenvalues are silently replaced by their absolute values in the
physical residual or tangent. This opt-in and the signed spectrum are included
in restart identity/evidence. Nonsymmetric tangents remain rejected.

### Remaining general finite-explicit gate

The finite bulk assembly now retains owned-cell force/energy contributions and
uses DOLFINx quadrature maps to exchange deformation gradients for ghost points.
Nonuniform finite plastic trajectories agree with serial execution on the tested
partitions; an empty owned-cell partition is valid. Rank-local geometry, force,
spectrum and accepted-energy failures reject collectively. Explicit Procedure
now synchronizes a completed material commit before any rank enters subsequent
monitoring, including an injected failure after one rank committed. The shared
transient schema v7 stores numeric auxiliary state per rank, while v6 remains
readable for serial arrays and v5 for JSON state. Finite same-partition MPI
restart now has interrupted/continuous, empty-rank, corrupt-payload,
swapped-partition and failed-publication checks. A rank's material/energy
payload is bound to its local identity in addition to the collective identity.
Cross-partition recovery and distributed nonmatching interfaces remain separate
gates. The material driver may update visible ghost points, but these never
contribute a second time to force, energy or negative-curvature cell counts.

The signed screen distinguishes the highest oscillatory frequency, negative
curvature and the declared complete-path ceiling. It never takes absolute
eigenvalues as a replacement constitutive tangent. Automatic material-owned
wave-speed/effective-modulus admission remains a separate gate.

The practical design sequence is: material-owned effective-modulus evidence;
Operator-owned geometric/mass conversion and additive interface contribution;
Procedure-owned increment selection and rejection; State-owned rollback.
Initial/accepted response sampling must not manufacture a positive time increment
to query a rate-dependent law. Energy output must consume an explicit accepted
response, including its initial energy, before claiming a global balance.

Abaqus documents separate user-supplied effective bulk/shear moduli when its
automatic estimate is not conservative for a highly nonlinear user material.
This supports a distinct stability contract rather than assuming that every
algorithmic tangent is a wave-speed model. Its explicit analysis guide also
distinguishes element-wise and global frequency estimates. These are design
references, not claims that AgentFEM implements those algorithms:
[VUMAT effective moduli](https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-vumat.htm),
[explicit stability estimation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm).

## Sources

- Johnen, Weill and Remacle, 2017, *Robust and efficient validation of the linear
  hexahedral element*, https://arxiv.org/abs/1706.01613.
  Admission uses the Bernstein convex-hull and subdivision construction, not
  the paper's optimized 20-tetrahedron implementation or its performance claim.
- Flanagan and Belytschko, 1981, *A uniform strain hexahedron and quadrilateral
  with orthogonal hourglass control*, https://doi.org/10.1002/nme.1620170504.
- Sandia, *Sierra/SM Theory Manual*, section 15.1, uniform-gradient Hex8 and
  orthogonal hourglass control:
  https://www.sandia.gov/files/sierra/SM_Theory_5_20/main/element_formulations.html.
  The candidate uses the affine-removal construction but explicitly documents
  its own stiffness normalization rather than claiming full Sierra equivalence.
- Sandia, *Sierra/SM Theory Manual*, Dynamics, element eigenvalue bounds:
  https://www.sandia.gov/files/sierra/SM_Theory_5_30/main/dynamics.html.
  The small-Gram calculation above applies this bound to the declared fixed
  linear operator; it does not implement a global Lanczos estimator.
- PETSc, `VecGhostUpdateBegin`, reverse-add accumulation of ghost contributions:
  https://petsc.org/main/manualpages/Vec/VecGhostUpdateBegin/.
- Abaqus theory, *Energy balance*, separating work, kinetic energy and physical
  internal-energy channels:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-energybalance.htm.
  AgentFEM's present discrete ledger and its tested material decomposition are
  bounded independently; this reference does not imply Abaqus equivalence.
- Abaqus verification documentation, *Performance of continuum and shell
  elements for linear analysis of bending problems*:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-linbending.htm.
  Motivation for thickness refinement, hourglass-energy inspection and not
  equating patch-test success with reliable bending performance.
