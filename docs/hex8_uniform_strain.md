# Uniform-gradient Hex8 development boundary

The experimental `elements.uniform_strain_hex8(...)` policy lowers ordinary
`model.step()` to a small-strain elastic explicit provider with owned-cell MPI
assembly. Nonmatching interface composition remains serial. The private
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

### Finite-deformation prerequisite (private, not a Step)

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

The private serial DOLFINx bridge now reuses `MaterialQuadratureResponse` at one
material point per cell, with fixed committed history during trial evaluations.
Downstream element/scatter failures discard trial state and restore scratch
stress/tangent fields. The caller, not the element, commits accepted material
state. A 12-cell global J2 patch checks interior equilibrium, boundary Piola
traction, current-configuration moment, global tangent differences, and four
loading/unloading increments. Its tiny dense Newton loop is a test oracle, not
a second production solver. Shared Q1 layout preparation is also consumed by
the existing small-strain operator.

Stored-energy availability is explicit in the shared material batch result:
missing optional energy is not a physically defined zero. This route rejects
providers without stored energy, and rejects a material-requested increment
reduction instead of ignoring it. Empty MPI material partitions preserve their
component schema and neutral time-scale summary (separate driver evidence,
not admission of distributed finite Hex8 execution).

An instantaneous spectral screen bounds the declared nodal tangent through
small Gram matrices, checked against an explicit 24-by-24 matrix. It requires a
symmetric positive-semidefinite material tangent and rejects unsupported
asymmetry/negative curvature; it is not a general elastoplastic wave-speed
policy or a guarantee over a future increment.

The diagnostic-only signed report separately encloses positive and negative
eigenvalues using the positive/negative parts of the symmetric material
tangent. It does not replace that tangent by its absolute value. A regression
demonstrates that an isochoric J2 patch can have negative material curvature
while its assembled, constrained free-DOF matrix is positive definite. Thus a
local curvature finding is not reported as a structural instability verdict.

The private serial residual lifecycle now runs under the existing central-
difference Procedure with joint nodal/material rejection, accepted-time guards,
and same-partition checkpoint recovery. Tests inject failure even after material
commit and reproduce the uninterrupted trajectory after retry. A constrained
one-cell elastic Hencky bar is checked against an independently integrated ODE,
`m_eff u'' + C log(1+u)/(1+u) = 0`, with extension exceeding 10% and second-order
time refinement over three increment sizes. This checks temporal integration
of the one-cell model, not spatial continuum convergence or plastic dynamics.

That private route requires a caller-declared complete-path spectral ceiling
and checks each endpoint against the nonnegative symmetric-tangent screen.
Ordinary isochoric J2 extension can fall outside this screen; negative material
curvature is not by itself a constitutive bug or proof of global instability.
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
Finite-strain public Step lowering, distributed/portable restart, general
finite-deformation interface kinematics and contact composition remain
unimplemented. The public small-strain policy is unchanged.
The batch stores compact geometry and evaluates forces/tangent actions in
bounded chunks without retaining dense 24-by-24 element matrices. No measured
finite-strain whole-solver speedup is claimed.

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

## Remaining implementation gates

Compact preparation and serial affine/bending/wave checks are implemented.
DOLFINx retains mesh/DOF ownership; Procedure retains time integration and
stability composition. The experimental policy, ordinary Step, same-partition restart
and separate artificial energy are implemented. Remaining work includes
distortion-dependent accuracy evidence, richer loading
evidence and distributed nonmatching interface composition. No implicit numerical
equivalence to imported commercial reduced-integration elements is assumed.

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
- Abaqus verification documentation, *Performance of continuum and shell
  elements for linear analysis of bending problems*:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-linbending.htm.
  Motivation for thickness refinement, hourglass-energy inspection and not
  equating patch-test success with reliable bending performance.
