# Nonlinear Solid Mechanics Architecture

This document defines the P1 boundary for turning AgentFEM into a credible
laboratory-scale nonlinear solid-mechanics platform. It separates executable
capability from planned extension seams so that a useful API is not confused
with unverified solver breadth.

## Public analysis language

A standard finite-deformation analysis should expose engineering decisions,
not weak-form or backend plumbing:

```python
output = results.output_plan(
    directory,
    field=results.field_output("U", "S", "E", "J", every="increment"),
    requests=(
        results.solver_history(),
        results.finite_strain_checks(),
    ),
)

step = model.step(
    target=u,
    material=material,
    constraints=constraints,
    incrementation=steps.automatic(
        initial=0.1,
        minimum=1.0e-5,
        maximum=0.25,
        max_increments=100,
    ),
    solver_options=solvers.newton(
        relative_tolerance=1.0e-8,
        maximum_iterations=25,
        linear_solver=solvers.direct_solver(),
    ),
    output=output,
)
result = output.finalize(
    model=model,
    step=step,
    result=step.solve_result(),
    target=u,
    material=material,
)
```

The `Study` and registered material choose a step provider. The public model
does not gain one step method per constitutive law. UFL remains the internal
formulation language and an expert escape hatch, but ordinary model scripts
use standard kinematics, materials, constraints, loads, step controls, and
result requests.

## Implemented reusable assets

- `constitutive.kinematics(u)` provides standard total-Lagrangian `F`, `C`,
  `J`, and Green--Lagrange strain.
- `solvers.newton(...)` is one backend-neutral Newton policy. AgentFEM adapts
  it to PETSc SNES or exact affine-reduction algebra.
- `steps.automatic(...)` owns increment growth, cutback, termination limits,
  and rollback of the current global displacement.
- `results.output_plan(...)` separates field frames, histories, diagnostics,
  presentation, model IR, and the result manifest.
- periodic-cell history requests integrate complete macro stress and strain
  tensors, stress triaxiality, normalized Lode state and Hill--Mandel work
  evidence over every accepted affine increment. Spatial field cadence is
  independent and may remain sparse.
- finite-strain checks report average `F`, average and quadrature bounds of
  `J`, maximum displacement, and optional periodic-equation mismatch.
- `SolutionProcedure` separates physical analysis from Standard/Explicit and
  Newmark/generalized-alpha/central-difference algorithm selection.
- `J2QuadratureState` owns committed/trial `PE`, `PEEQ`, `S`, and `DDSDDE`;
  the 3D global J2 provider consumes the reusable `QuadratureTransaction`,
  regional material dispatch, analytical algorithmic tangent, automatic
  cutback, non-monotone load amplitude, MPI global Newton, and portable
  full-Step checkpoint. The distributed route also passes a public
  thick-cylinder structural benchmark.
- J2 and creep quadrature state can be written collectively and restored after
  changing the MPI rank count. The archive uses DOLFINx original input-cell
  identities rather than runtime global cell numbers, which change with the
  partition; it also validates the quadrature rule, mesh fingerprint, state
  schema, and regional material contract before restoring any field.
- `steps.automatic(maximum_inelastic_increment=...)` can reject an otherwise
  converged J2 attempt when its equivalent plastic-strain increment is too
  large. Rejection restores displacement and every trial state field.
- accepted J2 increments retain elastic energy, hardening energy, plastic
  dissipation, total internal energy, and, for nonzero strong prescribed
  displacements, generalized reaction, external work, and balance error.
- isotropic thermoelastic properties feed both implicit heat transfer and the
  equivalent thermal-expansion operator in sequential thermal-stress studies.

The current Neo-Hookean path is stateless. Rolling back its displacement is
therefore sufficient. That fact must not be generalized to plasticity or
creep.

## Implemented state contract and the remaining creep consumer

Path-dependent integration uses an explicit quadrature-state subsystem before
a material is advertised as FEM-integrated:

```text
StateLayout
  names, tensor shapes, units, quadrature rule, schema version

MaterialPointBatch
  committed state at the start of an increment
  trial state associated with the current Newton iterate

ConstitutiveUpdate
  inputs: strain/deformation increment, time increment, temperature
  outputs: stress, algorithmic tangent, trial state, local diagnostics

StateTransaction
  begin_increment()
  update_trial()
  commit()      only after global equilibrium converges
  rollback()    after failed Newton attempt or cutback
```

The state is owned per integration point and material region, not as one
Python object per cell and not as a global material singleton. A global step
must never mutate committed state during a rejected Newton iterate. Checkpoint
and restart serialize the `StateLayout`, committed arrays, mesh/material
identity, step time/load, and schema version together.

The constitutive transaction is MPI-safe. The custom J2 global Newton path has
partition-interface, cutback/rollback, cross-rank-count restart, and external
thick-cylinder structural evidence. The creep global Newton path remains
experimental until its NAFEMS thick-cylinder promotion benchmark passes. This
boundary separates portable material state
from global algebra rather than treating one as evidence for the other.

This mirrors the old/current state distinction required by mature stateful
material systems; see the
[MOOSE stateful material property contract](https://mooseframework.inl.gov/releases/moose/2022-06-10/syntax/Materials/index.html).

`constitutive.QuadratureTransaction` now implements the common atomic
`begin/commit/rollback/snapshot/restore` mechanism and J2 is its first global
consumer. The transaction intentionally does not own a constitutive formula:
creep, damage, and restricted UMAT-style adapters must supply their own local
update, algorithmic tangent, error estimate, and state schema.

The material-point side of that boundary is now explicit through
`MaterialStateVariable`, `MaterialStateSchema`, and
`MaterialTangentConvention`. Scalar and tensor internal variables have a
versioned layout and physically meaningful initial values; a tangent declares
its stress measure, kinematic perturbation, configuration, storage, component
order, shear convention, and objective rate. `validated_material_update()`
fails closed if a provider changes either declaration. An undeclared legacy
6-by-6 array remains inspectable, but it is not eligible for a global Newton
consumer merely because it resembles a stiffness matrix. This is protocol
foundation only: those declarations do not by themselves promote a material.
Finite-strain J2 now has material paths, numerical tangent comparison, and
public `model.step(...)` lowerings for ordinary three-dimensional strong
boundaries with reference dead loads, displacement-only three-dimensional
affine/MPC periodic kinematics, and experimental mixed affine-periodic routes
using P2/DG0 in 3D or Q2/DPC1 in 2D plane strain.
The ordinary and displacement-only affine paths have cutback/restart
equivalence and MPI-stable state identity. The mixed route deliberately has a
narrower serial boundary described below. The capability remains experimental:
independent external structural evidence still gates a broader engineering
maturity claim.

`MaterialQuadratureState.create(domain, schema, ...)` is the first lowering of
that declaration. It creates one committed/trial quadrature pair for every
named scalar or tensor state, preserves output aliases, and embeds the full
schema in portable checkpoint identity. Existing verified J2, Chaboche and
creep containers remain supported while their eventual convergence onto this
neutral storage path is validated incrementally; the presence of shared
storage does not imply that their constitutive algorithms are interchangeable.

`check_material_tangent(material, point)` supplies the corresponding local
evidence for a declared first-Piola/deformation-gradient Jacobian. Every
perturbed call begins from the same old state, so the comparison differentiates
the discrete material update seen by Newton rather than following nine
different histories. Spatial rate tangents such as Abaqus `DDSDDE` are rejected
until an adapter provides and verifies the required convention transform.

The first native consumer of the complete neutral boundary is
`constitutive.finite_strain_j2_logarithmic(...)`. It declares the
multiplicative state `(FP, PEEQ)`, quadratic Hencky elasticity, associated
isochoric J2 flow, linear isotropic hardening, and a row-major `dP/dF`
contract. `constitutive.update_material_points(...)` reads committed
quadrature state, evaluates every local point, writes trial state, and rolls
the whole batch back if one point fails. Inside global Newton it is called with
`commit=False`; only the accepted structural increment may commit.

This is currently an **experimental public finite-element capability**, as
well as a material-point and neutral quadrature provider.
Rigid rotation, superposed rotation, plastic incompressibility, yield
consistency, unloading/reversal, tangent comparison, atomic rollback and
commit are executable tests. One constraint-neutral
`FiniteStrainJ2StateTransaction` owns trial/commit state and accepted fields.
`FiniteStrainJ2StandardProblem` consumes it for ordinary strong Dirichlet or
remote-displacement kinematics, a shared normalized amplitude, and reference-
configuration dead loads. The separate affine provider consumes the same
transaction with exactly one `AbaqusPeriodicConstraint`, using exact affine
elimination in serial or distributed `dolfinx_mpc` Newton. Both assemble the
total-Lagrangian residual from `P` and `dP/dF`, accept fixed or automatic
increments, and perform real rollback/cutback when an otherwise converged PEEQ
increment is excessive.
The older `mechanics.experimental_finite_strain_j2_step(...)` remains a
compatibility/development entry point rather than the recommended application
language.

### Mixed finite-strain J2 ownership

The first experimental mixed J2 lowerings intended to mitigate volumetric
locking use either a three-dimensional tetrahedral P2/DG0 field with one
constant pressure-like unknown per tetrahedron, or a
two-dimensional plane-strain Q2/DPC1 field with three discontinuous pressure
modes per quadrilateral and \(F_{33}=1\). Their scalar unknown is not the
positive-compression Cauchy pressure used by the mixed Neo-Hookean provider. It
is the mean Kirchhoff stress

\[
p=\tfrac13\operatorname{tr}\boldsymbol\tau,
\]

with positive values in tension, and is therefore published under the explicit
field name `MEAN_KIRCHHOFF_STRESS`. The volumetric equation is

\[
\ln J-\frac{p}{\kappa}=0.
\]

The displacement equation uses
\(\mathbf P=\operatorname{dev}(\boldsymbol\tau)\mathbf F^{-T}
+p\mathbf F^{-T}\). The Newton derivative is one monolithic mixed system:

\[
\begin{bmatrix}
K_{uu} & K_{up}\\
K_{pu} & K_{pp}
\end{bmatrix}
\begin{bmatrix}\Delta u\\\Delta p\end{bmatrix}
=-
\begin{bmatrix}R_u\\R_p\end{bmatrix}.
\]

All four blocks are assembled explicitly; pressure degrees of freedom are not
silently removed by the affine displacement reduction. The reported
`pressure_block_residual_norm` is the unnormalised Euclidean norm of the
assembled pressure-equation coefficient vector. It depends on the mesh,
pressure basis and scaling, so it is useful within one declared discretisation
but must not be compared directly across meshes or pressure spaces. The
separately reported maximum quadrature pressure-constraint defect is

\[
\max_q\left|\ln J_q-\frac{p_q}{\kappa_q}\right|.
\]

It is a quadrature representation diagnostic, not the mixed residual norm.

Energy output follows the same separation of physical state from numerical
formulation. For the mixed route,

\[
\mathrm{ELENER}=\psi_{e,\mathrm{dev}}+\frac{p^2}{2\kappa},
\qquad
\mathrm{SENER}=\mathrm{ELENER}+\mathrm{HARDENER}.
\]

Here \(p^2/(2\kappa)\) is a nonnegative condensed mixed representation of the
volumetric storage. It is pointwise identical to the primal
\(\kappa(\ln J)^2/2\) term only where \(p=\kappa\ln J\) holds locally; the weak
mixed equation does not make that identity unconditional at every quadrature
point. The integrated channel is therefore suitable for a declared discrete
mixed-energy diagnostic, but integrating it does not make it the primal
physical-energy observable. With

\[
r_p=\ln J-\frac{p}{\kappa},
\qquad
\overline{\psi}_{e,\mathrm{primal}}
-\overline{\psi}_{e,\mathrm{condensed}}
=\overline{p r_p}+\overline{\frac{\kappa r_p^2}{2}},
\]

the mixed-energy diagnostic reports the primal and condensed channels, their
signed gap, the signed pressure-orthogonality term, the nonnegative
constraint-defect term, and the residual of this decomposition. An external
physical-energy oracle must be compared with the explicit primal channel while
retaining those diagnostics. It must not be compared directly with mixed
`ELENER`. `MIXED_POTENTIAL` retains the saddle variational density containing
\(p\ln J-p^2/(2\kappa)\) and is never a stored-energy alias.

Portable checkpointing also respects ownership. The generic archive never
serializes an opaque mixed-vector layout. The state transaction exposes live
and accepted `U` and `MEAN_KIRCHHOFF_STRESS` as four standalone fields,
preserves the provider-owned quadrature history beside them, and reassembles
the mixed functions only after mesh, field, material, procedure, and
constraint identities pass restore validation. Fresh-Step checkpoint/continue
equivalence is verified for both serial mixed routes: 3D tetrahedral P2/DG0 and
2D plane-strain quadrilateral Q2/DPC1. Separately, the generic DPC cell-interior
state primitive has one-to-two and two-to-one-rank acceptance coverage. That
serializer evidence is not evidence for an MPI mixed-J2 solve or a cross-rank
restart of one.

These routes are intentionally limited to tetrahedral 3D P2/DG0 or
quadrilateral 2D plane-strain Q2/DPC1, one exact affine-periodic constraint,
and serial sparse reduction. A bulk-to-shear ratio of \(10^4\) is a temporary
implementation ceiling that guards the current subtractive tangent extraction;
it is not an audited accuracy range or a material-model limit. They do not yet
support distributed block-aware MPC, ordinary strong-boundary mixed problems,
or body/natural-load power. The Q2/DPC1 path supplies the three pressure modes
of the 9/3 formulation. The exact Zhang--Feng--Khandelwal geometry now has a
direct plane-strain diagnostic driver that reports both primal and condensed
energy channels and their decomposition. The same driver now condenses the
converged full Jacobian through the exact affine lift, records the current-state
homogenized algorithmic tangent, and checks its convention on a homogeneous
Q2/DPC1 patch. That execution remains an unpromoted diagnostic: Table 5
agreement, load-path, formulation and mesh convergence, replicated cells,
MPI/restart equivalence, and content-bound evidence remain open. The existing
thin-3D tetrahedral fixture is a separate experimental
diagnostic rather than evidence of 2D formulation identity or locking
convergence.

Accepted finite-strain state also owns the origin of its live algorithmic
linearization. A homogenized tangent is eligible only while its generation
token still identifies the exact accepted start and target increment used by
the converged Newton solve. Rollback invalidates that token. Resume may recover
the constitutive fields, but a checkpoint that did not persist the accepted
macro-tangent evidence leaves tangent recovery unavailable; AgentFEM never
relabels a zero-increment relinearization as the tangent of the accepted path.

The state transaction owns accepted quadrature `F`, `P`, `S`, `MISES`,
`SENER`, `ELENER`, `HARDENER`, `PDENER`, `FP`, and `PEEQ`; the mixed route also
owns `MIXED_POTENTIAL`. Scientific output uses those provider-owned fields;
it does not reconstruct an inelastic response from a stateless hyperelastic
formula. Explicitly named `*_CELL` fields are physical quadrature-weighted DG0
averages for visualization and do not replace the integration-point evidence.

For the displacement-only J2 providers, `ELENER` is the quadratic Hencky
elastic free-energy density. For the mixed routes it has the condensed meaning
defined above. In both cases, `HARDENER` is
\(\tfrac12 H\bar\varepsilon_p^2\). The backward-compatible `SENER` field is
their sum. `PDENER` is a separate committed state channel,
\(D_{n+1}=D_n+\sigma_{y0}\Delta\bar\varepsilon_p\), which records the
irrecoverable initial-yield work for this rate-independent linear-hardening
law. It is emitted from the accepted constitutive transaction, not reconstructed
from a final visualization field. A complete structural energy balance still
requires provider-owned external work for every active load and constraint.

Displacement-only portable checkpoints are accepted-state boundaries. They
store `U`, `U_ACCEPTED`, committed quadrature state, accepted and attempted
increment histories, and the next adaptive increment; the mixed split-field
contract is defined above. Restore validates
mesh/function identity, material and state schema, quadrature rule, procedure,
solver, increment control, amplitude, constraints, and natural-load identity
before changing the analysis. The affine route also binds its periodic
equations. The displacement-only checkpoint has been resumed between one and
two MPI ranks in both directions. A resumed solve restores the previous execution trace,
appends a new resumed segment, and starts a new field series from the accepted
boundary; earlier visualization frames remain in the earlier result artifact
and are not silently reconstructed or merged.

Adding `PDENER` changes the finite-strain J2 material-state schema from v0.1
to v0.2. A checkpoint written with the earlier schema therefore fails closed
instead of inventing a dissipation history. Re-run that accepted model with
the current material schema before using its state as new release evidence.

The present public scope is deliberately narrow. The ordinary route accepts
proportional prescribed motion and reference dead loads but not follower-load
tangents, absolute time histories, weak boundary models, contact, or MPC. The
displacement-only affine route accepts prescribed macroscopic deformation,
compatible regional materials, one periodic constraint, and no body-force or
natural-load power. The mixed affine route has the narrower serial boundary
defined above.
The true spherical-void RVE now exercises
geometric pairing, positive-J, public result lifecycle, two-rank execution,
and Hill--Mandel evidence. Its versioned fixed-stack Golden additionally
freezes one `h/L=0.25` first-order mesh, two-increment loading path, runtime
stack, and portable mesh identity. It is a software-regression contract for
macroscopic first-Piola stress, physical-weighted PEEQ statistics, and solid
fraction, not a mesh-converged RVE reference value. An opt-in certificate
separately compares two against four increments and successive `h/L=0.18` and
`0.14` meshes. Its thresholds establish only successive-refinement stability;
they do not establish an asymptotic range, GCI, or numerical uncertainty.
The deterministic four-void RVE adds a distinct fixed-mesh 2/4/8-increment
certificate: all paths must pass invariant gates, their case identities may
differ only by increment count, and the final stress and physical-weighted
PEEQ statistics must stabilize. This closes that internal monotonic-path
regression axis without promoting the separate Zhang external benchmark.
The Zhang--Feng--Khandelwal external fixture now has a thin-3D tetrahedral
P2/DG0 mixed diagnostic in addition to the older low-order displacement-only
route. Neither is relabelled as the publication's 2D quadrilateral Q2/DPC1 9/3
element. One unarchived local 502-cell, 20-increment thin-3D diagnostic passed
the global first-Piola vector-norm tolerance but failed the componentwise
\(P_{11}\) and \(P_{22}\) contracts. Its condensed mixed `ELENER` was
0.002627074. Table 5 reports the primal Hencky elastic energy, so no relative
Table 5 energy error is assigned to that different channel and the
physical-energy gate remains incomplete. Those numbers are diagnostic
observations, not a content-addressed Golden. The exact Q9/DPC1 route can now
reconstruct the primal channel and publish the signed energy-gap decomposition,
but its current execution remains diagnostic and unpromoted. Moreover, the
current Table 5 assessor checks caller-supplied comparison-completeness flags;
until it consumes content-bound evidence records, it is not by itself a
scientific promotion gate. Promotion still requires load-increment/path and
spatial convergence, Table 5 stress, primal Hencky elastic energy and
current-state effective tangent, 1x1/1x2/2x1/2x2 replication invariance, and
serial/MPI plus restart evidence. An analytically linearized production
deviatoric tangent also remains a performance and conditioning gate: the
current mixed transformation removes the numerical volumetric tangent from the
complete discrete `dP/dF`, which is correctness-first rather than a
production extreme-bulk-modulus route.

## Nonlinear control layers

Three control levels remain distinct:

1. **Step control** advances load or time, accepts/cuts back increments, and
   owns termination limits.
2. **Global equilibrium** uses Newton iterations and a line search; the linear
   solver is a nested policy.
3. **Local constitutive integration** may use its own iterations and error
   estimate at each quadrature point.

A local material failure must be reported with material, region, cell,
quadrature point, and reason. The global controller may cut back the increment,
but it must not turn a local integration failure into NaNs or silently accept
an elastic substitute.

Future load control belongs beside displacement control at the step level.
Reaction recovery, arc-length methods, and contact each require separate
verified formulations; they are not flags on the current affine solver.
Automatic increment growth and cutback follow the same high-level separation
used by
[Abaqus/Standard static procedures](https://docs.software.vt.edu/abaqusv2024/English/SIMACAECAERefMap/simacae-t-simconfigurestatic.htm),
without claiming equivalence of the detailed controller.

## Output contracts

The result system distinguishes:

- **field output**: distributed values over saved frames;
- **history output**: selected values evolving over load/time;
- **diagnostics**: convergence and physical admissibility evidence;
- **presentation**: replaceable PNG/GIF/MP4 products;
- **scientific manifest**: model, solver, histories, quantities, and artifacts.

`E` follows the finite-strain convention and resolves to logarithmic strain
`LE`; `GREEN` requests Green--Lagrange strain explicitly. Visualization fields
may be cell samples. Authoritative RVE histories use variational integration
from the governing expressions and full-cell normalization.

The field/history distinction follows established CAE result semantics:
[Abaqus field and history output](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-c-simconcfieldhistory.htm)
describes fields as spatial distributions saved at relatively few states and
histories as frequent output from selected regions.

`reaction_field()` now exposes the unconstrained residual for ordinary strong
Dirichlet linear, nonlinear, and J2 problems. This is the correct first
building block, but it is not yet a universal reaction contract. Weak,
affine-MPC, and future contact constraints require their own verified
definitions. `diagnostics.mechanical_energy(...)` evaluates visible \(M/K\)
quadratic energies. J2 now integrates strong prescribed-displacement reaction
work and stores its balance against internal energy; natural loads, weak
constraints, affine MPCs, and transient procedures still need their own
verified work definitions.

## Verification ladder

Each new nonlinear family advances only with evidence:

1. analytical material-point tests and invalid-state tests;
2. one-element paths under multiple loading modes;
3. tangent verification by directional finite differences;
4. increment-size and mesh convergence;
5. rollback/cutback and restart equivalence;
6. an external benchmark with stated tolerances;
7. serial/MPI agreement and result-schema checks;
8. a readable public example and documented unsupported cases.

The first integrated stateful path is now small-strain 3D J2 isotropic
hardening under natural or strong displacement loading. A tabular amplitude
can load, unload, and reverse while the step coordinate remains monotone.
Forced cutback, cyclic state growth, work/energy histories, and restart of the
adaptive increment proposal are automated. Global implicit creep must reuse
this transaction and restart machinery rather than create a second state
store.

## Explicit non-goals for P1

- no claim of general contact, arbitrary multi-physics, or finite-strain
  plasticity beyond the gated experimental J2 providers; demonstrated
  locking-mitigation convergence and independent external structural validation
  remain promotion gates;
- no generic Abaqus deck execution;
- no UMAT compatibility before state, tangent, tensor-convention, and ABI
  gates exist;
- no constitutive name promoted from material-point maturity merely because a
  Python formula is present.

P1 succeeds when supported nonlinear solid analyses are easy to state,
difficult to misuse silently, inspectable during execution, and accompanied
by numerical evidence that survives refactoring and parallel execution.
