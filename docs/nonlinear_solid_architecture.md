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
Finite-strain J2 now has material paths, numerical tangent comparison, a
public three-dimensional affine-periodic `model.step(...)` route,
cutback/restart equivalence, and MPI-stable state identity. That public route
is experimental: an independent external structural benchmark still gates a
broader engineering maturity claim.

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

This is currently an **experimental public affine-periodic finite-element
capability**, as well as a material-point and neutral quadrature provider.
Rigid rotation, superposed rotation, plastic incompressibility, yield
consistency, unloading/reversal, tangent comparison, atomic rollback and
commit are executable tests. `model.step(...)` lowers one or more explicitly
partitioned compatible 3D `FiniteStrainJ2Logarithmic` materials and exactly one
`AbaqusPeriodicConstraint` to a total-Lagrangian residual assembled from `P`
and `dP/dF`. It uses exact affine elimination in serial or the distributed
`dolfinx_mpc` reduction, accepts fixed or automatic increments, and performs a
real rollback/cutback when an otherwise converged PEEQ increment is excessive.
The older `mechanics.experimental_finite_strain_j2_step(...)` remains a
compatibility/development entry point rather than the recommended application
language.

The state transaction owns accepted quadrature `F`, `P`, `S`, `MISES`,
`SENER`, `ELENER`, `HARDENER`, `FP`, and `PEEQ`. Scientific output uses those
provider-owned fields;
it does not reconstruct an inelastic response from a stateless hyperelastic
formula. Explicitly named `*_CELL` fields are physical quadrature-weighted DG0
averages for visualization and do not replace the integration-point evidence.

For this J2 provider, `ELENER` is the quadratic Hencky elastic free-energy
density and `HARDENER` is
\(\tfrac12 H\bar\varepsilon_p^2\). The backward-compatible `SENER` field is
their sum. None of these names denotes plastic dissipation. A complete
dissipation balance still requires accepted-increment stress power and an
explicit cumulative ledger; it is intentionally not inferred from the final
state alone.

Portable checkpoints are accepted-state boundaries. They store `U`,
`U_ACCEPTED`, committed quadrature state, accepted and attempted increment
histories, the next adaptive increment, and execution events. Restore validates
the mesh/function identity, material and state schema, quadrature rule,
increment control, and periodic equations before changing the analysis. The
same checkpoint has been resumed between one and two MPI ranks in both
directions.

The present public scope is deliberately narrow: prescribed macroscopic
deformation, compatible regional materials, one periodic constraint, and no
body-force or natural-load power. The true spherical-void RVE now exercises
geometric pairing, positive-J, public result lifecycle, two-rank execution,
and Hill--Mandel evidence. Its versioned fixed-stack Golden additionally
freezes one `h/L=0.25` first-order mesh, two-increment loading path, runtime
stack, and portable mesh identity. It is a software-regression contract for
macroscopic first-Piola stress, physical-weighted PEEQ statistics, and solid
fraction, not a mesh-converged RVE reference value. An opt-in certificate
separately compares two against four increments and successive `h/L=0.18` and
`0.14` meshes. Its thresholds establish only successive-refinement stability;
they do not establish an asymptotic range, GCI, or numerical uncertainty.
The Zhang--Feng--Khandelwal external fixture remains fail-closed because the
current low-order displacement-only tetrahedral route has not matched the
published mixed displacement--pressure result. A locking-resistant mixed
formulation and an analytically linearized production tangent remain promotion
gates. The current numerical `dP/dF` is a correctness-first discrete
derivative, not a production-performance claim.

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

- no claim of general contact, arbitrary multi-physics, or globally integrated
  finite-strain plasticity beyond the gated serial patch;
- no generic Abaqus deck execution;
- no UMAT compatibility before state, tangent, tensor-convention, and ABI
  gates exist;
- no constitutive name promoted from material-point maturity merely because a
  Python formula is present.

P1 succeeds when supported nonlinear solid analyses are easy to state,
difficult to misuse silently, inspectable during execution, and accompanied
by numerical evidence that survives refactoring and parallel execution.
