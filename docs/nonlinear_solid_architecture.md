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
foundation only: finite-strain J2 promotion still requires material paths,
numerical tangent comparison, a consuming global solve, cutback/restart
equivalence, MPI-stable state identity, and an external structural benchmark.

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
  plasticity;
- no generic Abaqus deck execution;
- no UMAT compatibility before state, tangent, tensor-convention, and ABI
  gates exist;
- no constitutive name promoted from material-point maturity merely because a
  Python formula is present.

P1 succeeds when supported nonlinear solid analyses are easy to state,
difficult to misuse silently, inspectable during execution, and accompanied
by numerical evidence that survives refactoring and parallel execution.
