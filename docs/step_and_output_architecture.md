# Stable Steps and Compact Field Output

This note records two architectural decisions made while turning the Abaqus
C3D10H periodic-cell workflow into reusable AgentFEM infrastructure.

## Stable public step, extensible lowering

The public language is:

```python
from agentfem import steps

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
)
```

It should not become `model.neo_hookean_step`, `model.j2_step`,
`model.creep_step`, and one more method for every future constitutive law.
Those names mix two distinct decisions:

1. the Study selects the analysis family and physics;
2. a step provider decides whether one implemented material protocol can be
   lowered to an executable problem.

`step_providers.py` now owns that second decision. A provider declares:

- accepted analysis names;
- a read-only compatibility predicate;
- a lowering function;
- priority and a human-readable description.

The default registry currently contains linear-static, implicit-Euler heat,
finite-strain hyperelastic, 3D stateful J2 and creep,
Newmark/generalized-alpha implicit dynamics, and central-difference explicit
providers.
The registry is public and inspectable through `models.step_providers()`.

Each built-in provider also owns a `StepOptionContract`. This keeps the single
`model.step(...)` entry point extensible without turning its `**options`
boundary into an unchecked dictionary. Misspelled or procedure-inappropriate
keywords fail before form assembly with a repair suggestion; required physical
inputs such as `dt`, `steps`, or `duration` are checked at lowering. The same
contract is emitted by `agentfem capabilities --json`, so command-line agents,
IDEs, and future GUIs inspect exactly what runtime execution enforces.

`model.step(...)` normalizes this call into an immutable `StepRequest`. The
request carries the resolved `SolutionProcedure`, target, material reference,
and a read-only option mapping through both `step_capability(...)` and provider
lowering. An explicit `procedure=` may override the Study preference, but a
conflicting `method=` or incompatible equation order fails before assembly.
This is the first typed boundary behind the readable Python language; it does
not add a second user-facing configuration system.

The request also exposes an immutable `StepExecutionPolicy` containing the
declared solver, output, transient-history, progress, and checkpoint controls.
Individual keywords remain the public language; the policy is their common
inspection and provenance form. A Step execution context retains it, and a
completed result records the JSON-safe summary under
`metadata["execution_context"]["policies"]`. `None` means the selected provider
used its own default; resolved time-step and solver details remain visible in
the executable Step summary.

The provider registry now calls internal scientific builders directly for
linear static/steady conduction, J2 plasticity, and implicit creep. Their old
`Model` methods are compatibility delegates. This establishes the migration
pattern for hyperelasticity and dynamics without changing a case file:

```text
model.step(...) -> StepRequest -> provider -> scientific builder -> problem
                                  |                         |
                                  + execution policy ------+-> result lifecycle
```

This is an extension seam, not a promise that arbitrary registered code is
scientifically valid. A new stateful material provider is admissible only
after its constitutive state, trial/commit/rollback policy, tangent, increment
control, output, and benchmarks exist. Until then, the material-point law stays
truthfully below the FEM-integrated maturity level.

## Step, increment, iteration, and frame

These words are not interchangeable:

- a **Step** is one analysis procedure or loading stage;
- an **Increment** advances load or time within a Step;
- an **Iteration** is one Newton equilibrium correction inside an Increment;
- an **Attempt** is one try at an Increment, including retries after cutback;
- a **Frame** is a saved result state, not a unit of nonlinear solution.

Automatic incrementation is the default for the affine finite-strain path.
`max_increments=10` means “use no more than ten accepted increments”; it does
not request exactly ten. Exact fixed subdivision remains explicit:

```python
step = model.step(..., incrementation=steps.fixed(10))
```

The automatic controller increases the next increment after fast convergence,
keeps or reduces it after harder convergence, and rolls the solution back
before retrying a failed attempt with a cutback. `minimum`, `maximum`,
`max_increments`, and `max_cutbacks` are termination safeguards. Constitutive
models with internal variables must implement trial/commit/rollback before
using the same controller globally.

Solver convergence remains a separate, visible choice:

```python
step = model.step(
    ...,
    incrementation=steps.automatic(max_increments=100),
    solver_options=solvers.newton(
        maximum_iterations=25,
        linear_solver=solvers.direct_solver(),
    ),
)
```

Thus `max_increments` limits accepted load/time advances, whereas `max_it`
limits Newton iterations in one attempt.

`solvers.newton(...)` is independent of the constraint implementation.
AgentFEM translates the same public policy to PETSc SNES for ordinary boundary
conditions or to exact affine-reduction algebra for periodic equations.
Backend-specific classes remain available for expert tuning but are not the
standard model language.

## Output requests do not control the nonlinear algorithm

Saving every accepted increment follows the Abaqus/Standard convention:

```python
output = results.field_output("U", "S", "E", every="increment")
```

Uniform output marks use output intervals:

```python
output = results.field_output("U", "S", "E", intervals=6)
step = model.step(..., output=output)
```

With `intervals=6`, the initial state and six interval endpoints form seven
result frames. The nonlinear controller lands on those requested marks and may
insert additional internal increments after cutbacks. `frames` is reserved for
the states read back from a result dataset; it is not a solve-control argument.

Amplitude coordinates follow the same separation. A natural load without an
amplitude receives the nonlinear Step's proportional load factor. A load with
an amplitude evaluates that history at normalized step time and is not scaled
again. Linear static evaluates model histories at step end, while heat and
dynamics evaluate them at physical time.

## Standard run feedback

Nonlinear steps report progress on MPI rank zero without requiring routine CLI
flags:

```text
[STEP 1] periodic_neo_hookean | automatic_incrementation
  [INC 1 | ATT 1] 0 -> 0.25 (d=0.25)
    ITER 01 | residual=...
  [INC 1] CONVERGED | iterations=...
```

Passing `status_file="job.sta"` adds a flushed, line-oriented status record.
`progress=False` remains available for managed campaigns, but silence is not
the interactive default.

The same event is delivered to a complete in-memory trace before display
filtering. Consequently `print_every=50` reduces terminal traffic but does not
erase the other 49 accepted increments from the result manifest. Heat,
explicit dynamics, implicit dynamics, and J2 use this same evidence contract.

## One logical field dataset

The default finite-strain result consists of:

```text
results.xdmf   small temporal/field description
results.h5     compressed numerical payload
```

The HDF5 payload stores topology once, retains reference coordinates, and
stores each frame's deformed coordinates plus all requested point and cell
attributes. The XDMF presents those frames as one temporal collection. ParaView
can therefore:

- discover time/load values;
- play the true `x+u` geometry directly;
- switch `U`, stress, strain, `J`, energy, and other fields on the same grid;
- avoid one VTU file per frame and avoid a manual Warp filter.

XDMF and HDF5 are one logical dataset. Putting the heavy arrays inline in XML
would create a much larger, slower text file without improving scientific
meaning. PVD/VTU remains an optional compatibility backend, not the default.

Completed static elasticity, J2, and creep results use the same
`results.write_result_fields(...)` implementation. J2 and creep retain raw
integration-point `S/PE/CE` state in `SimulationResult`; the common writer
records that those fields were omitted from ordinary visualization attributes
and writes the explicitly recovered `*_CELL` fields instead. No hidden nodal
extrapolation or smoothing is introduced by asking for `output=`.

The directly deformed compact writer is serial. Under MPI, AgentFEM retains two
explicit products. DOLFINx collective XDMF/HDF5 is the compact scientific
route. `io.ParaViewTimeSeries` and
`results.write_parallel_vtk_series(...)` are the presentation route: each
saved time is one parallel unstructured-grid dataset carrying `U` as PointData
and DG0 stress/strain/state as CellData. ParaView can apply one Warp By Vector
filter without retaining duplicate unwarped Blocks. The PVD/PVTU/VTU family
uses multiple piece files and reference geometry, so it does not replace the
compact checkpoint/scientific store.

Arbitrary chained Abaqus `*EQUATION` constraints have a distributed
`dolfinx_mpc` backend. It resolves the source equation graph before mapping
labels to global dofs, explicitly supplies relations for owned and ghost
slaves, assembles constrained operators collectively, and back-substitutes
Newton corrections. The dependency is optional, compiled, version-matched to
DOLFINx, and never silently emulated when unavailable.

## Result roles remain distinct

- A field series answers “where is a quantity distributed?”
- A history answers “how did a selected value evolve?”
- A QoI answers “what value enters a report, decision, campaign, or dataset?”

The unified XDMF does not replace exact homogenized histories or training
tables. Visualization fields may be projected to cell centers, whereas the
periodic-cell NPZ/CSV response is integrated from the governing UFL
expressions. Consumers should not reconstruct authoritative macro response by
re-averaging a visualization field.

The complete declaration can collect these distinct roles:

```python
output = results.output_plan(
    output_directory,
    field=results.field_output("U", "S", "E", "J", every="increment"),
    requests=(
        results.solver_history(),
        results.periodic_cell_history(periodicity),
        results.source_node_history(nodes, RIGHT=7, TOP=9),
        results.finite_strain_checks(constraint=periodicity),
    ),
    presentation=results.presentation(animation="gif"),
)
```

The plan is passed into `model.step(...)` so exact output marks constrain only
where the nonlinear path must land. After the solve, `finalize(...)` writes
fields, evaluates histories and diagnostics, attaches artifacts, and writes
the model record and result manifest. A case-specific response plot remains in
the case directory rather than becoming a misleading general result function.

## Implemented adjacent foundations

- automatic affine finite-strain increments with convergence-based growth,
  rollback, cutback, accepted/attempt histories, and `.sta` reporting;
- axis-aligned solid `symmetry`/`roller` constraints;
- reference dead pressure and current follower pressure using Nanson pullback;
- exact piecewise-constant material-point creep histories;
- auditable fatigue assessment from a named `SimulationResult` history.

These additions are intentionally reusable. They do not turn material-point
creep into a global creep solver or scalar S-N fatigue into a general
multiaxial damage model.

## Accepted-increment checkpoint policy

All three transient routes—central difference, Newmark/generalized-alpha, and
implicit-Euler heat transfer—consume the same policy object:

```python
from agentfem import checkpointing

step = model.step(
    target=field,
    dt=dt,
    steps=1000,
    checkpoint=checkpointing.every(
        100,
        directory="checkpoints",
        final=True,
        keep_last=3,
    ),
)
```

`checkpointing.every(..., portable=True)` or
`step.save_checkpoint(..., portable=True)` additionally writes a global,
physical-node-keyed nodal state. It can be read with a different MPI partition
or rank count; two-rank heat and split-interface Explicit states are
automatically tested by continuation on one rank. The default remains the
smaller rank-sharded format. Portable state uses bounds-scaled integer
coordinate keys to absorb partition-dependent mesh-construction roundoff while
retaining a global connectivity identity. Coincident but independent interface
nodes are disambiguated by durable source input-node identity. The current NPZ
implementation is suitable for laboratory-scale restart; a
collective HDF5 implementation is the later scale path. J2 and creep
quadrature state still require stable cell-and-point keys before they can use
the same portability claim.

A checkpoint is written only after an increment has been accepted, its state
and scientific histories have been committed, and its execution event has been
recorded. The final state is retained even when the total increment count is
not divisible by the cadence. `keep_last` bounds scheduled-checkpoint storage;
AgentFEM publishes the new manifest and all rank shards before deleting older
generations, removes only files named by their manifests, and preserves an
explicit restart-source record. Omitting `keep_last` retains every scheduled
checkpoint. Cross-partition identity remains a separate future capability.

## Next gates

1. Extend collective MPI output from reference-configuration scientific fields
   to an ownership-safe directly deformed geometry product.
2. Extend the implemented MPI-global strong-BC resultant and proportional
   linear-static energy closure into named region histories; define
   affine/weak reactions and non-zero prescribed-displacement work separately.
3. Extend the implemented cumulative J2 serial checkpoint and scientific
   identity to portable MPI cell identity and multi-region state.
4. Extend the implemented analytical J2 Golden path, physical forced cutback,
   cyclic amplitude, quadrature S/PE/PEEQ/MISES and nodal RF fields, and strong-displacement work/energy
   history with projected visualization, mesh convergence, natural/affine work
   definitions, mesh convergence, and full external-deck reproduction; the
   published Abaqus homogeneous uniaxial constitutive state is now automated.
5. Build global implicit creep by consuming the shared
   `QuadratureTransaction`, adaptive controller state, and rollback machinery.
6. Let fatigue consume verified extracted histories; add a fatigue field only
   after hotspot/element mapping and provenance are defined.
7. Add load-controlled finite-strain examples for dead and follower pressure,
   including tangent and sign checks.
8. Build on the implemented shared transient energy/checkpoint lifecycle,
   accepted-increment automatic cadence, and heat-balance history with broader
   mechanical work terms, retention policy, and cross-partition MPI portability.
