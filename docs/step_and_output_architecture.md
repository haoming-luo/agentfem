# Stable Steps and Compact Field Output

This note records two architectural decisions made while turning the Abaqus
C3D10 periodic-cell migration into reusable AgentFEM infrastructure.

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

The default registry currently contains linear-static operator,
Neo-Hookean finite-strain static, and central-difference explicit providers.
The registry is public and inspectable through `models.step_providers()`.

This is an extension seam, not a promise that arbitrary registered code is
scientifically valid. A new plasticity or creep provider is admissible only
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
    solver_options=AffineNewtonOptions(max_it=25),
)
```

Thus `max_increments` limits accepted load/time advances, whereas `max_it`
limits Newton iterations in one attempt.

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

The directly deformed compact writer is serial. Under MPI, AgentFEM delegates
the scientific field history to DOLFINx's collective XDMF/HDF5 writer rather
than gathering a large distributed mesh to rank zero. That MPI product retains
the reference mesh, `U`, and requested cell fields; direct `x+u` presentation
is rendered later as a serial postprocess.

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

## Next gates

1. Extend collective MPI output from reference-configuration scientific fields
   to an ownership-safe directly deformed geometry product.
2. Add named point, region, reaction, and energy history extractors.
3. Introduce restartable quadrature state with trial/commit/rollback.
4. Integrate J2 only with a consistent tangent, cutback, and one-element path
   benchmarks.
5. Build a global implicit creep provider only after the same state and
   rollback machinery exists.
6. Let fatigue consume verified extracted histories; add a fatigue field only
   after hotspot/element mapping and provenance are defined.
7. Add load-controlled finite-strain examples for dead and follower pressure,
   including tangent and sign checks.
