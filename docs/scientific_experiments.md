# Scientific Experiments

AgentFEM treats a family of related simulations as a scientific experiment,
not as a directory of unrelated scripts. The model remains ordinary readable
Python; the experiment layer records what changed, which cases succeeded, and
which evidence supports a derived response.

## Freeze the execution contract

Before a blind study or long campaign, capture the runtime:

```python
from agentfem import provenance

provenance.freeze_runtime("frozen_runtime.json")
```

Continue only under the declared policy:

```python
provenance.require_runtime("frozen_runtime.json", policy="error")
```

The lock records AgentFEM, Python, DOLFINx/UFL/Basix/FFCx, PETSc, MPI rank
count, scalar precision, and source or distribution identity. Absolute paths
remain diagnostic evidence but are not equality gates. Every written
`SimulationResult` manifest also carries the current runtime record before its
artifact seal is calculated.

A matching runtime does not establish mesh convergence, verification, or
physical validation. It only establishes that a frozen execution contract has
not silently changed.

## Declare and fingerprint scientific inputs

Campaigns fingerprint the parameter schema, declared outputs, builder,
evaluator, and every scientific input that exposes `to_ir()`, `as_dict()`, or
`summary()`. Declare source files as `Path` objects so their byte content is
included rather than only their filename:

```python
from pathlib import Path

campaign = campaigns.create(
    name="impact_scan",
    parameter_space=space,
    outputs=quantities,
    build=build_case,
    evaluate=solve_case,
    scientific_inputs={
        "mesh": Path("mesh/specimen.inp"),
        "material": material,
        "loading": loading,
        "observers": observer_plan,
    },
)
```

Each case also fingerprints its resolved parameters and built scientific
object. Arrays use shape/dtype/content hashes. An object without a scientific
identity contract is retained as an opaque type and makes fingerprint coverage
`incomplete`; AgentFEM does not pretend that the input has been captured.

The same contract is available for an individual analysis, without requiring
a campaign:

```python
result = step.solve_result(output="result.xdmf")
result.add_scientific_inputs(
    mesh=Path("mesh/specimen.inp"),
    material=material,
    loading=loading,
    observers=observer_plan,
)
result.write_manifest("result.json")
```

Attach source assets or public scientific objects, not only display labels.
The input manifest is included before the result is sealed, so changing a
source file changes both its input fingerprint and the final provenance seal.

## Run independent cases on local cores

Use spawned local processes for independent cases:

```python
campaign = campaigns.create(
    ...,
    execution=campaigns.local_processes(workers=8),
)
report = campaign.run(plan, output_directory="campaign")
```

Each worker constructs and solves a fresh case. Completed case files retain
worker process evidence and are reusable on restart. `spawn` is fixed by the
provider; AgentFEM does not fork an initialized MPI/PETSc process or use Python
threads as a FEM executor. The local-process provider is across-case
parallelism. It cannot be nested inside a within-case MPI communicator; use
deterministic plan shards for separate MPI jobs or a future scheduler provider.

## Issue a multi-axis convergence certificate

```python
from agentfem import convergence

certificate = convergence.audit(
    report,
    axes=(
        convergence.axis("mesh_size", fixed={"dt": 1.0e-6}),
        convergence.axis(
            "dt",
            fixed={"mesh_size": 2.5e-4},
            discretization="time_step",
        ),
    ),
    observables=(
        convergence.observable("event_time", tolerance=0.02),
        convergence.observable(
            "event_order",
            source="provenance",
            path="events.order",
            comparison="exact",
        ),
    ),
    output="convergence.json",
)
```

Every axis is an explicit one-at-a-time slice: all other varying parameters
must be fixed. Scalar/vector quantities use declared relative or absolute
criteria; event order and topology can require exact invariance. Failed,
missing, duplicated-resolution, shape-incompatible, or insufficient sequences
produce `inconclusive` evidence rather than a silently filtered curve.

## Compose a loading basis

Named loading modes can be reused across dynamics, control, inverse problems,
and parameter campaigns:

```python
from agentfem import amplitudes

loading = amplitudes.basis(
    amplitudes.gaussian_modulated_sine(
        amplitude=1.0,
        frequency=20_000.0,
        width=2.0e-5,
        center=8.0e-5,
        name="packet_1",
    ),
    amplitudes.gaussian_modulated_sine(
        amplitude=1.0,
        frequency=20_000.0,
        width=2.0e-5,
        center=1.4e-4,
        name="packet_2",
    ),
    coefficient_names=("a1", "a2"),
    value_unit="m",
)

history = loading.combine({"a1": 1.0, "a2": -0.25})
audit = history.audit(0.0, 4.0e-4)
```

The combined amplitude exposes value, velocity, acceleration, endpoint/range
evidence, JSON metadata, and a content fingerprint. Custom callables remain
usable but are reported as non-serializable. `scaled(...)`,
`time_shifted(...)`, and `time_scaled(...)` transform any reusable history
without introducing a case-specific packet type.

## Compute a response operator

A finite-difference response is lowered to the existing Campaign engine:

```python
from agentfem import campaigns, datasets, responses

space = campaigns.ParameterSpace.create(
    campaigns.RealParameter("a1", -1.0, 1.0, nominal=0.0),
    campaigns.RealParameter("a2", -1.0, 1.0, nominal=0.0),
)

operator = responses.finite_difference(
    parameter_space=space,
    baseline={"a1": 0.0, "a2": 0.0},
    outputs=(
        datasets.Quantity("event_time", unit="s"),
        datasets.Quantity("peak_traction", unit="Pa"),
    ),
    perturbation=0.05,
    step_mode="absolute",
    scheme="central",
)

campaign_report, response = operator.run(
    build=build_case,
    evaluate=solve_and_observe,
    output_directory="response_study",
)
```

Baseline and perturbed cases receive ordinary deterministic Campaign IDs. The
response report contains the Jacobian, singular values, rank, condition number,
a central-difference nonlinearity indicator evaluated separately for each
declared quantity, and the case IDs behind every column. The per-parameter
indicator is the maximum of those dimensionless quantity-level checks; unlike
physical units are never combined in one norm. A failed perturbation produces
an explicit incomplete report rather than a partial matrix.

The raw Jacobian retains derivative units for every output/parameter pair.
Singular values and condition number are reported automatically only when the
input and output unit families are homogeneous; mixed-unit conditioning needs
explicit nondimensional scales and is not silently interpreted.

Finite differences are the first provider because they work with existing
black-box FEM cases. Their cost grows with parameter dimension. Future
tangent-linear and adjoint providers will retain this response contract, but
irreversible damage and topology-changing events require separate mathematical
and verification treatment.

## Observe threshold events

```python
from agentfem import events

passage = events.first_passage(
    result.histories["opening"],
    threshold=critical_opening,
    direction="rising",
)
```

The event records the containing bracket and whether it was observed,
left-censored, or right-censored. For a continuous signal, linear localization
provides a useful sub-frame estimate. For damage jumps, contact activation, or
other discontinuous changes, use the bracket to request local substepping.

## Scope

The core owns scientific identities, experiment plans, event semantics, and
result evidence. Specialized actuator families, publication figures, control
objectives, and confidential calibration assets belong in research or private
extension packages. This keeps the public platform useful without turning one
research programme into the global API.
