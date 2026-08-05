# AI-Native Campaigns and Scientific Learning

This document defines the first AgentFEM interface from finite-element
simulation collections to surrogate, reduced-order, neural-operator, and
physics-informed workflows.

The central decision is:

> AgentFEM does not become a machine-learning framework. It provides the
> scientific contract that makes learning from finite-element analyses
> traceable, reviewable, and safe to reconnect to simulation.

PyTorch, scikit-learn, Gaussian-process libraries, neural-operator packages,
and future training services may supply learning algorithms. AgentFEM owns the
meaning of inputs and outputs, case identity, simulation evidence, validation,
applicability, and high-fidelity fallback.

## The End-to-End Contract

```text
ParameterSpace
      |
      v
SamplingPlan -> CampaignPlan -> immutable case variants
                                    |
                                    v
                         FEniCSx/PETSc execution
                                    |
                                    v
                      case records + AF-IR provenance
                                    |
                                    v
                         ScientificDataset
                           /      |       \
                          v       v        v
                    classical   neural   external
                    ROM/QoI     models   trainers
                          \       |        /
                           v      v       v
                    independent validation
                              |
                              v
                   applicability-domain guard
                       /                   \
                      v                     v
             surrogate prediction     FEM fallback
```

This separation matters. A folder of arrays is not yet a scientific dataset,
and a trained network is not yet a reliable simulation asset.

## Implemented First Phase

### Typed parameter spaces

`agentfem.campaigns` provides:

- `RealParameter` with bounds, units, nominal value, and linear/log scale;
- `IntegerParameter`;
- `ChoiceParameter`;
- ordered `ParameterSpace`;
- explicit, random-uniform, Latin-hypercube, and full-factorial plans.

Every sample is validated before it becomes a case. Sampling is reproducible
from a seed. Case IDs are hashes of the campaign identity, schema version, and
canonical parameter values.

```python
from agentfem import campaigns

space = campaigns.ParameterSpace.create(
    campaigns.RealParameter("young", 150e9, 250e9, unit="Pa"),
    campaigns.RealParameter("traction_y", -2e6, -0.2e6, unit="Pa"),
    name="cantilever_design",
)
sampling = campaigns.latin_hypercube(space, 64, seed=2026)
```

### Campaign execution

A campaign requires two small functions:

- `build(parameters)` creates a fresh case;
- `evaluate(case)` executes it and returns declared quantities.

Fresh construction is deliberate. Mutating one live FEniCSx model repeatedly
can leak state, compiled objects, boundary conditions, or solver history from
one sample into another. A factory makes variants immutable by construction
even though runtime objects themselves remain mutable.

```python
campaign = campaigns.create(
    name="cantilever_sweep",
    parameter_space=space,
    outputs=(
        datasets.Quantity("tip_displacement", unit="m"),
    ),
    build=build_case,
    evaluate=evaluate_case,
)
report = campaign.run(
    sampling,
    output_directory="campaign_output",
    comm=MPI.COMM_WORLD,
)
```

The first runner is intentionally serial across cases. When an MPI
communicator is provided, all ranks cooperate on each FEniCSx solve while rank
zero alone writes evidence. This is **within-case parallelism**.
Case completion is agreed across ranks: a failure reported by any rank becomes
a failed campaign record rather than being hidden by rank zero's outcome.

For **case-level parallelism**, `CampaignPlan.shard(index, count)` creates
deterministic disjoint plans for separate MPI jobs, scheduler allocations, or
services. AgentFEM does not use Python threads as an implicit FEM executor.
MPI communicators, PETSc state, JIT compilation, and filesystem output make
that shortcut unsafe.

Completed case records are resumed by deterministic ID. Failed cases remain
explicit records with error type and message. Although successful cases can be
materialized as a partial dataset, `report.require_dataset()` refuses to feed
that dataset downstream by default until failures are reviewed and
`allow_partial=True` is chosen explicitly. The returned dataset records that
acceptance and the failed case IDs in its metadata.

### Scientific datasets

`agentfem.datasets` retains:

- the parameter schema and units;
- named scalar, curve, vector, or sampled-field outputs;
- exact output shapes;
- field-encoding metadata;
- case IDs;
- provenance, including available AF-IR model records;
- artifact links;
- deterministic train/validation splits.

Numeric arrays are stored in compressed NPZ form. A JSON manifest preserves
their scientific interpretation. Continuous and integer inputs use normalized
parameter coordinates; categorical inputs use explicit one-hot features rather
than an invented ordinal distance. The manifest records the feature names and
encoding so every value round-trips to its declared parameter.

```python
dataset = report.require_dataset(minimum_samples=4)
split = dataset.split(validation_fraction=0.2, seed=2026)
dataset.write("campaign_dataset")
restored = datasets.ScientificDataset.read("campaign_dataset")
```

When PyTorch is installed, the same reviewed dataset becomes an ordinary
`TensorDataset`/`DataLoader` without losing its scientific column schema:

```python
bundle = dataset.to_torch()
loader = bundle.loader(batch_size=64, seed=2026)
```

For field learning, `datasets.fem_field_sample(field, encoding)` exports owned
nodal coefficients and coordinates under an explicit `FieldEncoding`. The
first adapter is deliberately serial and `mesh_dofs` only. It rejects a
distributed concatenation until global dof identities and a partition
manifest exist; it also refuses to call unstructured dofs a structured FNO
grid. PyTorch remains responsible for tensors, autodiff, optimization, and
model architecture.

### Transparent baselines before neural complexity

The first built-in models are:

- `RidgeSurrogate` for parameter-to-QoI and small vector outputs;
- `PODRidgeSurrogate` for curves and sampled fields;
- optional `TorchMLPSurrogate` for parameter-to-QoI/vector mappings.

The ridge and POD baselines depend only on NumPy and write portable manifests
plus numerical state. They are not included because linear models are always
adequate. They provide a transparent reference that more complex models should
outperform on independent evidence.

```python
trained = surrogates.PODRidgeSurrogate(
    energy=0.999,
    max_modes=32,
).fit(split.train)

validation = trained.validate(
    split.validation,
    thresholds={"max_relative_l2": 0.02},
)
print(validation.format())
trained.write("trained_surrogate")
```

The optional PyTorch MLP imports PyTorch only during training. AgentFEM records
its architecture and scientific schema, but PyTorch remains responsible for
autodifferentiation and optimization. The first adapter is an in-memory
training template; portable, non-pickle model export remains a subsequent
adapter task.

The current residual-scale uncertainty is explicitly labeled as such. It is
not presented as epistemic uncertainty. Future Gaussian-process, ensemble, or
Bayesian adapters may provide stronger uncertainty estimates through the same
prediction contract.

### Applicability and high-fidelity fallback

A surrogate should not extrapolate merely because a tensor operation permits
it. `BoxApplicabilityDomain` records a first conservative envelope in
normalized parameter space. `GuardedSurrogate` either predicts inside that
domain, rejects the request, or invokes an explicit high-fidelity fallback.

```python
domain = surrogates.BoxApplicabilityDomain.from_dataset(split.train)
guarded = surrogates.GuardedSurrogate(
    trained,
    domain,
    fallback=run_one_fenicsx_case,
)
prediction = guarded.predict(candidate)
print(prediction.source, prediction.in_domain)
```

The box is only a first guard. It does not detect holes, sparse corners,
geometry changes, phase changes, bifurcations, or unrepresented boundary
conditions. Future domains should combine distance/density models, categorical
compatibility, physics diagnostics, and uncertainty. The public API already
makes the decision visible. Categorical inputs are stricter than padded numeric
bounds: a category absent from training is out of domain even when numeric
padding is requested.

For the common split-fit-validate-guard sequence, a small convenience workflow
keeps the evidence together without taking training ownership away from the
estimator:

```python
training = surrogates.train(
    dataset,
    estimator=surrogates.RidgeSurrogate(),
    validation_fraction=0.2,
    seed=2026,
)
guarded = training.guard(fallback=run_one_fenicsx_case)
print(training.validation.format())
```

## Three Roles for Learned Models

### 1. Substitute within a declared domain

The learned model replaces repeated FEM solves only for a declared mapping and
applicability domain. This is the common use in optimization, uncertainty
quantification, real-time estimation, and interactive design.

### 2. Accelerator inside FEM

A learned component may improve, rather than replace, deterministic solves:

- initial guesses;
- preconditioner or reduced-space selection;
- local constitutive updates;
- closure terms for multiscale models;
- mesh/adaptivity indicators;
- error estimators.

These integrations require backend-specific evidence and are not implemented
by the current surrogate module.

### 3. Hybrid decision system

The surrogate screens many candidates. Cases with high uncertainty, physical
diagnostic failure, or out-of-domain inputs return to FEM. Their new evidence
can be reviewed and appended to a later training campaign. This is the natural
route to active learning without treating every automatic retraining event as
scientifically approved.

## Neural Operators

A neural operator learns a map between functions rather than only between a
small parameter vector and scalar quantities. Examples include:

```text
material/load/boundary fields -> displacement or temperature field
initial state + forcing history -> transient solution field
geometry encoding + coefficients -> PDE solution field
```

The difficult part is not naming FNO, DeepONet, graph neural operators, or mesh
networks. It is specifying:

- input/output field units and components;
- mesh, grid, graph, sensor, or basis encoding;
- geometry and coordinate encoding;
- boundary-condition representation;
- projection between FEM spaces and learning tensors;
- treatment of changing meshes;
- held-out field error;
- boundary and conservation/balance errors;
- out-of-distribution behavior.

`FieldEncoding` and `NeuralOperatorSpec` define the learning contract.
`ObservationGrid` and `datasets.fem_observation_sample(...)` now make the
structured-grid branch executable in serial and MPI, including coordinates,
layout, units, and an optional geometry mask. An elementary FNO specification
is rejected when its fields are not represented on structured grids. This
avoids attaching a fashionable architecture name to incompatible data.

Production neural-operator trainers remain external. Planned adapters should
consume the same dataset and write a model artifact with the same validation
and applicability evidence.

## PINNs and Physics-Informed Learning

PINNs do not serve the same role as ordinary surrogates.

For selected equations, `TorchPINNAdapter` makes a reviewed `PINNSpec`
executable without pretending to translate arbitrary UFL. Residual and
condition callables use ordinary PyTorch/autograd; the adapter verifies that
their names exactly match the scientific contract, applies declared weights,
reports each loss contribution, and can run a minimal Adam loop. Network
architecture, autodiff, and optimization remain PyTorch responsibilities.

This changes the boundary from “PINN vocabulary only” to “one-stop execution
for explicitly bound equations.” Automatic UFL-to-PINN translation,
identifiability, collocation adequacy, and independent FEM validation remain
separate evidence obligations.

They are most attractive for:

- inverse parameter or source identification;
- sparse observation/data assimilation;
- data--physics fusion;
- selected problems where a differentiable residual is explicit and tractable.

They are not a default replacement for established FEM in complex geometry,
discontinuous media, high-frequency waves, contact, strong nonlinearity, or
industrial-scale systems.

Most importantly, an arbitrary UFL weak form cannot be assumed to become a
correct strong-form PINN residual. Boundary terms, regularity, discontinuities,
constitutive state, and integration-by-parts choices carry scientific meaning.

`PhysicsResidual`, `PhysicsCondition`, and `PINNSpec` therefore require an
explicit strong, weak, or discrete residual and explicit conditions. A spec
remains `contract_only` until every named term is bound to reviewed executable
callables. `TorchPINNAdapter` provides that binding mechanism; reusable
strong-, weak-, and discrete-residual libraries remain future work.

## What Is Not Yet Claimed

The first phase does not claim:

- asynchronous scheduling;
- Slurm/Kubernetes/cloud execution;
- automatic graph construction or basis encoding for arbitrary meshes;
- a production neural-operator trainer;
- a general UFL-to-PINN compiler;
- calibrated epistemic uncertainty;
- automatic active-learning approval;
- arbitrary model mutation or AF-IR round-trip reconstruction;
- surrogate validity outside independent tests and declared domains.

These omissions are public boundaries, not hidden placeholders.

## Next Implementation Stages

1. Add a formal execution-service protocol so local shards, Slurm jobs, and
   hosted runners write the same case record.
2. Link each dataset sample to a content-addressed AF-IR document and run
   record rather than embedding large records.
3. Build on implemented probes, integrals, strong-constraint resultants,
   projected fields, and structured observation grids with affine/weak
   reactions, broader energy curves, graph encodings, and reduced bases.
4. Add dataset merge/deduplication and explicit training/validation/test
   partitions.
5. Add Gaussian-process and uncertainty-calibrated ensemble adapters.
6. Add active-learning proposal records with human approval policy.
7. Add graph/basis field encodings and reviewed NeuralOperator data-processor
   adapters without taking ownership of external model architectures.
8. Add selected, reviewed physics-residual libraries on top of the executable
   PyTorch binding adapter.
9. Expose campaign planning, status, diagnostics, comparison, and artifact
   retrieval through tool-service/MCP operations.

## Reference Example

Run:

```bash
python examples/static_elasticity_surrogate_campaign.py
```

The example performs actual FEniCSx linear-elasticity cases, resumes completed
case records, builds a scientific dataset, trains and independently validates a
ridge baseline, writes the model artifact, and creates a guarded predictor with
FEM fallback.
