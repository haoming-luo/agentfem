# Learned constitutive models

AgentFEM owns the scientific and solver contract; an installed extension owns
the executable model runtime. This separation keeps the finite-element core
usable without PyTorch and lets laboratory, commercial, or future framework
providers share one material lifecycle.

```text
AgentFEM core
  named parameters + tensor convention + state schema
  batch material-point request/response
  trial / commit / rollback + portable quadrature state
  provider discovery + fail-closed compatibility checks

provider package
  artifact loading + device + dtype + framework
  batched inference + consistent tangent
  architecture-specific state encoding and diagnostics
```

The declaration is immutable and contains no executable model:

```python
from agentfem import constitutive, learning, materials

spec = learning.learned_constitutive(
    provider="agentfem-learning.torch-constitutive",
    architecture_id="laboratory_model.v1",
    artifact="materials/laboratory_model",
    revision="fixed-model-revision",
    artifact_sha256="...",
    tangent_convention=constitutive.MaterialTangentConvention.cauchy_small_strain(),
    parameter_schema=(
        learning.MaterialParameterSpec("young", "Pa", minimum=0.0),
        learning.MaterialParameterSpec("poisson", "1", minimum=-1.0, maximum=0.5),
    ),
    parameters={"young": 190.0e9, "poisson": 0.3},
    state_schema=my_state_schema,
)
material = materials.learned(spec)
```

The material then enters the same nonlinear solid workflow as a native
stateful material:

```python
model.material(material)
step = model.step(target=displacement, material=material)
result = step.solve_result()
```

The first global route supports 3D and 2D plane strain, strong Dirichlet
constraints, natural loads, batched local updates, cutback, MPI-atomic state,
and MPI-portable checkpoint/restart. Models that require temperature, rate, or
field variables are rejected until the selected procedure supplies those
inputs. A model without a provider-verified consistent tangent remains a
material-point capability and cannot enter implicit equilibrium.

The extension must be explicitly activated before `materials.learned(...)`.
Loading fails if the provider, artifact, checksum, state layout, tensor order,
shear convention, parameters, or tangent capability differ from the
declaration. No fallback or extrapolation is hidden.

## Small-strain contract

The canonical material coordinate is

```text
strain_old, strain_new, state_old, named parameters
  -> Cauchy stress, d sigma / d epsilon, state_new, energy, diagnostics
```

The standard Voigt order is `xx, yy, zz, xy, yz, xz`; the shear convention is
always declared. A production provider implements `update_batch(...)` once per
MPI rank. `update(...)` remains a compatibility and material-point inspection
path.

AgentFEM validates the complete response before changing trial state. A failure
on any MPI rank rolls every rank back. Newton failure discards trial state;
global convergence is the only point at which state may be committed.

Provider-declared storage and dissipation channels remain distinct
quadrature fields in `SimulationResult`; runtime evidence records the provider,
artifact and dataset revisions, checksums, dtype, device, framework, tangent
method, loading time, applicability counts, retries, and inference timing.

The provider reports `in_domain`, `warning`, `out_of_domain`, or
`invalid_state` per integration point. Execution success is not scientific
validation, and an out-of-domain point is never silently replaced with an old
or zero stress.

## Artifact preparation

Solves are offline. A remote model registry may be used in a separate,
explicit preparation command, but the finite-element run consumes a verified
local bundle. A bundle contains a manifest, safe tensor weights, checksums, and
human-readable provenance. Pickled executable objects are not a runtime
format.

The first maintained provider and reference architecture live in
`agentfem-learning`. Private providers use the same extension point; AgentFEM
core does not contain a Torch type, a network name, or an architecture-specific
state count.
