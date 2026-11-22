---
title: "Check, run and compare AgentFEM simulation results"
description: "Use the AgentFEM CLI to inspect projects, run cases, locate output files and compare declared quantities or histories with explicit tolerances."
---

# Check, run, inspect, compare

## Review the model, settings and resource budget

The three result templates support `python case.py --review` in the installed
environment. This prints the actual model card without creating a run or
solving a Step. It is a case-level entry, not `agentfem run -- --review`.
Boundary measures may compile/integrate forms; ordinary `agentfem check`
keeps its non-executing structure/runtime meaning.

```python
card = model.review(target=u, step_options=options,
    output_fields=(u, "S"), output_frames=5, output_directory="outputs")
```

The card reports element interpolation, registered independent unknown DOFs,
owned boundary entity counts and ranges, material assignments, physical load
conventions and a resource budget. Node-coordinate bounds do not establish
extrema between curved geometry nodes. Registered DOFs are not a prediction
of the assembled or constraint-reduced solver dimension.

Freeze settings before constructing a Step, using the same declared options:

```python
from pathlib import Path
from agentfem import provenance

settings = provenance.model_settings(model, target=u, step_options=options,
    source_files=(Path("case.py"),))
step = model.step(target=u, **options)
simulation = step.solve_result()
simulation.metadata["model_settings"] = settings
run.publish(simulation)
```

The result templates do this automatically. Compare two published runs with:

```bash
agentfem compare outputs/001-baseline outputs/002-candidate --settings --json
```

Scientific changes have object/parameter paths; operational settings, source
hashes and software-version changes are separate lists. The command also
accepts JSON files containing standalone `provenance.model_settings` records.
Exit 0 means unchanged within complete recorded scope; changed or incomplete
scope returns 2. Missing snapshots and malformed records fail explicitly.
Opaque expressions, direct unlocated constraints and unsupported constraint
identities remain coverage gaps. The comparison never replays Python or edits
either result; the existing quantity/history tolerance comparison is unchanged.

`diagnostics.resource_budget(model, fields=(u,), frames=5,
output_directory="outputs")` is the same budget without boundary quadrature.
All ranks must call these model-level functions. Resource snapshots retain
each rank's readable host/cgroup/address-space limits and available disk;
shared limits are not summed. Planned frames can be inferred for declared
fixed-step transient schedules; adaptive accepted counts remain unknown.
Payload estimates count uncompressed FE coefficients only. Derived fields
without a function retain unknown size. Mesh/connectivity, projection,
compression and histories can change actual file size; LU fill/peak memory
remain unknown without further analysis. No mesh, solver, output policy or
environment setting is automatically changed.

## Find a task and use its installed project

```bash
agentfem capabilities --task "载荷—位移曲线" --json
agentfem capabilities --task task:point-history --json
agentfem init my-static --template static-results
agentfem init my-dynamic --template dynamic-results
agentfem init my-path --template nonlinear-results
agentfem run --project my-dynamic
agentfem show latest --project my-dynamic --json
```

Six deterministic bilingual task entries cover modal frequencies, point/time
histories, load–displacement paths, mesh input, standard output and solver
options. Entries are generated from scientific cards; packaged templates are
the source of their full snippets. Search does not execute a case, initialize
FEM, install dependencies or access the network. An unmatched phrase means
no retrieval match, not unsupported physics. `task:ID` rejects unknown IDs.
Use `--api` for the installed signature and `models.step_capability(...)` for
the actual material/element/procedure combination. No second support matrix
or parameter configuration language is introduced.

| Installed recipe | Recorded results | Important convention |
| --- | --- | --- |
| `static-results` | 3D SI U/S fields, interpolated tip displacement and fixed-end reaction | Reaction is support-on-body force; Warp adds U to reference geometry once. |
| `dynamic-results` | Newmark U fields at sparse stations and dense `tip_U2` history | Physical time in seconds; field retention does not thin the scalar history. |
| `nonlinear-results` | Retained accepted load factors and mean-displacement/reaction curve | Serial, homogeneous displacement-only hyperelasticity; no natural loads/contact/MPC. A boundary mean is not a point displacement. |

The normal result view lists field positions/shapes/units, history coordinates
and ranges, saved frame counts and the recommended visualization entry. JSON
adds a read-only `result_menu`; it does not change the sealed manifest.
Registered visualization paths are resolved relative to the manifest, so a
complete run directory can be moved without selecting its old absolute path.
Transient primary fields and physical-time histories retain the model's
declared units; labels do not rescale values. Without declared time units,
the historical seconds convention is retained. Continuing a Step without
field output does not report the previous run's frames as new output.
Summaries read metadata and check file presence, never load field arrays or
rerun the model. Legacy missing metadata stays unknown. `show --files` and
the existing result verification retain their separate integrity roles.

`agentfem diagnose <execution.json> --json` preserves local messages, rank,
Step details and validation hints. It remains a read-only operation even
when the recorded simulation failed. Model-level checks require the original
`model.validate()/check()` path; ordinary CLI `check` does not execute it.
Detailed local evidence is excluded from the existing sanitized support and
telemetry inventories. Diagnostic suggestions never change physical inputs.

## Discover only what you need

```bash
agentfem capabilities workflow --json
agentfem capabilities materials --json
agentfem capabilities abaqus_user_material_bridge --json
```

Topic queries retain the selected contract, limitations and evidence while
omitting unrelated sections. Their `scope.complete` is false: an omitted
section is not a statement that the capability is unsupported. Workflow queries
do not probe unrelated numerical runtime or material catalogs. Use
`agentfem capabilities --json` for the unchanged complete record, and
`models.step_capability(model, target=..., options=...)` for a particular model.

When a parameter is uncertain, inspect the installed callable directly:

```bash
agentfem capabilities --api fields.displacement --json
agentfem capabilities --api models.Model.step --json
agentfem capabilities --api constitutive.StrainHistoryRuntime --json
```

This reports parameter names, kinds, required inputs, simple defaults and
declared annotations from the installed implementation. Model methods also
retain their lifecycle status. Exact simple signatures are read from installed
source without initializing the numerical runtime. Already-loaded modules,
dynamic defaults and decorated callables retain live inspection; the
`inspection` field identifies the route. Neither route executes the callable,
evaluates annotation expressions or runs a case.
Runtime-object defaults are identified by type rather than expanded.
Public classes explicitly exported by a module can be queried as constructors;
the query does not instantiate them or compile user materials. Step
provider-specific keywords still come from `models.step_capability`; a Python
signature alone is not evidence that a physical combination is supported.

## Run a project

One project remains one readable `case.py`, with operational metadata in
`agentfem.toml`. These commands do not require an AI assistant.

```bash
agentfem check --runtime
agentfem run --name baseline
agentfem show latest
agentfem show latest --files
```

For a distributed case, all ranks can call
`run.publish(simulation, comm=domain.comm)`. This reuses the Result's collective
agreement checks and gives rank zero ownership of summary/execution output.
Publication errors reach every rank; no case-level publication barrier is
needed. Omitting `comm` retains serial or explicitly root-owned publication.

`check` checks project structure and Python syntax without executing the case.
`--runtime` also checks numerical runtime readiness. Neither promises that a
particular mesh, material, boundary condition or solver combination is supported;
the model's own preflight remains authoritative.

## Review a defined model before solving

After defining materials, constraints and loads, review the actual model without
starting a solver:

```python
card = model.review(target=displacement, step_options={"procedure": procedure})
```

The returned JSON-safe model card records the selected analysis capability,
material and load semantics, boundary measures/resultants, constraint degrees of
freedom, rigid-mode audit, and findings such as an empty boundary selection. It
is a snapshot of the reference configuration: it does not update state or solve
an equilibrium system. Boundary integrals may compile and assemble forms.
`solver_executed` is consequently always `false`. It checks a declared
model, not whether the material law is scientifically valid or a nonlinear
equilibrium path will converge.

For a spatially varying traction, a tuple containing UFL expressions is accepted
and becomes a vector expression automatically:

```python
x = ufl.SpatialCoordinate(domain)
model.traction((load0 * (1 + 0.2 * x[1]), 0.0), on=loaded_edge)
```

The card distinguishes reference-configuration and current-configuration
pressure. Reference pressure integrates `-p N`; current (follower) pressure
uses the deformed area vector `-p J F^{-T}N`. In a two-dimensional solid study,
resultants are forces per declared out-of-plane thickness. These declarations
make the physical convention inspectable; they do not replace a user-selected
modeling assumption.

With `--json`, invalid CLI arguments return an `agentfem.cli-error` record and
exit code 2 before runtime initialization. The stage is `argument_parsing` and
the code is `AFM-CLI-ARGUMENT`. Arguments after `--` remain the case's arguments;
a case-level `--json` does not change AgentFEM's output mode.
JSON output rejects nonfinite numbers rather than emitting nonstandard
`NaN`/`Infinity` tokens or silently replacing scientific values with null.

`show --files` lists recorded artifacts and reports missing files with exit code
2. Keep an XDMF file together with its referenced HDF5 files. This command checks
recorded paths, not the internal contents of visualization files. Normal `show`
keeps the file list short; JSON and `--files` retain the complete list.

## Repair Step option errors

The selected provider checks `model.step` keywords against the same
`StepOptionContract` exposed by capability inspection. In Python, catch
`agentfem.step_providers.StepOptionError` (also a `TypeError`) and call
`error.details()` for provider, issue codes, option names, suggestions and
accepted option names. Details do not contain supplied values or material state.

During `agentfem run`, the same record is retained in `execution.json` under
`error.details`, with stage `step_options`. The command still exits nonzero;
no automatic correction or rerun is performed. An agent can propose a precise
change without parsing the traceback. Ordinary tracebacks remain diagnostic
artifacts and are not a privacy-filtered export.

Suggestions are possible spelling repairs, not approval of physical meaning.
For example, two mutually exclusive frequency inputs require a modeling choice,
not blindly deleting one. This contract checks provider options, not all units,
mesh/material combinations or scientific validity.

For an unsupported analysis combination, `StepCapabilityError` remains a
`NotImplementedError` and exposes `details()` with the analysis, candidate
providers and their rejection reasons. `agentfem run` retains these under stage
`step_capability`. No alternate physics or numerical route is selected as a
repair. Missing explanations remain empty rather than invented.

Capability inspection prefers the same fully satisfied provider as execution.
If only a partial contract matches, `supported` may be true while `ready` is
false; inspect `readiness_issues` to identify missing inputs.

## Compare selected results

```bash
agentfem compare outputs/001-baseline outputs/002-candidate \
  --quantity displacement_max --atol 1e-8 --rtol 1e-5
```

Choose quantities explicitly, with an absolute tolerance in their recorded
units. Repeat `--quantity` or use `--history reaction`. One invocation applies
the same tolerances to all selected observables; use separate invocations when
different quantities need different tolerances.

The comparison requires completed result manifests, identical declared units
and shapes, and, for histories, stored samples with identical increasing
coordinates. It never silently converts units, interpolates histories or maps
fields between meshes. Missing units are not assumed dimensionless: record `1`
explicitly. Missing or incompatible data fails closed. Relative error against
a zero reference norm is unavailable; absolute error is still reported.

Acceptance is componentwise `abs(candidate-reference) <= atol + rtol*abs(reference)`.
Exit code 0 means the selected observables meet these tolerances; code 2 means
failure or incompatibility. Add `--json` for machine-readable output. This is
not independent scientific validation and does not promote either result's trust.

## A small real finite-element campaign

```bash
agentfem init wall-study --template heat-campaign
cd wall-study
agentfem check --runtime
agentfem run --name first -- --samples 3
agentfem run --name resumed -- --samples 3
agentfem show latest --files
```

The installed template solves steady heat conduction, varies the hot boundary
temperature deterministically, and exports a scalar-response dataset and audit.
The project-local `campaign-data` ledger lets the unchanged plan reuse completed
cases. Edit the physical problem in `case.py`; consult the template README before
changing an existing campaign. This is a bounded workflow example, not a trained
surrogate or a claim that arbitrary thermal models are validated.
