# Lean projects and trusted user materials

`case.py` owns the engineering problem. A project does not need its own compiler
cache, time-integration loop, checkpoint handler or connectivity writer.

## Executing an existing Step

```python
from agentfem import project

context = project.current_run()
step = model.step(target=u, dt=dt, steps=steps, checkpoint=checkpoint_policy)
report = project.execute_step(
    step, context=context,
    policy=project.RunPolicy(until_step=100, journal_every=10,
                             progress_every=100, export_partial=True),
    metadata={"experiment": "reviewed study"},
)
```

This optional operational wrapper serves the existing serial transient Step
families. Ordinary short cases can still call `step.solve_result()` directly.
The wrapper does not build a Model, choose a numerical method, change the time
grid or adjust mass. Scheduled checkpoints remain declared on `model.step`.
Progress, accepted history journals, a final checkpoint, execution status and
failure reporting use existing contracts.

`RunPolicy.restart` restores through the Step's normal identity checks. A new
run directory is required. Completed paths publish a `SimulationResult`; partial
paths remain `paused`, not completed results. A wall limit is checked after
accepted increments and raises a recorded failure with an attempted accepted
checkpoint. It is not a hard process timeout. Use the CLI's process timeout for
unresponsive processes. The wrapper currently rejects MPI rather than inventing
collective file-writing behavior.

Optional `observations={name: callback}` receives the accepted Step and writes
a sampled observation journal. Observations must be pure and JSON-safe.
`export(step, directory)` returns named artifact paths. It runs at completion,
or at a partial endpoint if explicitly requested. An export failure remains a
failed delivery, with the accepted checkpoint retained; it does not erase the
numerical state or silently publish a completed result. Timing in `report.json`
covers Step execution and final export, not model construction or compilation.

## Accepted material output

```python
from agentfem import results

artifacts = results.export_accepted_response(
    step, directory,
    state_fields={"damage": ("history", 3)},
    units={"stress": "Pa", "displacement": "m", "time": "s"},
)
```

This bounded serial, one-material-point-per-cell route consumes
`accepted_material_fields()`. The caller explicitly identifies state-variable
meaning and components; there is no hard-coded damage column in core. Output
contains one standard visualization dataset, the complete raw accepted state,
cell keys, stress and gradient, and metadata. Invalid selectors and existing
output files are rejected. No field is silently smoothed or averaged across an
interface. Source labels, mesh provenance and physical units remain project
assets; cell input identity is not a source file's element label.

### One visualization entry point

For an animation, keep the endpoint in the same series instead of exporting a
second visualization of it:

```python
frames = results.AcceptedResponseSeries(
    context.output_directory / "fields.xdmf",
    state_fields={"damage": ("history", 3)},
)
report = project.execute_step(
    step, context=context,
    policy=project.RunPolicy(field_intervals=40),
    frames=frames,
    export=lambda accepted, directory: results.export_accepted_response(
        accepted, directory, visualization=False,
    ),
)
```

Open `fields.xdmf` in ParaView and select a time station; the last station is
the final accepted state. `fields.h5` is its companion data file, not another
visualization to open. `accepted_material.npz` preserves the full raw terminal
material state; it is not a restart checkpoint. The archive-only option creates
no second XDMF/HDF5 pair. Existing endpoint-only callers retain their default.
The series reuses a checked cell-identity mapping and DG0 spaces, never field
coefficients: each frame remains a detached, unsmoothed snapshot.

`report.json` identifies the primary visualization and separates restart,
initial fields, advancement, accepted callbacks, terminal checkpoint, terminal
fields/export and publication times. Advancement includes callbacks and
scheduled checkpoints; the callback-subtracted figure still includes scheduled
checkpoints and is not a pure solver-kernel time. These timings exclude model
construction and material compilation before `execute_step`, and the final
report write. Use process elapsed time for end-to-end comparisons.

## Trusted native source bundles

```toml
# material/material.toml
schema = "agentfem.native-libraries.v1"
files = ["law.f90", "parameters.inc"]
include_directories = ["."]
flags = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"]

[targets]
response = ["law.f90"]
```

```python
from agentfem import constitutive

bundle = constitutive.build_native_libraries(
    "material/material.toml", cache_directory=".agentfem/material-build",
    trusted=True,
)
library = bundle.library("response")
```

Explicit trust is required: native source is executable code, not safe data.
This is build infrastructure, not a new material ABI. Use `compile_vumat` for
the supported legacy VUMAT transport; use a reviewed provider for other ABIs.
The builder supports GNU Fortran on macOS/Linux/WSL2, snapshots declared inputs,
records flags and compiler identity, locks cache publication, verifies library
hashes on reuse, and retains failed-build logs. It never installs a compiler,
downloads code or modifies the environment. Declare source/include dependencies;
unsupported flags or undeclared Fortran includes fail before compilation.

Two generic arithmetic sources can be supplied by the installed distribution:

```toml
[bundled_sources]
"polar.f90" = "polar_series.f90"

[targets]
polar = ["polar.f90"]
```

`polar_series.f90` and `polar_wider.f90` implement guarded linear-right-stretch
path integration, with different fast-admission radii and the same residual
checks. `CompiledCorotationalIncrement(bundle, "polar")` repairs rejected
points through the reference `polar_strain_increment` before publication.
Neither is a universal interpretation of a vendor's VUMAT strain increment.
An application must explicitly choose and bind its kinematic convention.
Private material formulas, state layout, domain restrictions and specialized
fusion adapters remain outside core.

## Compatibility without private-module probing

Declare an installed-software requirement in the project configuration:

```toml
[project]
name = "my-study"
entrypoint = "case.py"
requires_agentfem = ">=0.4.3.dev0"
requires_api = [
  "constitutive.StrainHistoryRuntime",
  "results.AcceptedResponseSeries",
  "results.export_accepted_response(visualization)",
]
```

Project checks and the CLI run admission validate this constraint without
executing the case. This is minimum software compatibility, not a reproducibility
lock. Freeze runtime identity and scientific inputs separately for reviewed
runs. Moving code from a project into the library changes implementation
identity: retain the old environment for old checkpoints or start a fresh run;
do not remove restart guards to make old archives load.

`requires_api` optionally names public AgentFEM attributes. It catches missing
interfaces in older builds that share a development version, before executing
`case.py`. It imports the named AgentFEM modules but does not call the attributes.
The optional `(keyword)` suffix requires an explicitly named keyword-capable
parameter in the callable's signature. It is a declaration, never evaluated
Python: defaults, calls and multiple arguments are not accepted. A generic
`**kwargs` alone does not prove support. This catches older exporters before a
long solve, rather than failing only at final output. Presence/signatures still
do not prove semantic compatibility: record the package hash/commit for a
reproducible handoff, not just its version string.

A project deliberately selecting a native-only execution route can additionally
declare `requires_accelerators = ["hex_arithmetic", "material_body_pipeline",
"q1_geometry"]` under `[project]`. The check reports missing builds and loader
errors before executing the case. These names reuse `agentfem doctor`'s existing
availability probes; no worker is started. The empty default preserves ordinary
reference-only projects. Do not add this requirement to a project that supports
reference fallback, and do not confuse availability with measured performance.

## WSL execution environment

`agentfem check` and `agentfem run` issue read-only advisories for Windows-mounted
working/output directories, low available memory, or numerical-library thread
settings exceeding the CPU affinity. Run admission saves
`execution-environment.json` beside the outputs. It records visible resources,
mount detection and explicitly supplied thread settings, not a speed prediction
or a claim that all available cores are used.

For I/O-heavy runs, a WSL-native working copy can avoid Windows filesystem
interop costs. Export or back up results to Windows afterward: a distribution's
internal files can be lost when it is unregistered. AgentFEM never moves the
project, changes `.wslconfig`, or alters global thread settings automatically.
Keep the existing project-custody safeguards. CPU affinity and startup memory
are observations, not hardware-specific scientific limits; changing computers
does not change the model's physical inputs.

See Microsoft's [WSL filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
and [WSL configuration reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).
The advice does not replace measuring the actual installed backend: optional
accelerators being available and being selected by the Step are different facts.
