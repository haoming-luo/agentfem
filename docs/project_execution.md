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
```

Project checks and the CLI run admission validate this constraint without
executing the case. This is minimum software compatibility, not a reproducibility
lock. Freeze runtime identity and scientific inputs separately for reviewed
runs. Moving code from a project into the library changes implementation
identity: retain the old environment for old checkpoints or start a fresh run;
do not remove restart guards to make old archives load.
