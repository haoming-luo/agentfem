# Installed Project Workflow

AgentFEM is both a Python library and an installed finite-element application.
The same case remains executable with `python case.py`; the `agentfem` command
adds project discovery, checks, repeatable output locations, MPI launch, and
machine-readable run evidence.

## Prepare the numerical environment

AgentFEM currently uses a compatible FEniCSx/PETSc/MPI environment. On Linux,
macOS, and Windows through WSL2, create that stack with conda-forge and install
the AgentFEM wheel into it. Then check the actual runtime:

```bash
agentfem doctor
agentfem doctor --json
```

The JSON form is intended for issue reports, IDE integrations, and agents. It
records the platform route, exact interpreter and imported package, core
versions, and optional mesh, visualization, machine-learning, and
distributed-MPC integrations.

`agentfem capabilities --json` also separates the public Python surface into
`core`, `advanced`, and `expert` layers. A generated first case should normally
use only the core layer; this reduces API search without restricting direct
access to advanced finite-element or backend capabilities.
The same report separates the concise `Model` vocabulary from advanced and
0.2.x compatibility methods, and lists the accepted/required options for every
built-in Step provider. Agents and interfaces should consume those contracts
instead of guessing solver arguments from examples.

## Create a project anywhere

```bash
agentfem templates
mkdir beam
cd beam
agentfem init --template static-solid .
agentfem check
agentfem run
agentfem inspect
agentfem verify
```

Initial templates include `static-solid`, `steady-heat`, and
`structural-dynamics`. The generated layout is intentionally small:

```text
beam/
├── agentfem.toml
├── case.py
├── AGENTS.md
├── README.md
└── outputs/
```

`case.py` is the source of modeling truth. It contains the Study, mesh,
regions, fields, materials, constraints, loads, steps, and result requests.
`agentfem.toml` does not duplicate the physics; it only identifies the project
name, Python entrypoint, and output root.

Use the concise Study factories for common engineering work:

```python
study = studies.static_solid(dimension=3)
study = studies.steady_heat_transfer(dimension=3)
study = studies.transient_heat_transfer(dimension=3)
study = studies.dynamic_solid(dimension=3, method="explicit")
study = studies.dynamic_solid(dimension=3, method="newmark")
```

The Study states the physical problem; `method` selects the procedure without
changing that problem's identity. `model.check()` uses the same provider rules
as `model.step(...)`, so a declared but unsupported combination is rejected
before numerical work begins.

Time histories are reusable model assets rather than custom step callbacks:

```python
ramp = amplitudes.ramp(0.0, 1.0, start_time=0.0, end_time=0.1)
model.traction((0.0, -1.0e6), on=loaded, amplitude=ramp)
model.prescribed_temperature(T, temperature_history, on=heated)
model.convection(
    on=cooled,
    coefficient=25.0,
    ambient_temperature=ambient_history,
)
```

When the engineering input is a total end force rather than a traction, let
the model distribute it over the selected reference edge or surface:

```python
model.surface_force((0.0, -50_000.0), on=loaded)
```

AgentFEM assembles the MPI-global boundary measure and applies a uniform
traction whose resultant is the requested vector. In 2D the force is per unit
out-of-plane thickness.

Standard/Explicit dynamics and transient heat update registered histories at
their physical step times. Manual callbacks remain available for advanced
state not represented by a model asset.

Resolve input files from the case directory rather than the shell's current
directory:

```python
from pathlib import Path

HERE = Path(__file__).resolve().parent
mesh_path = HERE / "meshes" / "component.inp"
```

## Run directly or through the product shell

Direct Python remains supported:

```bash
python case.py
```

The product shell adds a stable run identity and output contract:

```bash
agentfem run
agentfem run --run-id baseline
agentfem run --mpi 4
```

The product shell is preferred because it selects the MPI launcher shipped by
the active environment.  When an explicit launcher is required, use that same
environment's executable rather than an unrelated `mpiexec` found earlier on
`PATH`:

```bash
"$CONDA_PREFIX/bin/mpiexec" -n 4 python -m agentfem.cli run
```

Do not start a second MPI launcher from an already distributed case. The CLI
detects the communicator and runs the entrypoint on it.

## Understand the output contract

Each run receives its own directory:

```text
outputs/<project>/<run-id>/
├── execution.json
├── result.json
├── fields.xdmf
├── fields.h5
└── logs/
```

- `execution.json` answers whether the application completed or failed. A
  failed run records its stage, stable validation/error code when available,
  MPI rank evidence, and complete traceback; numerical result files are not
  required in order to diagnose a preflight or execution failure.
- `result.json` is the published `SimulationResult`, including quantities,
  histories, artifacts, checkpoints, metadata, and verification evidence.
- `outputs/<project>/latest.json` points to the most recent run without a
  platform-specific symbolic link.
- XDMF/HDF5, CSV, NPZ, images, and reports are artifacts referenced by the
  result rather than replacements for it.

Published results are sealed automatically. `agentfem verify` follows the
latest run by default and checks both the manifest and all registered files.
It detects changed or incomplete output; it does not replace convergence,
mesh-sensitivity, or engineering validation checks.

`agentfem run --json` reserves standard output for the final machine record and
writes case and solver logs into the run directory. A GUI or agent can parse
the response without scraping progress prose.

## Check before solving

The first `agentfem check` validates project structure and Python syntax
without solving. For a case created with an earlier release, also run:

```bash
agentfem upgrade --json
```

This is a dry-run migration plan. `--apply-safe` is limited to deterministic
project metadata; findings marked `semantic_review=true` require inspection of
the finite-element meaning and renewed evidence. Model construction should
additionally call `model.check()` before the step. The evidence layers are
separate:

1. project and entrypoint checks;
2. model, region, material, load, and capability checks;
3. solver convergence evidence;
4. scientific verification and validation evidence.

A zero process exit code means the requested operation completed. It does not
by itself promote a result from computed to verified.

## Pause and resume transient work

Heat transfer, Standard dynamics, and Explicit dynamics share one restart
workflow:

```python
step.run(until_step=50)
checkpoint = step.save_checkpoint("restart/step-50")

resumed = build_the_same_step()
resumed.load_checkpoint(checkpoint)
result = resumed.solve_result(output="restart/continuation.xdmf")
```

The checkpoint detects incompatible procedures, time contracts, mesh/function
layouts, missing shards, and corrupted shards before applying state. Fast
rank-local shards support the same MPI size and mesh partition. For supported
nodal transient state, `save_checkpoint(..., portable=True)` or
`checkpointing.every(..., portable=True)` additionally writes a verified
physical-node-keyed state that can be restored with a different MPI partition
or rank count. Ordinary nodes use quantized coordinates and field components;
coincident independent nodes on a split cohesive interface additionally use
their durable source-node identity. Constitutive integration-point state
requires a separate portable
cell/quadrature identity and is not implied by the nodal option. The continued
field artifact starts at the checkpoint time and is labeled as a continuation
segment; full histories and execution evidence remain attached to the result.

For common post-processing, prefer named, MPI-safe quantities over ad hoc local
array inspection: `results.region_integral(...)`,
`results.region_average(...)`, `results.boundary_resultant(...)`, and
`results.field_extrema(...)`.

## Continue into campaigns and learning

One case and a parameter campaign should use the same case builder and result
contract. Campaigns consume declared inputs and `SimulationResult` outputs,
then create a `ScientificDataset` only when the selected evidence policy is
satisfied. The dataset can be exported to NumPy, passed to built-in
ridge/POD/PyTorch surrogates, or used by an existing user model. A
neural-field model can also participate directly in the ordinary Step lifecycle
through `model.step(target=spec, executor=my_solver)`; no companion package or
framework-specific base class is required. Maintained reference providers for
selected methods live in the optional
[AgentFEM-Learning](https://github.com/haoming-luo/agentfem-learning)
companion.
