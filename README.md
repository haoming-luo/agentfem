# AgentFEM

AgentFEM is an agent-oriented finite-element workflow platform built on
DOLFINx/PETSc. It aims to make finite-element scripts readable to both
researchers and AI agents by keeping the standard CAE workflow visible:

```text
Study -> Model -> Mesh/Regions -> Fields -> Materials -> Loads/Constraints
      -> Operators -> Step -> Solve -> Diagnostics/Output
```

The current MVP focuses on linear elasticity, transient heat conduction,
explicit elastodynamics, reusable loads and constraints, material records,
operator-level `K/M/C/F` notation, model inspection, and ParaView/XDMF output.

## Why AgentFEM

- Finite-element language for humans: `model.linear_static_step(...)`,
  `model.tree()`, `operators.xtmx(...)`, and field algebra such as
  `u_next = u + dt * v + 0.5 * dt**2 * a`.
- Inspectable objects for agents: `study.summary()`, `model.manifest()`,
  `operator.summary()`, and step summaries.
- Transparent layers: daily workflows use `models`, `fields`, `loads`,
  `operators`, and `problems`; advanced users can still drop to `forms`,
  `assembly`, PETSc, or DOLFINx when needed.

## Install

AgentFEM currently expects a working FEniCSx/DOLFINx stack. The recommended
route is conda-forge:

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 gmsh mpi4py petsc4py \
  meshio matplotlib jupyterlab ipykernel
mamba activate agentfem-env
```

For local development from this repository:

```bash
python -m pip install -e .
```

After the first PyPI release, users will install AgentFEM with:

```bash
python -m pip install agentfem
```

`requirements.txt` records the tested MVP stack and optional documentation /
notebook helpers. Pure pip installation of DOLFINx can be fragile because MPI,
PETSc, and HDF5 must match.

## Quick Start

Run the beginner static-elasticity example:

```bash
python examples/static_elasticity_2d.py
```

From the parent development directory used in this workspace:

```bash
python agentfem/examples/static_elasticity_2d.py
```

The output is written to `examples_output/static_elasticity_2d.xdmf` and can be
opened in ParaView.

## Minimal Workflow

```python
from mpi4py import MPI
import numpy as np

from agentfem import fields, mesh, models, studies
from agentfem.constitutive import elasticity

study = studies.linear_static(
    physics="solid_mechanics",
    dimension=2,
    assumption="plane_strain",
)

domain = mesh.rectangle(
    lower=(0.0, 0.0),
    upper=(1.0, 0.2),
    cells=(40, 8),
    comm=MPI.COMM_WORLD,
    cell_type="quadrilateral",
)
model = models.create(study=study, mesh=domain, name="cantilever")

u = model.field(fields.displacement(domain, degree=1))
model.material(elasticity.isotropic_elastic(young=210e9, poisson=0.3, density=7800))

left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1)
right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2)
model.fix(u, on=left, value=0.0)
model.traction(value=(0.0, -1.0e6), on=right)

step = model.linear_static_step(target=u)
step.solve()

print(model.tree())
```

The `Step` path is the recommended public workflow. It still exposes the
operator system for review:

```python
print(step.system.summary())
```

## Public Workflow Modules

Beginner and agent-generated workflows should prefer:

```python
from agentfem import studies, mesh, models, fields, materials, constitutive
from agentfem import amplitudes, constraints, loads, operators, problems
from agentfem import solvers, time, io, diagnostics
```

Lower-level modules such as `forms`, `assembly`, `spaces`, and `kernel` remain
available for extension work and debugging, but they should not be the first
thing a new model exposes.

## Examples

- `examples/static_elasticity_2d.py`: beginner linear-static mechanics example.
- `examples/transient_heat_2d.py`: intermediate first-order transient heat solve.
- `examples/wave_packet_plate_2d.py`: advanced explicit dynamics wave example.
- `examples/wave_packet_inclusion_2d.py`: advanced wave propagation with
  regional material assignment and absorbing/periodic boundary handling.

## Documentation

- `WORKFLOW.md`: standard AgentFEM modeling sequence.
- `INSTALL.md`: tested MVP environment and smoke-test command.
- `CONCEPTS.md`: shared vocabulary for finite-element and agent workflows.
- `AGENT_GUIDE.md`: first file for AI agents working in this repository.
- `docs/`: design notes, module map, validation notes, and extension rules.
- `docs/publishing.md`: PyPI release checklist and Trusted Publisher setup.
- `site/index.html`: generated static documentation site.

Rebuild the local documentation site with:

```bash
python build_docs.py
```

## MVP Status

AgentFEM is a research-oriented MVP. It is not yet a general-purpose CAE
replacement. The stable direction is:

- model-first workflows for common analyses,
- operator-first workflows for transparent research code,
- structured model manifests for agents,
- examples and benchmarks that make assumptions explicit.
