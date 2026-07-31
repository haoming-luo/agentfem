<p align="center"><img src="logo/AgentFEM_logo_transparent.png" alt="AgentFEM logo" width="320"></p>

# AgentFEM

AgentFEM is an open-source platform for AI-assisted finite-element simulation,
agent-readable CAE workflows, reusable operators, and human-agent collaborative
scientific computing. Its first collection-level interfaces connect typed
parameter studies and traceable FEniCSx runs to scientific datasets,
surrogates, reduced-order models, and guarded high-fidelity fallback.

AgentFEM was initiated by Haoming Luo and open-sourced on GitHub in July 2026.
It is an early open-source practice toward making finite-element simulation
workflows readable, structured, reproducible, and usable by both researchers
and AI agents.

AgentFEM focuses on making finite-element simulation workflows readable,
structured, and reproducible. It aims to make engineering modeling, material
definition, region management, loads and boundary conditions, solution steps,
and result outputs understandable and reusable by researchers, while also
readable, checkable, editable, and automatically orchestrated by AI agents.

The standard AgentFEM workflow remains visible:

```text
Study -> Model -> Mesh/Regions -> Fields -> Materials -> Loads/Constraints
      -> Operators -> Step -> Solve -> Diagnostics/Output
```

The current executable core covers linear elasticity, transient heat
conduction, explicit elastodynamics, and a first nonlinear-static
Neo-Hookean path. J2 plasticity and power-law creep are currently verified
material-point tools, while stress-life fatigue is a postprocessor. Those
maturity levels are deliberately distinct and queryable in code.

## Why AgentFEM

- Finite-element language for humans: `model.linear_static_step(...)`,
  `model.tree()`, `operators.xtmx(...)`, and field algebra such as
  `u_next = u + dt * v + 0.5 * dt**2 * a`.
- Inspectable objects for agents: `study.summary()`, `model.manifest()`,
  `model.to_ir()`, structured validation reports, `operator.summary()`, and
  step summaries.
- Collection-level scientific workflows: deterministic campaign cases,
  resumable evidence, unit/shape-aware datasets, transparent surrogate
  baselines, independent validation, and explicit applicability domains.
- One result path: solved fields, quantities, histories, visualization
  artifacts, batch outputs, and training data share `SimulationResult`.
- CAE mesh interoperability: inspect and convert supported external meshes
  while retaining named volume/boundary memberships in a conversion manifest.
- Transparent layers: daily workflows use `models`, `fields`, `loads`,
  `operators`, and `problems`; advanced users can still drop to `forms`,
  `assembly`, PETSc, or DOLFINx when needed.

## Architecture

AgentFEM adopts a three-layer design:

- Engineering simulation application layer: `Study`, `Model`, `Region`,
  `Material`, `Load`, `Step`, and `Result`.
- Finite-element operator and weak-form extension layer: reusable operators,
  constitutive laws, constraints, and custom variational forms.
- Numerical solver kernel layer: integration with FEniCSx/DOLFINx, PETSc, MPI,
  and related scientific-computing infrastructure.

The current implementation is intentionally FEniCSx-first. Experimental AF-IR
records preserve the supported scientific structure separately from live
backend objects, and a narrow backend adapter makes compilation boundaries
explicit. This is a progressive architecture, not a claim of complete
backend-neutral execution.

## Install

AgentFEM depends on the FEniCSx/DOLFINx scientific computing stack. The
recommended route is conda-forge, because it can resolve DOLFINx, PETSc, MPI,
and HDF5 together.

### Recommended: conda-forge

Create a fresh environment:

```bash
mamba create -n agentfem-env -c conda-forge python=3.11 agentfem
mamba activate agentfem-env
```

Or install into an existing conda environment:

```bash
mamba install -c conda-forge agentfem
```

### PyPI

If you already have a working FEniCSx/DOLFINx environment:

```bash
python -m pip install agentfem
```

### Local development

```bash
git clone https://github.com/haoming-luo/agentfem.git
cd agentfem
python -m pip install -e .
```

`requirements.txt` records the tested MVP stack and optional documentation and
notebook helpers. Pure pip installation of the full FEniCSx stack can be fragile
because MPI, PETSc, and HDF5 must match.

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

Inspect the model before execution or write a deterministic experimental AF-IR
record:

```python
report = model.validate()
print(report.format())
model.write_ir("cantilever.afir.json")
```

Validation issues contain stable codes, scientific-object paths, and repair
hints. AF-IR 0.1 is marked experimental: it records supported public semantics
and identifies unresolved backend runtime objects as opaque.

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
from agentfem import campaigns, datasets, results, surrogates
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
- `examples/static_elasticity_surrogate_campaign.py`: FEniCSx parameter
  campaign, scientific dataset, surrogate validation, and guarded FEM fallback.
- `examples/campaign_from_json.py`: safe JSON campaign specification with a
  trusted Python evaluator.
- `examples/material_models.py`: nonlinear-material maturity, J2 load path,
  creep history, rainflow counting, and Miner damage.

## Documentation

- `WORKFLOW.md`: standard AgentFEM modeling sequence.
- `INSTALL.md`: tested MVP environment and smoke-test command.
- `LICENSE`: Apache-2.0 license for the open-source core.
- `CONTRIBUTING.md`: contribution expectations and sign-off guidance.
- `CONCEPTS.md`: shared vocabulary for finite-element and agent workflows.
- `AGENT_GUIDE.md`: first file for AI agents working in this repository.
- `docs/`: design notes, module map, validation notes, and extension rules.
- `docs/product_roadmap.md`: product-first priorities and capability release
  gates.
- `docs/nonlinear_materials.md`: implemented nonlinear laws, formulas, and
  honest FEM/material-point/postprocessor boundaries.
- `docs/results_and_campaigns.md`: result, QoI, campaign, and dataset contract.
- `docs/mesh_interoperability.md`: external mesh conversion and named-set
  preservation.
- `docs/air_architecture_roadmap.md`: FEniCSx-first engineering plan and
  staged AF-IR/AIR architecture.
- `docs/ai_native_learning.md`: implemented campaign/dataset/surrogate
  contracts and the staged neural-operator/PINN route.
- `docs/licensing.md`: licensing strategy for the open-source core and
  optional commercial extensions.
- `docs/publishing.md`: PyPI release checklist and Trusted Publisher setup.
- `site/index.html`: generated static documentation site.

Rebuild the local documentation site with:

```bash
python build_docs.py
```

## Citation

If AgentFEM helps your research or engineering work, please cite the project
metadata in `CITATION.cff`.

```yaml
title: "AgentFEM: AI-assisted finite-element simulation and agent-readable CAE workflows"
authors:
  - family-names: Luo
    given-names: Haoming
    affiliation: "Materials Department, Xi'an Thermal Power Research Institute (TPRI)"
date-released: 2026-07-24
```

## Author / Maintainer

Haoming Luo is the initiator and maintainer of AgentFEM. His interests include
computational mechanics, materials engineering, finite-element simulation, and
AI-assisted scientific computing, with education and research experience
associated with INSA Lyon and Ecole Polytechnique.

The project is also motivated by engineering needs in materials evaluation,
defect inspection, and simulation analysis for power-generation equipment.

## MVP Status

AgentFEM is a research-oriented MVP. It is not yet a general-purpose CAE
replacement. The stable direction is:

- a dependable FEniCSx-first product for selected engineering analyses,
- model-first workflows for common analyses,
- operator-first workflows for transparent research code,
- one result-to-visualization/campaign/dataset flow,
- staged nonlinear materials whose maturity is tied to tests and benchmarks,
- versioned scientific records and structured repair reports for agents,
- examples and benchmarks that make assumptions explicit.

## License

AgentFEM is licensed under the Apache License, Version 2.0. The open-source core
can be used in research, education, and commercial settings under that license.
Commercial services, validated industrial workflows, hosted products, and
proprietary extensions may be developed separately.
