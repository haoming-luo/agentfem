<p align="center"><img src="logo/AgentFEM_logo_transparent.png" alt="AgentFEM logo" width="320"></p>

# AgentFEM

[![Test](https://github.com/haoming-luo/agentfem/actions/workflows/test.yml/badge.svg)](https://github.com/haoming-luo/agentfem/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/agentfem.svg)](https://pypi.org/project/agentfem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

AgentFEM is an open-source platform for **AI-native finite-element computing**.
It explores how engineering simulation may evolve in the age of AI agents—from
software designed primarily for human operation toward scientific workflows
that humans and AI agents can jointly understand, construct, and improve.

AgentFEM was initiated by Haoming Luo and open-sourced on GitHub in July 2026.

Its immediate goal is practical: to become a dependable and unusually usable
open-source FEM platform. Its longer-term vision is to turn finite-element
simulation into an accessible, shared scientific workspace connecting
engineering, computation, data, and AI.

## Why AgentFEM

- **AI-Native FEM** — finite-element software designed from the start for
  agents to construct, operate, and automate naturally, without replacing
  deterministic mechanics and numerical computation with AI.

- **Humans and Agents, Together** — people and AI agents work through the same
  readable materials, regions, loads, solution steps, and results. AI work
  remains understandable, editable, and reusable by humans.

- **Results You Can Check** — convergence, failures, required outputs,
  benchmark comparisons, and applicability limits remain attached to the
  result instead of being separated from the simulation that produced it.

- **One Run or Thousands** — the same model can support an individual
  analysis, parameter campaigns, parallel execution, restartable studies, and
  reproducible data generation.

- **Simulation to Learning** — results can flow into scientific datasets,
  PyTorch, surrogate models, and high-fidelity fallback without rebuilding the
  workflow around separate glue scripts.

- **Open at Every Layer** — users can begin with a clear engineering workflow
  and still reach operators, UFL, DOLFINx, PETSc, and custom constitutive
  models whenever needed.

> **Our conviction:** AI will make code abundant, but trustworthy scientific
> structure will remain scarce. AgentFEM is being built so that finite-element
> knowledge can be created, checked, communicated, and accumulated by humans
> and AI agents together.

## What Works Today

| Area | Implemented path |
| --- | --- |
| Engineering workflow | Study, model, regions, fields, materials, loads, constraints, steps, results, and concise model summaries |
| FEM procedures | Linear and thermoelastic statics, implicit heat transfer, Newmark/generalized-alpha dynamics, and central-difference explicit dynamics |
| Nonlinear solids | Compressible Neo-Hookean finite strain and a 3D small-strain J2 path with quadrature state, consistent tangent, cyclic loading, cutback, energy histories, and serial restart |
| Time-dependent solids | 3D isothermal power-law creep with backward Euler, shared quadrature state, analytical tangent, automatic physical-time cutback, CE/CEEQ/S/MISES/RF, dissipation, and serial restart |
| Meshes and constraints | Structured and XDMF meshes, optional Gmsh and meshio routes, Abaqus C3D10 import, equation constraints, and distributed periodic workflows |
| Results and trust | Unified fields, quantities, histories, artifacts, progress events, checkpoints, Golden benchmarks, and exploratory/engineering/release quality policies |
| Simulation and learning | Reproducible campaigns, scientific datasets, PyTorch adapters, transparent surrogate baselines, validation thresholds, applicability guards, and FEM fallback |

Power-law creep now has a bounded global 3D isothermal route. Arrhenius,
Kachanov--Rabotnov, and Sinh relations remain verified material-point tools;
modified theta is a curve-projection tool, and stress-life fatigue is a
postprocessor. AgentFEM keeps these maturity levels explicit rather than
letting one global material path silently promote the others.

The public workflow remains recognizable to a finite-element user:

```text
Study -> Model -> Mesh/Regions -> Fields -> Materials -> Loads/Constraints
      -> Operators -> Step -> Solve -> Results/Verification
```

## Architecture

AgentFEM uses three visible layers:

1. **Engineering workflow** — studies, models, regions, materials, loads,
   steps, campaigns, and results.
2. **Finite-element extension layer** — reusable operators, weak forms,
   constitutive laws, constraints, and custom scientific components.
3. **Numerical kernel** — the current FEniCSx/DOLFINx, PETSc, and MPI
   foundation for assembly, solution, and distributed computation.

The implementation is deliberately FEniCSx-first. Advanced users can descend
through every layer, while a narrow adapter boundary and experimental AF-IR
records preserve room for future evolution. AF-IR is not presented as a
universal simulation language or a neural-network compiler IR.

## Install

AgentFEM expects a compatible FEniCSx environment. The recommended route is to
create the numerical stack with conda-forge and then install AgentFEM from
PyPI:

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 mpich mpi4py petsc4py h5py
mamba activate agentfem-env
python -m pip install --pre agentfem
```

The `0.2` series is currently a public alpha. `--pre` opts into this preview;
ordinary `pip install agentfem` continues to select the latest non-prerelease.
AgentFEM is not yet distributed as a conda-forge package.

Optional integrations remain separate from the Apache-2.0 core:

```bash
python -m pip install --pre 'agentfem[mesh-formats]'  # Abaqus/NASTRAN/etc.
python -m pip install --pre 'agentfem[gmsh]'          # Gmsh model/.msh import
python -m pip install --pre 'agentfem[visualization]'
python -m pip install --pre 'agentfem[ml]'            # PyTorch adapters
```

Gmsh is a separately distributed GPL-licensed optional package and is not
bundled with AgentFEM. Windows users should currently use WSL2. See
[`INSTALL.md`](INSTALL.md) for platform details and development installation.

After installation, verify the numerical environment and create a project in
any working directory:

```bash
agentfem doctor
mkdir beam && cd beam
agentfem init --template static-solid .
agentfem check
agentfem run
agentfem inspect
```

`case.py` remains ordinary Python and can also be run directly. The CLI adds a
repeatable project root, run identity, MPI launch, structured result manifest,
and machine-readable interface for IDEs, GUIs, and AI agents. See the
[`Installed Project Workflow`](docs/getting_started.md).

## Quick Start

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
    (0.0, 0.0),
    (1.0, 0.2),
    (40, 8),
    comm=MPI.COMM_WORLD,
    cell_type="quadrilateral",
)
model = models.create(study=study, mesh=domain, name="cantilever")

u = model.field(fields.displacement(domain, degree=1))
model.material(
    elasticity.isotropic_elastic(
        young=210e9,
        poisson=0.3,
        density=7800,
    )
)

left = mesh.boundary(
    domain,
    lambda x: np.isclose(x[0], 0.0),
    name="left",
    tag=1,
)
right = mesh.boundary(
    domain,
    lambda x: np.isclose(x[0], 1.0),
    name="right",
    tag=2,
)
model.fix(u, on=left, value=0.0)
model.traction(value=(0.0, -1.0e6), on=right)

step = model.linear_static_step(target=u)
result = step.solve_result()
result.verify("engineering").require()

print(model.tree())
print(result)
```

From a source checkout, run the complete repository example with:

```bash
python examples/static_elasticity_2d.py
```

Models can be inspected before execution with `model.validate()`,
`model.tree()`, and `model.manifest()`. Experimental AF-IR records can be
written with `model.write_ir(...)` when a JSON-safe scientific record is
useful.

## Release Workflows

- [`static_elasticity_2d.py`](examples/static_elasticity_2d.py) — the readable
  beginner FEM path.
- [`transient_heat_2d.py`](examples/transient_heat_2d.py) — implicit heat
  transfer with structured progress and XDMF output.
- [`wave_packet_inclusion_2d.py`](examples/wave_packet_inclusion_2d.py) — wave
  propagation with an inclusion, source amplitude, and boundary models.
- [`abaqus_c3d10_periodic_cell/`](examples/abaqus_c3d10_periodic_cell/) —
  imported quadratic tetrahedra, Abaqus equations, distributed periodicity,
  Neo-Hookean large deformation, and homogenized output.
- [`creep_hot_wall_assessment.py`](examples/creep_hot_wall_assessment.py) —
  thermoelastic FEM followed by an explicitly local creep assessment.
- [`implicit_creep_relaxation_3d.py`](examples/implicit_creep_relaxation_3d.py)
  — global isothermal power-law creep with real state-based cutback and
  standard creep fields.
- [`static_elasticity_surrogate_campaign.py`](examples/static_elasticity_surrogate_campaign.py)
  — campaign, accepted dataset, surrogate validation, and FEM fallback.

These examples are executable release assets with numerical contracts; they
are not only syntax demonstrations.

## Documentation

- [`WORKFLOW.md`](WORKFLOW.md) — the standard modeling sequence.
- [`CONCEPTS.md`](CONCEPTS.md) — shared engineering and agent vocabulary.
- [`AGENT_GUIDE.md`](AGENT_GUIDE.md) — the entry point for AI agents working
  with the repository.
- [`docs/product_roadmap.md`](docs/product_roadmap.md) — capability priorities
  and release gates.
- [`docs/nonlinear_solid_architecture.md`](docs/nonlinear_solid_architecture.md)
  — the nonlinear-solid platform and quadrature-state contract.
- [`docs/results_and_campaigns.md`](docs/results_and_campaigns.md) — results,
  campaigns, datasets, and learning handoff.
- [`docs/scientific_verification.md`](docs/scientific_verification.md) — trust
  levels, quality policies, convergence studies, and evidence boundaries.

The complete design reference is under [`docs/`](docs/), and the generated
static site can be rebuilt with `python build_docs.py`.

## Direction and Scope

AgentFEM is an alpha-stage research and engineering platform, not yet a
general-purpose CAE replacement. The near-term priority is depth rather than
an inflated feature list: dependable nonlinear solids, thermal and dynamic
procedures, practical mesh interoperability, consistent output, and a smooth
path from simulation to trustworthy learning data.

The current release does not claim temperature-coupled global creep, global
creep damage or rupture prediction, portable MPI restart for quadrature
material state, general UMAT/UHYPER binary
compatibility, arbitrary-mesh automatic neural-operator training, industrial
code compliance, or a fully tested native-Windows solver stack. These are
visible engineering boundaries and roadmap gates, not hidden fine print.

## Citation

If AgentFEM helps your research or engineering work, please cite the project
metadata in [`CITATION.cff`](CITATION.cff).

```yaml
title: "AgentFEM: An AI-native open-source platform for finite-element computing"
authors:
  - family-names: Luo
    given-names: Haoming
    affiliation: "Materials Department, Xi'an Thermal Power Research Institute (TPRI)"
date-released: 2026-08-03
```

## Author

Haoming Luo is the initiator and maintainer of AgentFEM. His interests include computational mechanics, materials engineering, finite-element simulation, and AI-assisted scientific computing, with education and research experience associated with NWPU, INSA Lyon and Ecole Polytechnique.

The project is also motivated by engineering needs in materials evaluation,
defect inspection, and simulation analysis for power-generation equipment.

## License

AgentFEM is licensed under the Apache License, Version 2.0. The open-source core
can be used in research, education, and commercial settings under that license.
Commercial services, validated industrial workflows, hosted products, and
proprietary extensions may be developed separately.
