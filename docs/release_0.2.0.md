# AgentFEM 0.2.0

**Channel:** first non-prerelease release  
**Released:** 13 August 2026

AgentFEM 0.2.0 establishes a public foundation for AI-native finite-element
computing: a readable engineering model shared by humans and agents, a reusable
finite-element extension layer, and a deterministic FEniCSx/PETSc/MPI numerical
kernel. It is a stable package release, while scientific maturity continues to
be declared separately for every capability.

## Highlights

- **Readable installed workflows.** A user can create a static-solid, heat, or
  structural-dynamics project anywhere and use one
  `check -> run -> inspect -> verify` lifecycle.
- **Nonlinear state with distributed evidence.** J2 plasticity has regional
  quadrature state, consistent tangent, cutback, histories, portable restart,
  and a public thick-cylinder first-yield benchmark that agrees across serial
  and two-rank MPI execution.
- **Cohesive fracture and fatigue infrastructure.** Fixed-path Mode-I and
  mixed-mode interfaces support full-vector jumps and tractions, cycle
  coordinates, cycle jump, rollback/restart, ordered non-proportional paths,
  energy evidence, and persistent crack-component identity. These advanced
  workflows remain explicitly experimental.
- **Engineering interoperability.** Structured, XDMF, optional Gmsh/meshio,
  Abaqus C3D10H, equation constraints, physical groups, and distributed
  periodic workflows retain engineering semantics above the numerical mesh.
- **Simulation to learning.** Accepted results can become scientific datasets,
  feed built-in or user-provided PyTorch models, pass validation and
  applicability guards, and fall back to FEM outside their trusted domain.

## Installation

AgentFEM supports Linux, macOS, and Windows through WSL2. Until the conda-forge
recipe is merged, create the compiled numerical environment and install the
released wheel from PyPI:

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 mpich mpi4py petsc4py h5py
mamba activate agentfem-env
python -m pip install agentfem
agentfem doctor
```

## Release evidence

The release workflow builds the wheel and source archive once, then verifies
those immutable artifacts through:

- the complete serial test suite;
- distributed constraints, results, fracture, transient, and inelastic tests;
- checkpoint recovery across MPI rank counts;
- the external J2 thick-cylinder serial/MPI comparison;
- all installed project templates and release-facing examples;
- strict documentation and scientific-knowledge checks;
- an optional CPU PyTorch bridge job;
- package metadata, payload identity, and provenance attestation.

## Capability maturity

| Capability | Maturity in 0.2.0 |
| --- | --- |
| Linear static solids and transient heat | Release |
| Explicit waves, C3D10H periodic hyperelasticity, J2 plasticity, global power-law creep, simulation-to-learning | Engineering |
| Dynamic cohesive fracture and cyclic cohesive fatigue | Experimental |

The machine-readable scope is
[`agentfem/release/0.2.0.json`](https://github.com/haoming-luo/agentfem/blob/main/src/agentfem/release/0.2.0.json).
It states both executable evidence and claims that this release does not make.

## Upgrade notes

- `pip install agentfem` now selects this release without `--pre`.
- Existing preview projects can be audited with
  `agentfem upgrade --project PATH --json`; scientific Python is never silently
  rewritten.
- Linux is the CI platform, macOS is developer-verified with conda-forge
  FEniCSx, and WSL2 is the supported Windows route. Native Windows remains
  experimental.
- The Gmsh and PyTorch integrations remain optional; neither is bundled into
  the Apache-2.0 core.
