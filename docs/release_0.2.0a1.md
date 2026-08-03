# AgentFEM 0.2.0a1

**Released:** 3 August 2026
**Status:** public alpha preview

AgentFEM was initiated by Haoming Luo and open-sourced on GitHub in July 2026.

Version 0.2.0a1 is the first public preview of AgentFEM as an AI-native
finite-element platform rather than a collection of isolated helpers. It is a
deliberately early release for researchers, engineers, and AI/CAE developers
who want to test the workflow, inspect its design, and help shape the 0.2
series.

## Highlights

- A readable `Study -> Model -> Step -> Result` FEM workflow for humans and
  agents.
- Linear, nonlinear, thermal, implicit-dynamic, and explicit-dynamic
  procedures on the current FEniCSx/PETSc/MPI kernel.
- Neo-Hookean finite strain and a global small-strain J2 path with quadrature
  state, consistent tangent, physical cutback controls, cyclic loading,
  energy histories, and serial restart.
- Abaqus C3D10 and equation import, external mesh conversion, and distributed
  periodic workflows.
- Unified result, progress, checkpoint, Golden-benchmark, and quality-policy
  evidence.
- Reproducible campaigns, scientific datasets, PyTorch adapters, surrogate
  validation, applicability guards, and high-fidelity fallback.
- Versioned scientific knowledge cards, benchmark contracts, and explicit
  capability maturity boundaries.

## Install

Create the FEniCSx numerical environment first, then opt into the preview:

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 mpich mpi4py petsc4py h5py
mamba activate agentfem-env
python -m pip install --pre agentfem
```

## Release Evidence

The tagged source is accepted only after:

- the source version, package version, and Git tag agree;
- the wheel and source distribution pass metadata and payload checks;
- the installed wheel passes serial and distributed FEniCSx tests;
- the two-rank MPI regression passes;
- the static, heat, creep, wave, and nonlinear release contracts pass; and
- a separate environment installs the official CPU-only PyTorch wheel and
  verifies the optional simulation-to-learning interfaces.

## Honest Boundaries

This alpha is not a universal CAE replacement. It does not claim global
adaptive creep/damage, portable MPI restart for quadrature state, general
UMAT/UHYPER binary compatibility, industrial code compliance, automatic
arbitrary-mesh neural-operator training, or a complete native-Windows solver
stack. WSL2 is the recommended Windows route.

The 0.2 series will deepen the implemented workflows and their external
benchmarks before expanding the public vocabulary indiscriminately.
