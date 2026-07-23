# AgentFEM Workflow

AgentFEM uses a standard finite-element workflow. Application code should make
this sequence visible unless there is a strong reason to encapsulate it.

## Standard Sequence

1. Define, import, convert, or read the mesh.
2. Create function spaces.
3. Create unknown fields and state containers.
4. Define constitutive laws and material properties.
5. Define essential constraints.
6. Define weak loads and boundary models.
7. Build weak forms.
8. Compile and assemble forms.
9. Solve algebraic systems or advance in time.
10. Evaluate diagnostics.
11. Write outputs.

## Module Map

- Mesh import, tags, and measures: `mesh.py`
- External CAE mesh conversion: `mesh_formats.py`
- Spaces and fields: `spaces.py`
- Dofs and vector access: `dofs.py`
- Constraints: `constraints.py`, `boundary.py`
- Loads: `loads.py`
- Constitutive laws: `constitutive/`
- Boundary models: `boundary_models/`
- Weak-form blocks: `forms.py`
- Assembly: `assembly.py`
- Time kernels: `time.py`
- Problem/state containers: `problems.py`
- Runtime cadence: `runtime.py`
- Solvers: `solvers.py`
- Diagnostics: `diagnostics.py`
- Output: `io.py`

## Design Principle

The workflow should be easy for a human researcher to read and easy for an
agent to audit. A simulation file may call reusable helpers, but it should still
show the finite-element meaning of each step.
