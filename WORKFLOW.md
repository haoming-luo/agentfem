# AgentFEM Workflow

AgentFEM uses a standard finite-element workflow. Application code should make
this sequence visible unless there is a strong reason to encapsulate it.

## Standard Sequence

1. Define, import, convert, or read the mesh.
2. Inspect the mesh summary and require expected material/boundary tags.
3. Define named mesh regions for boundaries, material zones, or point sets.
4. Create function spaces or application unknown fields.
5. Create state containers when needed.
6. Load or define material properties, then choose constitutive laws.
7. Define essential constraints.
8. Define weak loads and boundary models.
9. Build operators or weak forms.
10. Compile and assemble forms.
11. Solve algebraic systems or advance in time.
12. Evaluate diagnostics.
13. Write outputs.

## Module Map

- Mesh import, named regions, tags, summaries, checks, and measures: `mesh/`
- External CAE mesh conversion: `mesh_formats.py`
- Spaces and fields: `spaces.py`
- Low-level dofs and vector access: `dofs.py`
- Constraints: `constraints.py`
- Loads and natural boundary data: `loads.py`
- Constitutive laws: `constitutive/`
- Material library: `materials/`
- Boundary models: `boundary_models/`
- Weak-form blocks: `forms.py`
- Assembly: `assembly.py`
- Operator families: `operators/`
- Time kernels: `time.py`
- Problem summaries and state containers: `problems.py`
- Runtime cadence: `runtime.py`
- Solvers: `solvers.py`
- Diagnostics: `diagnostics.py`
- Output: `io.py`
- Element/integration policies: `elements/`
- Verification benchmarks: `benchmarks/`

## Design Principle

The workflow should be easy for a human researcher to read and easy for an
agent to audit. A simulation file may call reusable helpers, but it should still
show the finite-element meaning of each step.

First-level Python modules are reserved for the main FEM workflow. Subpackages
hold reusable asset families, such as constitutive laws, material records,
boundary models, element policies, operator families, and benchmarks.
Low-level implementation helpers such as `boundary.py`, `dofs.py`, and
`kernel/` should not be featured in beginner workflows.
