# AgentFEM Module Map

This map links FEM concepts to the current Python modules.

| Concept | Module |
| --- | --- |
| Mesh import, tags, and measures | `mesh.py` |
| External mesh formats | `mesh_formats.py` |
| Function spaces | `spaces.py` |
| Dof lookup and field copying | `dofs.py` |
| Strong BC construction | `boundary.py` |
| Constraint containers | `constraints.py` |
| Loads and time signals | `loads.py` |
| Constitutive laws | `constitutive/` |
| Boundary models | `boundary_models/` |
| UFL weak-form blocks | `forms.py` |
| Assembly | `assembly.py` |
| Time kernels | `time.py` |
| Problem/state containers | `problems.py` |
| Runtime cadence | `runtime.py` |
| Linear solvers | `solvers.py` |
| Diagnostics | `diagnostics.py` |
| Output | `io.py` |
| Runnable workflows | `examples/` |

Application geometry, case inputs, and problem-specific sources should stay in
application packages or examples until they become reusable platform concepts.
