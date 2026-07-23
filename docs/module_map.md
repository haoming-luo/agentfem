# AgentFEM Module Map

This map links FEM concepts to the current Python modules.

| Concept | Module |
| --- | --- |
| Mesh import, named regions, tags, summaries, checks, and measures | `mesh/` |
| External mesh formats | `mesh/formats.py` |
| Function spaces and Lagrange defaults | `spaces.py` |
| Application-level unknown fields | `fields.py` |
| Low-level dof lookup and field copying | `kernel/dofs.py` |
| Constraint containers and strong BC construction | `constraints/` |
| Loads, natural boundary data, and time signals | `loads.py` |
| Constitutive response relations | `constitutive/` |
| Material records and property containers | `materials/` |
| Boundary models | `boundary_models/` |
| UFL weak-form blocks, including stiffness, mass, diffusion, and loads | `forms.py` |
| Assembly | `assembly.py` |
| Engineering-level K/M/C/F operators | `operators/` |
| Time integration kernels and runtime cadence | `time/` |
| Problem summaries, system problems, and state containers | `problems.py` |
| Linear solvers | `solvers.py` |
| Diagnostics | `diagnostics.py` |
| Output writers and scalar logs | `io.py` |
| Element and integration policies | `elements/` |
| Verification benchmarks | `benchmarks/` |
| Runnable workflows | `examples/` |

Application geometry, case inputs, and problem-specific sources should stay in
application packages or examples until they become reusable platform concepts.
