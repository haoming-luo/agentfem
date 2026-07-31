# AgentFEM Module Map

This map links FEM concepts to the current Python modules.

| Concept | Module |
| --- | --- |
| Mesh import, boundary/cell regions, tags, summaries, checks, and measures | `mesh/` |
| External mesh inventory, cell/facet set conversion, and manifests | `mesh/formats.py` |
| Abaqus labels, equations, periodic-cell geometry, and source-order output | `mesh/abaqus.py` |
| Study context: analysis type, physics, dimension, and assumptions | `studies.py` |
| Standard/Explicit family, equation order, algorithm, and state policy | `procedures.py` |
| Model registry, amplitudes, material assignments, checks, summaries, and model-first operators | `models.py` |
| Extensible analysis/material lowering behind `model.step()` | `step_providers.py` |
| Function spaces and Lagrange defaults | `spaces.py` |
| Application-level unknown fields | `fields.py` |
| Time histories and scale factors for prescribed data | `amplitudes.py` |
| Low-level dof lookup and field copying | `kernel/dofs.py` |
| Constraint containers and strong BC construction | `constraints/` |
| Loads and natural boundary data | `loads.py` |
| Constitutive response relations and maturity catalog | `constitutive/` |
| Quadrature-point committed/trial state | `constitutive/quadrature.py` |
| Global stateful solid-mechanics procedures | `mechanics/` |
| Material-point contracts and UMAT/UHYPER bridge specifications | `constitutive/user_material.py` |
| Material records and property containers | `materials/` |
| Boundary models | `boundary_models/` |
| UFL weak-form blocks, including stiffness, mass, diffusion, and loads | `forms.py` |
| Assembly | `assembly.py` |
| Engineering-level K/M/C/F operators | `operators/` |
| Central difference, Newmark, generalized-alpha, and runtime cadence | `time/` |
| Analysis steps, system problems, and state containers | `problems.py` |
| Automatic/fixed incrementation and cutback policy | `steps.py` |
| Linear/nonlinear solvers and convergence evidence | `solvers.py` |
| Scientific results, QoIs, histories, artifacts, and dataset bridge | `results/` |
| Finite-strain fields and periodic-cell homogenization | `results/finite_strain.py` |
| Standard result variables and context-aware aliases | `results/field_catalog.py` |
| Declarative field requests and unified XDMF/HDF5 time-series writer | `results/output.py` |
| Combined field/history/diagnostic/presentation output contracts | `results/plan.py` |
| Diagnostics | `diagnostics.py` |
| Output writers and scalar logs | `io.py` |
| Element and integration policies | `elements/` |
| Test-linked verification obligations | `benchmarks/` |
| Runnable workflows | `examples/` |
| Versioned AF-IR scientific records | `ir/` |
| Structured validation issues and reports | `validation.py` |
| Backend descriptors, registry, and lowering adapters | `backends/` |
| Typed/JSON parameters, sampling, campaign plans, resumable case execution | `campaigns/` |
| Scientific dataset schemas, arrays, provenance, and splits | `datasets/` |
| Surrogate/ROM models, validation, applicability, neural-operator/PINN contracts | `surrogates/` |

Application geometry, case inputs, and problem-specific sources should stay in
application packages or examples until they become reusable platform concepts.
