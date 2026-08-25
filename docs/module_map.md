# AgentFEM Module Map

This map links FEM concepts to the current Python modules.
Module paths in this table are relative to `src/agentfem/` unless stated
otherwise; runnable examples remain in the repository-level `examples/`.

| Concept | Module |
| --- | --- |
| Dependency-free public module, Model-verb, CLI-command, and workflow-stage inventories | `_api_contract.py` |
| Mesh import, boundary/cell regions, tags, summaries, checks, and measures | `mesh/` |
| External mesh inventory, cell/facet set conversion, and manifests | `mesh/formats.py` |
| Abaqus migration inventory, element semantics, labels, equations, periodic-cell geometry, and source-order output | `mesh/abaqus.py` |
| Scope-aware Abaqus engineering migration plans and fail-closed project scaffolds | `mesh/abaqus_migration.py` |
| Study context: analysis type, physics, dimension, and assumptions | `studies.py` |
| Shared axisymmetric (r,z) kinematics and full-revolution lowering | `_axisymmetric.py` |
| Standard/Explicit family, equation order, algorithm, and state policy | `procedures.py` |
| Model registry/facade, amplitudes, material assignments, checks, summaries, and model-first operators | `models.py` |
| Built-in scientific Step construction behind the facade | `_step_builders.py` |
| Extensible analysis/material lowering and execution-policy normalization behind `model.step()` | `step_providers.py` |
| Function spaces and Lagrange defaults | `spaces.py` |
| Application-level unknown fields | `fields.py` |
| Time histories and scale factors for prescribed data | `amplitudes.py` |
| Low-level dof lookup and field copying | `kernel/dofs.py` |
| Constraint containers and strong BC construction | `constraints/` |
| Loads and natural boundary data | `loads.py` |
| Constitutive response relations and maturity catalog | `constitutive/` |
| Standard-neutral engineering creep/fatigue assessment consumers and interaction diagrams | `assessments.py` |
| Quadrature-point committed/trial state | `constitutive/quadrature.py` |
| Monotonic cohesive laws, paired interface topology, and physical-keyed state | `interfaces.py`, `cohesive_checkpoint.py` |
| Independent cycle coordinate, cyclic cohesive damage, cycle jumps, and 3D fatigue-crack observations | `fatigue_fracture.py` |
| Finite-strain cohesive forces, named interface composition, energy, wave speeds, and dynamic crack evidence | `fracture.py` |
| Global stateful solid-mechanics procedures | `mechanics/` |
| Material-point contracts and UMAT/UHYPER bridge specifications | `constitutive/user_material.py` |
| Named material definitions, project Python assets, physics-role compatibility, reference records, and property containers | `materials/` |
| Boundary models | `boundary_models/` |
| UFL weak-form blocks, including stiffness, mass, diffusion, and loads | `forms.py` |
| Assembly | `assembly.py` |
| Engineering-level K/M/C/F, transport, SUPG, and reaction operators | `operators/` |
| Central difference, Newmark, generalized-alpha, and runtime cadence | `time/` |
| Analysis steps, system problems, and state containers | `problems.py` |
| Shared transient checkpoint envelope, integrity, accepted-increment cadence, and partition identity | `checkpointing.py` |
| Automatic/fixed incrementation and cutback policy | `steps.py` |
| Linear/nonlinear solvers and convergence evidence | `solvers.py` |
| Scientific results, MPI-safe point/path/integral QoIs, histories, artifacts, and dataset bridge | `results/` |
| Finite-strain fields and periodic-cell homogenization | `results/finite_strain.py` |
| Global and regional L2 projection; engineering-default `S`/`E`/`MISES`, opt-in `SENER`, and explicit processing metadata | `results/projection.py`, `results/core.py` |
| Standard result variables and context-aware aliases | `results/field_catalog.py` |
| Declarative field requests and unified XDMF/HDF5 time-series writer | `results/output.py` |
| Combined field/history/diagnostic/presentation output contracts | `results/plan.py` |
| Reactions, work/energy, thermal balance, progress, and distributed diagnostics | `diagnostics.py`, `results/quantities.py` |
| Output writers and scalar logs | `io.py` |
| Element and integration policies | `elements/` |
| Test-linked verification obligations | `benchmarks/` |
| Versioned external scientific contracts | `integrations/` |
| Runnable workflows | `examples/` |
| Versioned AF-IR scientific records | `ir/` |
| Structured validation issues and reports | `validation.py` |
| Scientific trust, claims, and convergence evidence | `verification.py` |
| Backend descriptors, registry, and lowering adapters | `backends/` |
| Explicit installed/private package discovery and staged registration | `extensions.py` |
| Typed/JSON parameters, sampling, campaign plans, resumable case execution | `campaigns/` |
| Scientific dataset schemas, arrays, provenance, splits, and FEM-field export | `datasets/` |
| Unified learning entry; neural-field objectives, conditions, sampling, inferred parameters, and compatibility access | `learning/` |
| Surrogate/ROM models, validation, applicability, observation grids, neural-operator/PINN contracts | `surrogates/` |

Application geometry, case inputs, and problem-specific sources should stay in
application packages or examples until they become reusable platform concepts.
