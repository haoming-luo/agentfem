# Module Map Reference

- `mesh/`: mesh import/read/write, boundary/cell regions, summaries, tag
  checks, tags, and measures
- `mesh/formats.py`: external CAE inventory, cell/facet conversion, set tags,
  and conversion manifests
- `mesh/abaqus.py`: Abaqus node labels, `*EQUATION` parsing, periodic-cell
  geometry, and source-order VTU output
- `studies.py`: analysis context, physics, dimensions, and modeling assumptions
- `models.py`: lightweight model registry, amplitudes, material assignments,
  checks, summaries, and model-first operators
- `step_providers.py`: extensible analysis/material lowering behind
  `model.step(...)`
- `spaces.py`: function spaces and named functions
- `fields.py`: application-level unknown bundles
- `amplitudes.py`: time histories and scale factors for prescribed data
- `kernel/dofs.py`: low-level dof lookup and field copying
- `constraints/`: constraint containers and semantic Dirichlet constructors
- `loads.py`: loads, natural boundary data, and semantic constructors
- `constitutive/`: local response relations and queryable maturity catalog
- `constitutive/user_material.py`: solver-neutral material-point contract and
  non-executable UMAT/UHYPER bridge specifications
- `materials/`: material-property records and loaders
- `boundary_models/`: weak boundary models
- `forms.py`: UFL weak-form blocks
- `assembly.py`: matrix/vector/lumped assembly
- `operators/`: engineering-level K/M/C/F operators and system containers
- `time/`: explicit/implicit time integration routes, progress, and cadence
- `problems.py`: analysis steps, system problems, and state containers
- `solvers.py`: PETSc solver wrappers and convergence evidence
- `steps.py`: automatic/fixed analysis-step incrementation and cutback policy
- `results/`: simulation results, assembled QoIs, histories, artifacts, and
  campaign/dataset bridge
- `results/finite_strain.py`: named finite-strain visualization fields and
  complete-RVE homogenized histories
- `results/field_catalog.py`: standard field keys, aliases, and context rules
- `results/output.py`: declarative field requests plus compact unified
  XDMF/HDF5 and optional PVD/VTU writers
- `diagnostics.py`: norms and scalar diagnostics
- `io.py`: output writers and scalar logs
- `elements/`: element and integration-policy namespace
- `benchmarks/`: test-linked verification registry
- `ir/`: experimental versioned AF-IR scientific records
- `validation.py`: structured issue codes, paths, hints, and reports
- `backends/`: backend descriptors and advanced lowering adapters
- `campaigns/`: typed parameters, deterministic sampling, case plans,
  resumable execution, and MPI-aware persistence
- `datasets/`: unit/shape-aware learning data and case provenance
- `surrogates/`: baseline learned/ROM models, validation, applicability guards,
  and neural-operator/PINN contracts
