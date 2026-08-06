---
title: Python API
description: Automatically generated public AgentFEM Python API index.
---

# Python API

This index is generated from the public workflow modules declared by
AgentFEM. It is a discovery surface: detailed scientific meaning, maturity,
and evidence remain in the linked guides and scientific function reference.

!!! info "Generated reference"
    Run `python build_docs.py` to refresh this page after public API changes.

## `agentfem.studies`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Study` | Early analysis context for a finite-element workflow. |
| function | `define(*, analysis: str, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None, preferred_procedure: str \| None = None) -> Study` | Define a general finite-element study context. |
| function | `linear_static(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None) -> Study` | Define a linear static study. |
| function | `nonlinear_static(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None) -> Study` | Define a nonlinear static study. |
| function | `static_solid(*, dimension: int, assumption: str \| None = None, nonlinear: bool = False, name: str \| None = None) -> Study` | Define a static solid-mechanics study with concise engineering syntax. |
| function | `steady_heat_transfer(*, dimension: int, name: str \| None = None) -> Study` | Define steady heat conduction, including source, flux, and convection. |
| function | `first_order_transient(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None) -> Study` | Define a first-order transient study. |
| function | `transient(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None) -> Study` | Compatibility alias for ``first_order_transient``. |
| function | `transient_heat_transfer(*, dimension: int, name: str \| None = None) -> Study` | Define an implicit first-order heat-transfer study. |
| function | `nonlinear_transient(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None, procedure: str \| None = None) -> Study` | Define a nonlinear time-domain study. |
| function | `creep_solid(*, dimension: int = 3, assumption: str \| None = None, name: str \| None = None) -> Study` | Define an implicit quasi-static creep study. |
| function | `second_order_dynamics(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None, procedure: str \| None = None) -> Study` | Define a second-order dynamics study. |
| function | `implicit_dynamics(*, physics: str, dimension: int, assumption: str \| None = None, method: str = 'newmark', name: str \| None = None) -> Study` | Define second-order dynamics with a Standard/implicit preference. |
| function | `explicit_dynamics(*, physics: str, dimension: int, assumption: str \| None = None, name: str \| None = None) -> Study` | Define second-order dynamics with an Explicit preference. |
| function | `dynamic_solid(*, dimension: int, assumption: str \| None = None, method: str = 'explicit', name: str \| None = None) -> Study` | Define structural dynamics without repeating the physics name. |

## `agentfem.mesh`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `FEMMesh` | DOLFINx mesh plus optional cell and facet tags. |
| class | `TagSummary` | Summary of integer mesh tags on one topological entity dimension. |
| class | `MeshSummary` | Human- and agent-readable mesh summary. |
| class | `BoundaryRegion` | Named exterior boundary region on a mesh. |
| class | `CellRegion` | Named cell/material region on a mesh. |
| class | `NodeRegion` | Named source-node region, including high-order geometry nodes. |
| function | `import_gmsh_model(model, comm: MPI.Comm = MPI.COMM_WORLD, *, model_rank: int = 0, gdim: int = 3) -> FEMMesh` | Convert an in-memory Gmsh model to a DOLFINx mesh. |
| function | `rectangle(lower, upper, cells, comm: MPI.Comm = MPI.COMM_WORLD, *, cell_type: str \| mesh.CellType = 'quadrilateral')` | Create a structured 2D rectangular mesh. |
| function | `cuboid(lower, upper, cells, comm: MPI.Comm = MPI.COMM_WORLD, *, cell_type: str \| mesh.CellType = 'hexahedron')` | Create a structured 3D cuboid mesh. |
| function | `read_gmsh_mesh(path: str \| Path, comm: MPI.Comm = MPI.COMM_WORLD, *, model_rank: int = 0, gdim: int = 3) -> FEMMesh` | Read a ``.msh`` file with Gmsh and convert it to a DOLFINx mesh. |
| function | `require_gmsh()` | Return the optional Gmsh Python API used only for direct Gmsh import. |
| function | `optional_mesh_capabilities() -> tuple[dependencies.DependencyStatus, ...]` | Return availability of optional mesh-format integrations. |
| function | `read_xdmf_mesh(path: str \| Path, comm: MPI.Comm = MPI.COMM_WORLD, *, mesh_name: str = 'mesh', cell_tags_name: str \| None = None, facet_tags_name: str \| None = None) -> FEMMesh` | Read a DOLFINx XDMF mesh and optional cell/facet meshtags. |
| function | `write_xdmf_mesh(path: str \| Path, domain, comm: MPI.Comm \| None = None, *, mode: str = 'w') -> None` | Write a DOLFINx mesh to XDMF. |
| function | `convert_external_mesh_to_xdmf(*args, **kwargs)` | Convert Abaqus/NASTRAN/COMSOL-like external meshes to XDMF. |
| function | `convert_external_mesh_bundle(*args, **kwargs)` | Convert selected source topologies into explicit solver-domain files. |
| function | `inspect_external_mesh(path)` | Inventory external element blocks and named sets before conversion. |
| function | `read_abaqus_mesh(path: str \| Path, converted_path: str \| Path, comm: MPI.Comm = MPI.COMM_WORLD, *, cell_type: str \| None = None, reuse_conversion: bool = True) -> abaqus.AbaqusMeshImport` | Convert and read an Abaqus mesh while retaining source node labels. |
| function | `external_mesh_formats() -> dict[str, str]` | Return common external formats supported through optional ``meshio``. |
| function | `read_converted_xdmf(conversion, comm: MPI.Comm = MPI.COMM_WORLD, *, mesh_name: str = 'Grid', tag_grid_name: str = 'Grid') -> FEMMesh` | Read a :class:`mesh.formats.MeshConversionResult` into DOLFINx. |
| function | `summarize_tags(tags) -> TagSummary \| None` | Summarize a DOLFINx meshtags object. |
| function | `summarize_mesh(domain, cell_tags = None, facet_tags = None) -> MeshSummary` | Return local/global mesh size and tag summaries. |
| function | `require_tags(tags, required: int \| tuple[int, ...] \| list[int], *, name: str = 'tags', comm = None) -> None` | Raise if required tags are absent globally. |
| function | `require_cell_tags(cell_tags, required: int \| tuple[int, ...] \| list[int], *, comm = None) -> None` | Require cell/material region tags. |
| function | `require_facet_tags(facet_tags, required: int \| tuple[int, ...] \| list[int], *, comm = None) -> None` | Require boundary/facet tags. |
| function | `boundary(domain, marker, *, name: str = 'boundary', tag: int = 1) -> BoundaryRegion` | Create a named exterior boundary region from a geometric marker. |
| function | `tagged_boundary_region(domain, facet_tags, *, tag: int, name: str = 'tagged_boundary', marker = None) -> BoundaryRegion` | Create a boundary whose canonical selection is an imported facet tag. |
| function | `audit_boundary_region(region: BoundaryRegion, *, strict: bool = False) -> dict[str, object]` | Inspect a boundary's identity, size, orientation, and tag/marker agreement. |
| function | `face(domain, *, axis: str \| int, value: float, name: str \| None = None, tag: int = 1, tolerance: float \| None = None) -> BoundaryRegion` | Create a planar exterior boundary region such as ``x = 0``. |
| function | `boundary_region(domain, marker, *, name: str = 'boundary', tag: int = 1) -> BoundaryRegion` | Alias for ``boundary`` when a more explicit name reads better. |
| function | `cell_region(domain, cell_tags = None, *, tag: int, name: str = 'cell_region', marker = None) -> CellRegion` | Create a named cell/material region. |
| function | `region_measure(location)` | Return a region's restricted measure or pass through a measure. |
| function | `region_marker(location)` | Return a region's marker or pass through a marker callable. |
| function | `cells(domain, *, name: str, where, tag: int = 1) -> CellRegion` | Create a named cell region from a selector. |
| function | `partition_cells(domain, **regions) -> RegionSet` | Partition mesh cells into named cell regions. |
| function | `partition_boundaries(domain, **regions) -> RegionSet` | Create named exterior boundary regions from selectors. |
| function | `locate_cells(domain, marker)` | Locate cells using a geometrical marker. |
| function | `mark_cells(domain, cells, tag: int)` | Create cell meshtags for a set of cells. |
| function | `mark_cell_regions(domain, tag_to_marker: dict[int, object])` | Create cell meshtags from several geometric cell markers. |
| function | `tag_field(domain, tags, *, name: str = 'Tag')` | Create a DG0 visualization field from cell tags. |
| function | `locate_boundary_facets(domain, marker)` | Locate exterior facets using a geometrical marker. |
| function | `mark_facets(domain, facets, tag: int)` | Create a meshtags object for a set of facets. |
| function | `mark_boundary_facets(domain, marker, tag: int)` | Locate and tag exterior facets in one step. |
| function | `boundary_measure(domain, facet_tags = None)` | Create a boundary integration measure. |
| function | `cell_measure(domain, cell_tags = None)` | Create a domain integration measure. |
| function | `facet_normal(domain)` | Return the outward facet normal for boundary models. |
| function | `tagged_boundary_measure(domain, marker, tag: int)` | Locate/tag exterior facets and return ``(ds, facet_tags)``. |
| class | `RegionSet` | Named collection of regions sharing one mesh tag object. |
| class | `Selector` | Boolean selector evaluated on coordinate arrays. |
| function | `ball(center, radius: float) -> Selector` | Select points inside a 3D ball. |
| function | `box(lower, upper) -> Selector` | Select points inside an axis-aligned box. |
| function | `disk(center, radius: float) -> Selector` | Select points inside a 2D disk. |
| function | `layer(axis: str \| int, lower = None, upper = None) -> Selector` | Select points inside a coordinate interval along one axis. |
| function | `plane(axis: str \| int, value: float, *, tolerance: float = 1e-12) -> Selector` | Select points near a coordinate plane such as ``x = 0``. |
| function | `where(predicate, *, name: str \| None = None) -> Selector` | Create a selector from a vectorized coordinate predicate. |

## `agentfem.models`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Model` | Finite-element model registry for humans and agents. |
| function | `create(*, study, mesh = None, name: str = 'model', units = None) -> Model` | Create a lightweight model registry. |

## `agentfem.fields`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Field` | Tensor-like finite-element field with immediate-value algebra. |
| class | `UnknownField` | Finite-element unknown bundle for application-level workflows. |
| class | `DisplacementPressureUnknown` | Mixed displacement/pressure unknown for hybrid solid mechanics. |
| function | `scalar_unknown(domain, *, name: str = 'Unknown', degree: int = 1, value = 0.0) -> UnknownField` | Create a scalar finite-element unknown. |
| function | `vector_unknown(domain, *, name: str = 'Unknown', degree: int = 1, dim: int \| None = None, value = 0.0) -> UnknownField` | Create a vector finite-element unknown. |
| function | `displacement(domain, *, degree: int = 1, dim: int \| None = None, value = 0.0) -> UnknownField` | Create a displacement unknown for mechanics workflows. |
| function | `displacement_pressure(domain, *, displacement_degree: int = 2, pressure_degree: int = 0, name: str = 'DisplacementPressure') -> DisplacementPressureUnknown` | Create a mixed displacement/pressure unknown. |
| function | `temperature(domain, *, degree: int = 1, value = 0.0) -> UnknownField` | Create a temperature unknown for heat-transfer workflows. |
| function | `wrap(function, *, name: str \| None = None) -> Field` | Wrap a DOLFINx function as an AgentFEM field. |
| function | `unwrap(field_or_function)` | Return the underlying DOLFINx function when given an AgentFEM field. |
| function | `empty_like(field_or_function, *, name: str \| None = None) -> Field` | Create a zero-valued field with the same function space. |
| function | `compute(expression, *, name: str \| None = None) -> Field` | Return a computed field. |
| function | `assign(target, source) -> None` | Assign a scalar, compatible field, or DOLFINx function into ``target``. |
| function | `dot(left, right) -> float` | Return the distributed algebraic dot product of two compatible fields. |
| function | `weighted_dot(left, weights, right = None) -> float` | Return ``left^T diag(weights) right`` for compatible fields. |
| function | `norm(field, *, weight = None) -> float` | Return the distributed algebraic norm of a field. |
| function | `require_same_space(left, right) -> None` | Raise if two fields/functions are not on the same function space. |
| function | `same_space(left, right) -> bool` | Return whether two fields/functions share the same function space. |

## `agentfem.materials`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `MaterialRecord` | Material-library record before conversion to a constitutive law. |
| function | `list_material_models(name: str) -> tuple[str, ...]` | List model names available for one material. |
| function | `list_materials(*, model: str \| None = None) -> tuple[str, ...]` | List available material names, optionally filtered by model. |
| function | `load_material(name: str, model: str \| None = None)` | Load one material model and return a constitutive material object. |
| function | `material_record(name: str) -> MaterialRecord` | Return a validated material record without constructing a model object. |
| function | `register_material(name: str, data: dict, *, overwrite: bool = False) -> None` | Register or override a material record in memory. |
| class | `ElasticAnisotropic2DProperties` | 2D linear-elastic properties using engineering-strain Voigt notation. |
| class | `ElasticIsotropicProperties` | Isotropic linear-elastic material properties. |
| class | `ThermoElasticIsotropicProperties` | Isotropic thermoelastic and heat-conduction properties. |
| function | `validate_material_record(name: str, record: dict) -> None` | Validate one material-centered library record. |

## `agentfem.mechanics`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CreepEnergyFrame` | Public AgentFEM object. |
| class | `CreepIncrementInfo` | Public AgentFEM object. |
| class | `CreepPathInfo` | Public AgentFEM object. |
| class | `ImplicitCreepStep` | Adaptive backward-Euler creep step with global Newton equilibrium. |
| function | `implicit_creep_step(*, displacement, material, duration: float, external_force, constraints = (), study = None, incrementation = None, solver_options = None, quadrature_degree: int = 2, progress = True, status_file = None, amplitude = None, name: str = 'implicit_creep') -> ImplicitCreepStep` | Build the first global 3D implicit power-law creep step. |
| class | `J2IncrementInfo` | Public AgentFEM object. |
| class | `J2LoadPathInfo` | Public AgentFEM object. |
| class | `J2PlasticityStep` | Incremental global equilibrium for 3D small-strain J2 plasticity. |
| function | `j2_plasticity_step(*, displacement, material, external_force, constraints = (), study = None, incrementation = None, solver_options = None, quadrature_degree: int = 2, progress = True, status_file = None, amplitude = None, name: str = 'j2_plasticity') -> J2PlasticityStep` | Build a global 3D J2 step from a displacement and load operator. |

## `agentfem.constitutive`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ConstitutiveCapability` | What a material capability can truthfully do in this release. |
| function | `capabilities() -> tuple[ConstitutiveCapability, ...]` | Return all constitutive capabilities in stable name order. |
| function | `capability(name: str) -> ConstitutiveCapability` | Return one capability or raise with the available names. |
| class | `ArrheniusPowerLawCreep` | Temperature-dependent Mises power-law creep. |
| class | `CreepDamageState` | Local creep strain and scalar continuum-damage state. |
| class | `CreepDamageUpdate` | Accepted material-point increment from a creep-damage law. |
| class | `CreepHistory` | Integrated piecewise-constant stress history. |
| class | `ImplicitCreepState` | Committed small-strain creep state at one integration point. |
| class | `ImplicitCreepUpdate` | Backward-Euler material-point update and consistent tangent. |
| class | `IsotropicPowerLawCreepMaterial` | Isotropic elasticity with an implicit Mises power-law creep branch. |
| class | `KachanovRabotnovCreep` | Classical scalar Kachanov--Rabotnov creep-damage coupling. |
| class | `ModifiedThetaProjection` | Three-parameter modified-theta representation of a creep curve. |
| class | `PowerLawCreep` | Mises time-hardening creep law. |
| class | `SinhCreep` | Stress-sensitive hyperbolic-sine Mises creep law. |
| function | `integrate_stress_history(law: PowerLawCreep, times, interval_stresses) -> CreepHistory` | Integrate a piecewise-constant scalar or tensor stress history. |
| function | `isotropic_power_law(*, young: float, poisson: float, density: float, coefficient: float, stress_exponent: float, time_exponent: float = 0.0, reference_stress: float = 1.0, reference_time: float = 1.0, name: str = 'isotropic power-law creep') -> IsotropicPowerLawCreepMaterial` | Create one Abaqus-style material record with elastic and creep data. |
| function | `anisotropic_stress_2d(displacement, properties: ElasticAnisotropic2DProperties, *, study = None)` | 2D anisotropic stress from engineering-strain Voigt stiffness. |
| function | `anisotropic_elastic_2d(*, stiffness_voigt, density: float, name: str = 'anisotropic elastic 2D') -> ElasticAnisotropic2DProperties` | Create 2D anisotropic linear-elastic properties. |
| function | `estimate_elastic_wave_speeds(material) -> tuple[float, float]` | Return approximate ``(pressure_speed, shear_speed)`` for a material. |
| function | `isotropic_stress(displacement, properties: ElasticIsotropicProperties, *, study = None)` | Small-strain isotropic stress, ``sigma(u)``. |
| function | `isotropic_elastic(*, young: float, density: float, poisson: float, name: str = 'isotropic elastic') -> ElasticIsotropicProperties` | Create isotropic linear-elastic properties. |
| function | `thermal_expansion_stress(temperature, properties, *, study = None, dimension = None)` | Return positive ``C:epsilon_thermal`` for an equivalent thermal load. |
| function | `thermal_strain(temperature, properties, *, dimension: int)` | Return isotropic free thermal strain ``alpha (T-T_ref) I``. |
| function | `thermoelastic(*, young: float, density: float, poisson: float, thermal_expansion: float, conductivity: float, specific_heat: float, reference_temperature: float = 293.15, name: str = 'isotropic thermoelastic') -> ThermoElasticIsotropicProperties` | Create one material record for sequential thermal-stress workflows. |
| function | `thermoelastic_stress(displacement, temperature, properties, *, study = None)` | Small-strain isotropic stress including thermal eigenstrain. |
| function | `orthotropic_plane_stress_2d(*, ex: float, ey: float, nuxy: float, gxy: float, density: float, name: str = 'orthotropic plane-stress elastic 2D') -> ElasticAnisotropic2DProperties` | Create 2D orthotropic plane-stress elastic properties. |
| function | `stress(displacement, properties, *, study = None)` | Dispatch to the matching elastic stress relation. |
| class | `BasquinCurve` | Fully reversed stress-life curve ``sigma_a = sigma_f' (2N)^b``. |
| class | `FatigueAssessment` | Auditable stress-life assessment derived from one scalar history. |
| class | `FatigueBlock` | One constant-amplitude block for cumulative-damage assessment. |
| class | `StressCycle` | One rainflow-counted stress cycle or residual half-cycle. |
| class | `TabulatedSNCurve` | Log-log interpolated S-N data with explicit extrapolation policy. |
| function | `assess_history(history, curve, *, ultimate_strength: float \| None = None, source: str \| None = None) -> FatigueAssessment` | Return cycles, Miner damage, and repeated-history life together. |
| function | `assess_result_history(result, history_name: str, curve, *, ultimate_strength: float \| None = None) -> FatigueAssessment` | Assess one named ``SimulationResult`` history with provenance. |
| function | `damage_from_history(history, curve, *, ultimate_strength: float \| None = None) -> float` | Rainflow count a stress history and apply Palmgren-Miner damage. |
| function | `goodman_amplitude(stress_amplitude: float, mean_stress: float, ultimate_strength: float) -> float` | Return fully reversed amplitude using the linear Goodman correction. |
| function | `life_scale_factor(blocks: Iterable[FatigueBlock], curve) -> float` | Return the number of repeated block sequences to Miner damage one. |
| function | `miner_damage(blocks: Iterable[FatigueBlock], curve) -> float` | Return Palmgren-Miner cumulative damage ``sum(n_i / N_i)``. |
| function | `rainflow_cycles(history) -> tuple[StressCycle, ...]` | Count full and residual half-cycles from a scalar stress history. |
| function | `turning_points(history) -> np.ndarray` | Return endpoints and local reversals from a scalar stress history. |
| class | `FiniteStrainKinematics` | Standard total-Lagrangian kinematics derived from one displacement. |
| class | `MixedNeoHookeanProperties` | Isochoric Neo-Hookean solid with an independent pressure field. |
| class | `NeoHookeanProperties` | Compressible Neo-Hookean parameters derived from ``E`` and ``nu``. |
| function | `kinematics(displacement) -> FiniteStrainKinematics` | Return the standard finite-strain kinematic measures for ``u``. |
| function | `mixed_neo_hookean(*, young: float, poisson: float, density: float \| None = None, name: str = 'mixed Neo-Hookean') -> MixedNeoHookeanProperties` | Create a pressure-displacement Neo-Hookean material. |
| function | `neo_hookean(*, young: float, poisson: float, density: float \| None = None, name: str = 'compressible Neo-Hookean') -> NeoHookeanProperties` | Create a compressible Neo-Hookean material. |
| class | `J2LinearIsotropicHardening` | Rate-independent von Mises plasticity with linear isotropic hardening. |
| class | `J2PlasticState` | History variables for small-strain isotropic J2 plasticity. |
| class | `J2Update` | Result of one radial-return material-point update. |
| class | `UniaxialPlasticState` | History variables for the exact one-dimensional counterpart. |
| function | `update_uniaxial(total_strain: float, material: J2LinearIsotropicHardening, state: UniaxialPlasticState \| None = None) -> tuple[float, UniaxialPlasticState]` | Return stress and state for a one-dimensional bilinear material test. |
| function | `von_mises(stress) -> float` | Return ``sqrt(3/2 s:s)`` for a symmetric Cauchy stress. |
| class | `CreepQuadratureState` | Committed/trial integration-point state for implicit 3D creep. |
| class | `J2QuadratureState` | Committed/trial integration-point state for 3D small-strain J2. |
| class | `QuadratureField` | A DOLFINx quadrature function with an explicit NumPy point view. |
| class | `QuadratureTransaction` | Shared trial/commit/rollback contract for integration-point state. |
| class | `AbaqusUserMaterialBridge` | Truthful capability description for an intended UMAT/UHYPER adapter. |
| class | `MaterialPointInput` | Solver-neutral finite-strain input for one material-point update. |
| class | `MaterialPointOutput` | Constitutive response returned to a nonlinear finite-element driver. |
| class | `UserMaterial` | Protocol implemented by native or adapted material-point models. |

## `agentfem.coordinates`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CartesianSystem` | Right-handed orthonormal Cartesian coordinate system. |
| class | `ReferencePoint` | Named engineering point used for remote resultants and kinematics. |
| function | `cartesian(*, origin = None, axes = None, x = None, y = None, z = None, name = 'local') -> CartesianSystem` | Create a Cartesian system from a matrix or named basis vectors. |
| function | `reference_point(coordinates, *, name = 'reference_point', system = None) -> ReferencePoint` | Create a named engineering reference point. |

## `agentfem.constraints`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `DirichletConstraint` | Strong Dirichlet constraint and its optional mutable value object. |
| class | `TimeDependentDirichlet` | Dirichlet constraint driven by an amplitude. |
| class | `RemoteDisplacementConstraint` | Rigid boundary motion prescribed about a named reference point. |
| class | `PrescribedValuePath` | Update ordinary strong boundary values along a normalized step path. |
| function | `prescribed_value_path(constraints) -> PrescribedValuePath` | Create a normalized load-factor driver from registered constraints. |
| function | `dirichlet_constraints(constraints) -> tuple[object, ...]` | Return concrete Dirichlet assets from nested model constraint sets. |
| function | `scalar_dirichlet(V, marker = None, value = 0.0, *, location = None, on = None, name: str = 'dirichlet') -> DirichletConstraint` | Semantic wrapper for scalar essential boundary data. |
| function | `component_dirichlet(V, component: int, marker = None, value = 0.0, *, location = None, on = None, name: str = 'dirichlet') -> DirichletConstraint` | Semantic wrapper for vector-component essential boundary data. |
| function | `dirichlet(V, marker = None, value = 0.0, *, component: int \| None = None, location = None, on = None, name: str = 'dirichlet') -> DirichletConstraint` | Create scalar or component-wise Dirichlet data from one entry point. |
| function | `time_dependent_component_dirichlet(target, component: int, marker = None, value = None, *, amplitude = None, location = None, on = None, name: str = 'time_dependent_dirichlet') -> TimeDependentDirichlet` | Create a component-wise Dirichlet constraint driven by an amplitude. |
| function | `time_dependent_scalar_dirichlet(target, marker = None, value = None, *, amplitude = None, location = None, on = None, name: str = 'time_dependent_dirichlet') -> TimeDependentDirichlet` | Create a scalar Dirichlet constraint driven by an amplitude. |
| function | `apply_dirichlet_bcs(function, bcs) -> None` | Apply strong Dirichlet boundary conditions to a function vector. |
| function | `fixed(target, *, location = None, on = None, value = 0.0, components: int \| tuple[int, ...] \| list[int] \| None = None, name: str \| None = None) -> 'ConstraintSet'` | Create fixed-value Dirichlet constraints for an application field. |
| function | `fixed_component(target, component: int, *, location = None, on = None, value = 0.0, name: str \| None = None)` | Create a fixed-value constraint for one vector component. |
| function | `symmetry(target, *, on = None, location = None, normal_axis: int \| str, value = 0.0, name: str \| None = None) -> 'ConstraintSet'` | Apply an axis-aligned solid-mechanics symmetry condition. |
| function | `roller(target, *, on = None, location = None, normal_axis: int \| str, value = 0.0, name: str \| None = None) -> 'ConstraintSet'` | Alias for an axis-aligned frictionless roller/support condition. |
| function | `fixed_all(target, *, location = None, on = None, value = 0.0, name: str \| None = None)` | Create a scalar/all-dof fixed-value constraint. |
| function | `prescribed(target, *, on = None, location = None, value = 0.0, component = None, components = None, name: str \| None = None)` | Create prescribed scalar or vector-component values. |
| function | `clamped(target, *, on = None, location = None, value = 0.0, name: str \| None = None)` | Fix every displacement component on a support boundary. |
| function | `prescribed_temperature(target, value, *, on = None, location = None, name: str \| None = None)` | Prescribe temperature on a named boundary. |
| function | `remote_displacement(target, *, reference_point, on = None, location = None, translation = None, rotation = None, system = None, name: str = 'remote_displacement') -> RemoteDisplacementConstraint` | Prescribe rigid translation/rotation of a solid boundary. |
| class | `PeriodicProjectionConstraint` | Projection-style periodic constraint for explicit field updates. |
| function | `periodic(target, *, master, slave, match_axis: str \| int = 0, method: str = 'projection', tolerance: float = 1e-12, name: str = 'periodic')` | Create a periodic constraint with an explicit method choice. |
| function | `periodic_projection(target, *, master, slave, match_axis: str \| int = 0, tolerance: float = 1e-12, name: str = 'periodic_projection') -> PeriodicProjectionConstraint` | Create component-wise dof pairs for projection-style periodicity. |
| class | `PeriodicConstraintSpec` | Geometric description of a periodic constraint. |
| class | `ConstraintSet` | Collection of constraints used by assembly or field updates. |
| class | `AbaqusPeriodicConstraint` | Periodic cell equations controlled by a macroscopic deformation gradient. |
| class | `AffineReduction` | Sparse serial representation of ``u = T q + offset``. |
| class | `DistributedAffineReduction` | Homogeneous correction space for a distributed affine constraint. |
| function | `abaqus_periodic_cell(target, *, nodes: AbaqusNodeTable, equations: AbaqusEquationSet, deformation_gradient, anchor_node: int, reference_nodes, tolerance: float = 1e-09, name: str = 'abaqus_periodic_cell') -> AbaqusPeriodicConstraint` | Create exact periodic-cell constraints from Abaqus equation data. |

## `agentfem.amplitudes`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Amplitude` | Named scalar history function. |
| function | `as_amplitude(value, *, name: str = 'amplitude') -> Amplitude` | Convert a scalar, callable, or ``Amplitude`` into an ``Amplitude``. |
| function | `constant(value: float, *, name: str = 'constant') -> Amplitude` | Create a constant amplitude. |
| function | `ramp(start_value: float = 0.0, end_value: float = 1.0, *, start_time: float = 0.0, end_time: float = 1.0, name: str = 'ramp') -> Amplitude` | Create a clipped linear ramp amplitude. |
| function | `tabular(times, values, *, name: str = 'tabular', left: float \| None = None, right: float \| None = None) -> Amplitude` | Create a linearly interpolated tabular amplitude. |
| function | `sine(amplitude: float = 1.0, frequency: float = 1.0, *, phase: float = 0.0, offset: float = 0.0, name: str = 'sine') -> Amplitude` | Create a sinusoidal amplitude. |
| function | `gaussian_modulated_sine(amplitude: float, frequency: float, width: float, *, center: float \| None = None, phase: float = 0.0, name: str = 'gaussian_modulated_sine') -> Amplitude` | Create a Gaussian-windowed sinusoidal pulse. |

## `agentfem.loads`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `time_dependent_component_dirichlet(V, component: int, marker, time_function)` | Compatibility wrapper for time-dependent component Dirichlet constraints. |
| function | `apply_dirichlet_bcs(function, bcs) -> None` | Apply strong Dirichlet boundary conditions to a function vector. |
| function | `constant_time_function(value: float, name: str = 'constant') -> amplitudes.Amplitude` | Represent a constant value with the same interface as transient data. |
| class | `BodyLoad` | Domain source/body-force term for a weak form. |
| class | `GravityLoad` | Gravity body force ``rho g`` over a material domain. |
| class | `CentrifugalLoad` | Rotating-frame body force ``rho omega x (omega x r)`` outward. |
| class | `BoundaryLoad` | Boundary flux/traction term for a weak form. |
| class | `PressureLoad` | Pressure load pulled back to a reference boundary measure. |
| class | `HydrostaticPressureLoad` | Pressure varying with elevation from a reference free surface. |
| class | `SurfaceResultantLoad` | A requested total force uniformly distributed over a reference boundary. |
| class | `DistributedCouplingLoad` | Force and moment distributed over a continuum surface. |
| class | `NeumannLoad` | Natural boundary condition applied through the weak-form right hand side. |
| class | `AmplitudeLoad` | A spatial load multiplied by one reusable scalar amplitude. |
| class | `LoadSet` | Ordered collection of weak-form load terms. |
| function | `body_load(value, measure = ufl.dx, *, name: str = 'body_load', domain = None, target = None) -> BodyLoad` | Create a domain source/body-force load. |
| function | `body_force(value, *, domain = None, target = None, measure = ufl.dx, system = None, name: str = 'body_force') -> BodyLoad` | Create a mechanical body-force load in global or local components. |
| function | `gravity(acceleration, *, density, domain = None, target = None, region = None, measure = None, system = None, name: str = 'gravity') -> GravityLoad` | Create a gravity load from acceleration and material density. |
| function | `centrifugal(angular_velocity, *, density, center = None, domain = None, target = None, region = None, measure = None, name: str = 'centrifugal') -> CentrifugalLoad` | Create the outward body force caused by constant angular velocity. |
| function | `heat_source(value, *, domain = None, target = None, measure = ufl.dx, name: str = 'heat_source') -> BodyLoad` | Create a volumetric heat-source load. |
| function | `boundary_load(value, measure = None, *, location = None, on = None, name: str = 'boundary_load') -> BoundaryLoad` | Create a generic natural boundary load. |
| function | `neumann(value, measure, *, name: str = 'neumann_load') -> NeumannLoad` | Create a Neumann force/flux/traction term for the weak RHS. |
| function | `with_amplitude(load, amplitude, *, domain = None, name: str \| None = None) -> AmplitudeLoad` | Drive an existing load by a scalar amplitude multiplier. |
| function | `traction(value, *, location = None, on = None, system = None, name: str = 'traction') -> BoundaryLoad` | Create a traction in global or an explicit local coordinate system. |
| function | `surface_force(resultant, *, location = None, on = None, reference_measure: float \| None = None, system = None, name: str = 'surface_force') -> SurfaceResultantLoad` | Distribute a total reference-configuration force over a boundary. |
| function | `distributing_coupling(force, *, moment = None, reference_point = None, location = None, on = None, system = None, name: str = 'distributing_coupling') -> DistributedCouplingLoad` | Distribute force/moment over a surface with tributary-area weighting. |
| function | `remote_force(force, *, reference_point, moment = None, location = None, on = None, system = None, name: str = 'remote_force') -> DistributedCouplingLoad` | Apply a reference-point force/moment through a continuum surface. |
| function | `pressure(value, *, location = None, on = None, normal = None, configuration: str = 'reference', displacement = None, name: str = 'pressure') -> PressureLoad` | Create inward pressure on a reference or current boundary. |
| function | `hydrostatic_pressure(*, density, gravity, reference_point, reference_pressure = 0.0, on = None, location = None, clip_at_zero: bool = True, configuration: str = 'reference', displacement = None, name: str = 'hydrostatic_pressure') -> HydrostaticPressureLoad` | Create ``p = p_ref + rho g dot (x - x_ref)`` on a boundary. |
| function | `heat_flux(value, *, location = None, on = None, name: str = 'heat_flux') -> BoundaryLoad` | Create a prescribed heat flux applied on a boundary region. |
| function | `body_force_form(force, test_function)` | Create a body-force virtual-work form. |
| function | `boundary_traction_form(traction, test_function, ds_measure)` | Create a boundary-traction virtual-work form. |

## `agentfem.operators`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `OperatorForm` | Named scientific operator with a current backend expression. |
| function | `action(operator, field)` | Return the algebraic action of a matrix-like operator on a field. |
| function | `assemble_matrix(operator, *, bcs = None, backend = None)` | Assemble an operator-level matrix from an ``OperatorForm`` or UFL form. |
| function | `assemble_vector(operator, *, backend = None)` | Assemble an operator-level vector from an ``OperatorForm`` or UFL form. |
| function | `bilinear_form(operator, left, right) -> float` | Return the algebraic scalar ``left^T operator right``. |
| function | `body_force_vector(force, test_function, *, measure = ufl.dx) -> OperatorForm` | Create a body-force/source vector ``F``. |
| function | `boundary_load_vector(load = None, test_function = None, *, value = None, target = None, measure = None, location = None) -> OperatorForm` | Create a boundary load vector ``F_boundary``. |
| function | `boundary_force_vector(*, target, value = None, location = None, load = None) -> OperatorForm` | Create a boundary force vector from a load object or value/location pair. |
| function | `boundary_model_vector(boundary_model, velocity, test_function = None) -> OperatorForm` | Create a vector contribution from a weak boundary model. |
| function | `capacity_operator(temperature, capacity, *, measure = ufl.dx) -> OperatorForm` | Create a capacity/storage operator ``C``. |
| function | `combine(*operators, name: str = 'combined_operator', kind: str = 'combined_operator') -> OperatorForm` | Combine operator forms or raw UFL expressions into one operator form. |
| function | `compile_form(operator: OperatorForm, *, backend = None)` | Compile an ``OperatorForm`` or raw UFL form. |
| function | `conduction_operator(temperature, conductivity, *, measure = ufl.dx) -> OperatorForm` | Create a conduction/diffusion stiffness operator ``K``. |
| function | `damping_operator(trial_function, test_function = None, coefficient = None, *, measure = ufl.dx) -> OperatorForm` | Create a viscous damping operator ``C``. |
| function | `dual_product(vector_operator, field) -> float` | Return the global discrete pairing ``field^T vector_operator``. |
| function | `diffusion_operator(trial_function, test_function = None, conductivity = None, *, measure = ufl.dx) -> OperatorForm` | Create a scalar diffusion/conduction operator. |
| function | `force_vector(target, loads = None, *, load = None) -> OperatorForm` | Create a total force/source vector from one or more load objects. |
| function | `form_arity(expression) -> int \| None` | Return the number of UFL arguments, or ``None`` for opaque backends. |
| function | `flux_vector(flux, target, *, measure = None, location = None) -> OperatorForm` | Create a prescribed scalar boundary-flux vector. |
| function | `heat_capacity_operator(temperature, capacity, *, measure = ufl.dx) -> OperatorForm` | Create a heat-capacity operator ``C`` for transient heat problems. |
| function | `heat_capacity_vector(previous_temperature, temperature, capacity, *, measure = ufl.dx) -> OperatorForm` | Create the known heat-capacity vector ``C * T_previous``. |
| function | `heat_conduction_operator(temperature, conductivity, *, measure = ufl.dx) -> OperatorForm` | Create a heat-conduction operator ``K`` for ``-div(k grad(T))``. |
| function | `heat_source_vector(source, temperature, *, measure = ufl.dx) -> OperatorForm` | Create a heat-source vector ``Q`` for a temperature unknown. |
| function | `inertial_force_vector(acceleration, target, density = 1.0, *, measure = ufl.dx) -> OperatorForm` | Create the inertial virtual-work vector ``F_inertia = M a``. |
| function | `load_vector(target, loads = None, *, load = None) -> OperatorForm` | Create a total external-load vector ``F`` for a target unknown. |
| function | `lumped_mass(V, density = 1.0, *, measure = ufl.dx)` | Assemble a lumped mass vector for explicit dynamics. |
| function | `lumped_operator(V, coefficient = 1.0, *, measure = ufl.dx)` | Assemble a generic lumped diagonal operator. |
| function | `mass_action_vector(field, target, coefficient = 1.0, *, measure = ufl.dx) -> OperatorForm` | Create a vector from a mass-like operator acting on a known field. |
| function | `mass_operator(trial_function, test_function = None, density = 1.0, *, measure = ufl.dx) -> OperatorForm` | Create a consistent mass operator ``M``. |
| function | `linearize(residual, unknown, direction = None, *, name: str = 'K_t') -> OperatorForm` | Differentiate a residual to obtain its consistent tangent operator. |
| function | `residual_operator(expression, *, name: str = 'R', family: str = 'nonlinear', metadata: dict[str, object] \| None = None) -> OperatorForm` | Wrap a nonlinear weak residual ``R(u; v)`` as a public operator. |
| function | `rayleigh_damping(mass, stiffness, *, mass_coefficient = 0.0, stiffness_coefficient = 0.0) -> OperatorForm` | Create proportional damping ``C = alpha M + beta K``. |
| function | `robin_operator(target, coefficient, *, measure = None, location = None) -> OperatorForm` | Create the boundary matrix ``K_R = integral(h trial test)``. |
| function | `robin_source_vector(target, coefficient, reference_value, *, measure = None, location = None) -> OperatorForm` | Create the Robin environment vector ``F_R = integral(h x_ref test)``. |
| function | `scale(operator, factor, *, name: str \| None = None, kind: str \| None = None) -> OperatorForm` | Scale an operator or vector form while preserving its engineering role. |
| function | `source_vector(source, target, *, measure = ufl.dx) -> OperatorForm` | Create a scalar or vector source/load vector for a target unknown. |
| function | `stiffness(field, properties = None, *, law = None, study = None, measure = ufl.dx) -> OperatorForm` | Create the primary stiffness-like operator ``K`` for an unknown field. |
| function | `quadratic_form(operator, field) -> float` | Return the algebraic scalar ``field^T operator field``. |
| function | `xtmx(field, operator) -> float` | Cast3M-style alias for ``field^T operator field``. |
| function | `xtmy(left, operator, right) -> float` | Cast3M-style alias for ``left^T operator right``. |
| function | `elastic_stiffness(displacement, properties, *, study = None, measure = ufl.dx) -> OperatorForm` | Create an elastic stiffness operator ``K`` from a displacement unknown. |
| function | `internal_force_vector(displacement, test_function = None, properties = None, *, study = None, measure = ufl.dx) -> OperatorForm` | Create an elastic internal-force vector contribution. |
| function | `stiffness_operator(displacement, test_function = None, properties = None, *, study = None, measure = ufl.dx) -> OperatorForm` | Create an elastic stiffness/internal virtual-work operator ``K``. |
| function | `thermal_expansion_vector(target, temperature, properties, *, study = None, measure = ufl.dx, name: str = 'F_thermal') -> OperatorForm` | Equivalent nodal load produced by isotropic thermal expansion. |
| class | `FirstOrderSystem` | First-order transient system, ``C x_dot + K x = F``. |
| class | `LinearSystem` | Engineering-level static system, usually ``K x = F``. |
| class | `SecondOrderSystem` | Engineering-level second-order system, ``M a + C v + K u = F``. |
| function | `first_order_system(C, K, F = None, *, name: str = 'Cxdot_plus_Kx_eq_F')` | Create ``C x_dot + K x = F`` for heat/diffusion-like evolution. |
| function | `linear_system(K, F = None, *, name: str = 'Kx_eq_F') -> LinearSystem` | Create a static linear system in engineering notation, ``K x = F``. |
| function | `second_order_system(M, K, C = None, F = None, *, name: str = 'Ma_plus_Cv_plus_Ku_eq_F')` | Create ``M a + C v + K u = F`` with optional damping and force. |

## `agentfem.problems`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `FEMProblem` | Lightweight finite-element problem description. |
| class | `LinearVariationalProblem` | A standard linear variational problem, ``a(u, v) = L(v)``. |
| class | `LinearSystemProblem` | Engineering-level linear system problem, usually ``K x = F``. |
| class | `NonlinearVariationalProblem` | Nonlinear residual problem ``R(u; v) = 0`` solved by PETSc SNES. |
| class | `NonlinearLoadIncrementInfo` | Convergence evidence for one ordinary nonlinear load increment. |
| class | `NonlinearLoadPathInfo` | Accepted and attempted increments for an ordinary nonlinear step. |
| class | `IncrementalNonlinearVariationalProblem` | Ordinary nonlinear equilibrium with automatic load incrementation. |
| class | `AffineNonlinearVariationalProblem` | Nonlinear equilibrium under an exact affine dof reduction. |
| class | `AnalysisStep` | Inspectable analysis step that owns one algebraic solve. |
| class | `ExplicitDynamicsStep` | Inspectable second-order explicit dynamics step. |
| class | `ImplicitDynamicsStep` | Linear Newmark/generalized-alpha structural-dynamics step. |
| class | `FirstOrderTransientStep` | Reusable implicit-Euler step loop for heat/diffusion problems. |
| class | `TransientState` | Current/next fields for a first-order transient unknown. |
| class | `SecondOrderDynamicsState` | Displacement/velocity/acceleration fields for second-order dynamics. |
| class | `LumpedMassOperator` | Diagonal mass operator for explicit dynamics. |
| function | `second_order_state(field_or_space, **kwargs) -> SecondOrderDynamicsState` | Create a second-order dynamics state from a field or function space. |
| function | `linear_system(K, F, *, unknown = None, solution = None, constraints = None, bcs = None, solver_options: LinearSolverOptions \| None = None, name: str = 'Kx_eq_F') -> LinearSystemProblem` | Create a ``K x = F`` problem without exposing variational boilerplate. |
| function | `linear_static(K, F, *, study = None, unknown = None, solution = None, constraints = None, bcs = None, solver_options: LinearSolverOptions \| None = None, result_field_factory = None, name: str = 'linear_static') -> AnalysisStep` | Create a linear static analysis step in ``K x = F`` notation. |
| function | `nonlinear(residual, solution, *, jacobian = None, constraints = None, bcs = None, solver_options: NonlinearSolverOptions \| NewtonSolverOptions \| None = None, name: str = 'nonlinear', petsc_options_prefix: str = 'agentfem_nonlinear_') -> NonlinearVariationalProblem` | Create a general nonlinear residual problem. |
| function | `incremental_nonlinear(residual, solution, *, factor, value_path, update_load = None, acceptance_check = None, jacobian = None, incrementation = None, constraints = None, bcs = None, solver_options: NonlinearSolverOptions \| NewtonSolverOptions \| None = None, output_every: int \| None = 1, progress = True, status_file = None, name: str = 'incremental_nonlinear', petsc_options_prefix: str = 'agentfem_incremental_nonlinear_') -> IncrementalNonlinearVariationalProblem` | Create standard-BC nonlinear equilibrium over a normalized load path. |
| function | `affine_nonlinear(residual, solution, *, jacobian, constraint, load_factors = None, incrementation = None, solver_options: AffineNewtonOptions \| NewtonSolverOptions \| None = None, output_every: int \| None = 1, output_factors = (), progress = True, status_file = None, name: str = 'affine_nonlinear') -> AffineNonlinearVariationalProblem` | Create a nonlinear problem reduced by an affine constraint map. |
| class | `LoadIncrementSnapshot` | A copied solution state at one nonlinear load factor. |
| function | `first_order_transient(*, capacity, stiffness, history, source = None, dt: float, study = None, unknown = None, solution = None, constraints = None, bcs = None, solver_options: LinearSolverOptions \| None = None, name: str = 'first_order_transient_step', method: str = 'implicit_euler') -> AnalysisStep` | Create a first-order transient step. |
| function | `first_order_transient_run(*, capacity, stiffness, history, current, previous, dt: float, steps: int, source = None, study = None, constraints = None, bcs = None, solver_options: LinearSolverOptions \| None = None, update_load = None, save_every: int \| None = None, print_every: int \| None = None, progress = True, status_file = None, checkpoint_policy = None, name: str = 'first_order_transient') -> FirstOrderTransientStep` | Create an executable implicit-Euler time step and loop. |
| function | `explicit_dynamics(*, state, integrator, residual, stiffness = None, dt: float, steps: int, study = None, prescribed = (), constraints = (), update_load = None, save_every: int \| None = None, print_every: int \| None = None, progress = True, status_file = None, checkpoint_policy = None, name: str = 'explicit_dynamics') -> ExplicitDynamicsStep` | Create a second-order explicit dynamics step. |
| function | `implicit_dynamics(*, state, mass, stiffness, force, damping = None, dt: float, steps: int, parameters = None, study = None, constraints = (), bcs = None, solver_options: LinearSolverOptions \| None = None, update_load = None, progress = True, status_file = None, checkpoint_policy = None, save_every: int \| None = None, print_every: int \| None = None, name: str = 'implicit_dynamics') -> ImplicitDynamicsStep` | Create a linear Newmark or generalized-alpha dynamics step. |

## `agentfem.project`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `new_run_id(now: datetime \| None = None) -> str` | Return a sortable, collision-resistant identifier for one execution. |
| class | `ProjectConfig` | Operational metadata for an AgentFEM case directory. |
| function | `discover(start: str \| Path \| None = None) -> ProjectConfig` | Find the nearest ``agentfem.toml`` from ``start`` upward. |
| class | `RunContext` | Filesystem and identity contract shared by scripts, CLIs, GUIs, and agents. |
| function | `current_run(*, project_root: str \| Path \| None = None, project_name: str \| None = None) -> RunContext` | Return the CLI-provided context or create one for direct Python use. |

## `agentfem.provenance`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `seal_manifest(manifest: Mapping[str, object], *, base: str \| Path, producer_version: str) -> dict[str, object]` | Return a deterministic integrity seal for an unsealed manifest. |
| class | `SealVerification` | Outcome of checking one stored provenance seal. |
| function | `verify_manifest(path: str \| Path) -> SealVerification` | Verify a result manifest and every artifact recorded in its seal. |

## `agentfem.platforms`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `PlatformSupport` | One operating-system support decision with explicit limitations. |
| class | `RuntimeReport` | Compact runtime inventory for bug reports and agent inspection. |
| function | `support_for(system: str, *, wsl: bool = False) -> PlatformSupport` | Return the first-release support tier for an operating-system route. |
| function | `current_support() -> PlatformSupport` | Detect the current OS, including Windows Subsystem for Linux. |
| function | `runtime_report() -> RuntimeReport` | Return versions and optional integrations useful in issue reports. |

## `agentfem.procedures`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `SolutionProcedure` | Inspectable, backend-neutral description of a solution algorithm. |
| function | `linear_static() -> SolutionProcedure` | Public AgentFEM object. |
| function | `nonlinear_static(*, stateful: bool = False) -> SolutionProcedure` | Public AgentFEM object. |
| function | `implicit_euler(*, nonlinear: bool = False, stateful: bool = True) -> SolutionProcedure` | Public AgentFEM object. |
| function | `implicit_creep() -> SolutionProcedure` | Quasi-static backward-Euler creep with global Newton equilibrium. |
| function | `newmark() -> SolutionProcedure` | Public AgentFEM object. |
| function | `generalized_alpha() -> SolutionProcedure` | Public AgentFEM object. |
| function | `central_difference() -> SolutionProcedure` | Public AgentFEM object. |
| function | `for_step(*, analysis: str, method: str \| None = None, stateful: bool = False)` | Resolve a procedure without coupling ``Study`` to one solver route. |

## `agentfem.results`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CheckpointRecord` | One restart asset with an explicit portability boundary. |
| class | `FieldResult` | A named live field or an external field artifact. |
| class | `HistoryResult` | Time, load, or iteration history with a fixed value shape. |
| class | `ResultQuantity` | One scalar or fixed-shape quantity of interest. |
| class | `SimulationResult` | Scientific results and artifacts from one simulation. |
| function | `dof_statistics(field) -> dict[str, float \| int]` | Return global finite dof statistics for a DOLFINx-like field. |
| function | `from_solution(solution, *, name: str = 'result', field_name: str \| None = None, unit: str \| None = None, metadata: Mapping[str, object] \| None = None) -> SimulationResult` | Wrap one solved field in a :class:`SimulationResult`. |
| class | `ForceMomentResultant` | Integrated force and moment about an explicit physical point. |
| class | `PathSample` | Values sampled along one straight physical-space path. |
| class | `StaticForceBalance` | Global algebraic force equilibrium for one linear static solid. |
| function | `average(expression, *, measure = ufl.dx, comm = None)` | Return the measure-weighted global average of an expression. |
| function | `boundary_resultant(traction, *, on)` | Integrate a traction/flux expression over a named boundary. |
| function | `field_extrema(field, *, magnitude: bool = False, location: bool = False) -> dict[str, object]` | Return MPI-global field extrema, optionally with physical locations. |
| function | `free_body_resultant(*, boundary_tractions = (), body_forces = (), about) -> ForceMomentResultant` | Integrate boundary and volume forces into one free-body resultant. |
| function | `external_force_resultant(problem)` | Return the MPI-global resultant of a linear problem's assembled RHS. |
| function | `integral(expression, *, measure = ufl.dx, comm = None)` | Return the global integral of a scalar, vector, or tensor expression. |
| function | `l2_norm(expression, *, measure = ufl.dx, comm = None) -> float` | Return ``sqrt(integral(inner(value, value)))`` globally. |
| function | `probe(field, *, at, padding: float = 1e-10)` | Return one scalar, vector, or tensor field value at a physical point. |
| function | `quadrature_extrema(expression, domain, *, degree: int = 4) -> tuple[float, float]` | Return global min/max sampled at Basix quadrature points. |
| function | `reaction_resultant(problem, *, on = None, component: int \| None = None, name: str = 'RF')` | Return an MPI-global strong-constraint reaction resultant. |
| function | `region_average(expression, *, on)` | Return a measure-weighted average over a named mesh region. |
| function | `region_integral(expression, *, on)` | Integrate a scalar, vector, or tensor over a named mesh region. |
| function | `region_measure(*, on) -> float` | Return the global length, area, or volume of a named region. |
| function | `sample_path(field, *, start, end, count: int = 101, padding: float = 1e-10, missing: str = 'raise') -> PathSample` | Sample a field along the straight segment from ``start`` to ``end``. |
| function | `sample_points(field, points, *, padding: float = 1e-10, missing: str = 'raise') -> np.ndarray` | Evaluate a finite-element field at common physical points under MPI. |
| function | `section_resultant(stress, *, on, normal = None, about = None) -> ForceMomentResultant` | Integrate section force and moment from a Cauchy/nominal stress field. |
| function | `static_force_balance(problem) -> StaticForceBalance` | Evaluate ``R + F = 0`` for a converged linear static solid. |
| function | `project(expression, *, domain = None, family: str = 'DG', degree: int = 0, name: str = 'ProjectedField')` | Return the global L2 projection of a UFL expression. |
| function | `project_piecewise(terms, *, domain = None, family: str = 'DG', degree: int = 0, name: str = 'ProjectedField')` | Project region-dependent expressions into one finite-element field. |
| function | `small_strain_cell_fields(displacement, properties, *, study = None, variables = ('S', 'E', 'MISES', 'SENER'), degree: int = 0) -> tuple[object, ...]` | Create standard projected fields for linear small-strain elasticity. |
| function | `small_strain_partition_fields(displacement, assignments, *, study = None, variables = ('S', 'E', 'MISES', 'SENER'), degree: int = 0) -> tuple[object, ...]` | Create standard fields for a complete regional material partition. |
| function | `add_execution_trace(result, events: Iterable[object]) -> tuple[dict[str, object], ...]` | Attach one complete execution trace and its standard histories. |
| function | `execution_records(events: Iterable[object]) -> tuple[dict[str, object], ...]` | Normalize solver events without depending on a particular procedure. |
| class | `HomogenizedFrame` | Macroscopic response reconstructed from one periodic-cell state. |
| function | `finite_strain_diagnostics(displacement, *, constraint = None, quadrature_degree: int = 4) -> dict[str, object]` | Evaluate reusable physical checks for a finite-deformation solution. |
| function | `finite_strain_cell_fields(displacement, properties, *, variables = ('F', 'E', 'GREEN', 'P', 'S', 'MISES', 'J', 'SENER', 'EVOL')) -> tuple[object, ...]` | Create requested standard P0 finite-strain cell fields. |
| function | `homogenize_periodic_cell(displacement, properties, *, macro_deformation_gradient, cell_reference_volume: float, load_factor: float) -> HomogenizedFrame` | Return volume-normalized macroscopic finite-strain response. |
| function | `homogenize_periodic_path(snapshots, properties, *, constraint) -> tuple[HomogenizedFrame, ...]` | Homogenize every saved state of an affine periodic-cell analysis. |
| function | `write_homogenized_csv(path: str \| Path, frames) -> Path` | Write flattened macro tensors in a human-readable table. |
| function | `write_homogenized_history(path: str \| Path, frames) -> Path` | Write an exact, compact NumPy history for plotting and ML reuse. |
| class | `FieldVariable` | Stable public meaning of one result variable. |
| function | `field_variable(name: str, *, finite_strain: bool = False) -> FieldVariable` | Resolve a standard variable, including the context-dependent ``E`` alias. |
| function | `preselected_fields(*, physics: str, finite_strain: bool = False) -> tuple[str, ...]` | Return the engineering-default field set for one physics context. |
| function | `resolve_field_variables(names, *, finite_strain: bool = False) -> tuple[FieldVariable, ...]` | Resolve aliases, preserve request order, and remove duplicates. |
| class | `FieldOutput` | What fields to save, how often, and in which configuration. |
| class | `FieldOutputArtifacts` | Files and final live fields produced by one output plan. |
| function | `field_output(*variables, every: int \| str \| None = None, intervals: int \| None = None, configuration: str = 'deformed', deformation_scale: float = 1.0, backend: str = 'xdmf') -> FieldOutput` | Create a concise, inspectable field-output request. |
| function | `read_unified_xdmf_series(xdmf_path) -> tuple[object, ...]` | Read AgentFEM's compact XDMF/HDF5 frames as PyVista grids. |
| function | `write_deformed_vtk_series(pvd_path, snapshots, cell_fields, *, deformation_scale: float = 1.0) -> tuple[Path, tuple[Path, ...]]` | Write one deformed VTU grid per frame and a ParaView PVD collection. |
| function | `write_unified_xdmf_series(xdmf_path, snapshots, cell_fields, *, deformation_scale: float = 1.0, store_reference_geometry: bool = True, compression: int = 4) -> Path` | Write one temporal XDMF and one compressed HDF5 heavy-data file. |
| class | `FiniteStrainDiagnosticRequest` | Record physical admissibility and constraint checks. |
| class | `HistoryRequest` | Evaluate one scientific quantity on every accepted output frame. |
| class | `OutputPlan` | One declarative output contract for a completed finite-strain step. |
| class | `PeriodicCellHistoryRequest` | Record complete tensor histories for a finite-strain periodic cell. |
| class | `ProbeHistoryRequest` | Record a field value at one physical point on every accepted frame. |
| class | `PresentationOutput` | Optional serial rendering from the scientific XDMF/HDF5 series. |
| class | `SolverHistoryRequest` | Record accepted-increment convergence history. |
| class | `SourceNodeHistoryRequest` | Record U and current coordinates using source-mesh node labels. |
| function | `finite_strain_checks(*, constraint = None, quadrature_degree: int = 4) -> FiniteStrainDiagnosticRequest` | Public AgentFEM object. |
| function | `history(name: str, evaluate, *, coordinate = None, unit: str \| None = None, abscissa_name: str \| None = None, abscissa_unit: str \| None = None, description: str = '') -> HistoryRequest` | Create a quantity history evaluated on accepted snapshots. |
| function | `output_plan(directory, *, field: FieldOutput \| None = None, requests = (), presentation: PresentationOutput \| None = None, basename: str = 'results') -> OutputPlan` | Create a complete finite-strain output plan. |
| function | `periodic_cell_history(constraint, *, basename: str = 'homogenized_history') -> PeriodicCellHistoryRequest` | Public AgentFEM object. |
| function | `probe_history(name: str, *, at, field = None, component: int \| None = None, unit: str \| None = None, description: str = '') -> ProbeHistoryRequest` | Create a point-probe history for an accepted field sequence. |
| function | `presentation(*, comparison: bool = True, animation: str \| None = 'gif', scalar: str = 'UMAG', fps: int = 2) -> PresentationOutput` | Public AgentFEM object. |
| function | `solver_history() -> SolverHistoryRequest` | Public AgentFEM object. |
| function | `source_node_history(nodes, **points: int) -> SourceNodeHistoryRequest` | Public AgentFEM object. |
| function | `render_deformation_animation(undeformed_path, snapshots, nodes, output_path, *, fps: int = 2) -> Path` | Render scale-one deformation history as GIF or MP4. |
| function | `render_deformation_comparison(undeformed_path, deformed_path, output_path, *, scalar: str = 'DisplacementMagnitude') -> Path` | Render side-by-side undeformed/deformed surfaces with PyVista. |
| function | `render_unified_xdmf_animation(xdmf_path, output_path, *, scalar: str = 'UMAG', fps: int = 2) -> Path` | Render a GIF or MP4 from AgentFEM's single XDMF/HDF5 series. |
| function | `render_unified_xdmf_comparison(xdmf_path, output_path, *, scalar: str = 'UMAG') -> Path` | Render the first and final grids from a unified XDMF series. |
| function | `render_vtk_series_animation(frame_paths, output_path, *, scalar: str = 'UMAG', fps: int = 2) -> Path` | Render a GIF directly from a combined-field deformed VTU series. |

## `agentfem.solvers`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `LinearSolverOptions` | PETSc KSP options for a linear solve. |
| function | `direct_solver(*, package: str \| None = None) -> LinearSolverOptions` | Create a direct linear-solver policy without PETSc option names. |
| class | `NonlinearSolverOptions` | PETSc SNES/KSP policy for nonlinear finite-element solves. |
| class | `NewtonSolverOptions` | Backend-neutral Newton policy for nonlinear equilibrium. |
| function | `newton(*, relative_tolerance: float = 1e-08, absolute_tolerance: float = 1e-09, maximum_iterations: int = 30, line_search: str \| None = 'backtracking', linear_solver: LinearSolverOptions \| None = None, error_if_not_converged: bool = True) -> NewtonSolverOptions` | Create one Newton policy for ordinary and affine-constrained steps. |
| class | `NonlinearSolveInfo` | Convergence evidence returned by a PETSc SNES solve. |
| class | `AffineNewtonOptions` | Newton policy for an affine-reduced nonlinear equilibrium path. |
| class | `AffineLoadIncrementInfo` | Convergence evidence for one macroscopic load increment. |
| class | `AffineLoadPathInfo` | Convergence evidence for an incrementally applied affine constraint. |
| class | `SolveEvent` | One structured event emitted by an analysis procedure. |
| function | `create_ksp(comm, options: LinearSolverOptions \| None = None)` | Create and configure a PETSc KSP object. |
| class | `LinearSolveInfo` | PETSc KSP convergence evidence for one linear system solve. |
| function | `solve_matrix_system(A, b, x, options: LinearSolverOptions \| None = None, *, raise_on_failure: bool \| None = None) -> LinearSolveInfo` | Solve ``A x = b`` and return explicit PETSc convergence evidence. |
| function | `solve_linear_problem(bilinear_form, linear_form, solution, *, bcs = None, options: LinearSolverOptions \| None = None, return_info: bool = False)` | Assemble and solve a standard linear variational problem. |
| function | `solve_nonlinear_problem(residual_form, solution, *, bcs = None, jacobian_form = None, options: NonlinearSolverOptions \| NewtonSolverOptions \| None = None, petsc_options_prefix: str = 'agentfem_nonlinear_') -> tuple[object, NonlinearSolveInfo]` | Solve ``R(u; v) = 0`` with the current DOLFINx PETSc/SNES interface. |
| function | `solve_affine_nonlinear_path(residual_form, jacobian_form, solution, constraint, *, load_factors = None, incrementation = None, output_factors = (), options: AffineNewtonOptions \| NewtonSolverOptions \| None = None, on_increment = None, reporter = None, step_name: str = 'affine_nonlinear', step_number: int = 1) -> tuple[object, AffineLoadPathInfo]` | Solve a nonlinear path under ``u = T q + u_bar`` constraints. |

## `agentfem.steps`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `AutomaticIncrementation` | Adaptive load/time incrementation for one analysis step. |
| class | `FixedIncrementation` | A prescribed monotone load-factor path for one analysis step. |
| function | `automatic(*, initial: float = 0.1, minimum: float = 1e-05, maximum: float = 0.25, max_increments: int = 100, max_cutbacks: int = 5, cutback_factor: float = 0.25, growth_factor: float = 1.5, fast_iterations: int = 4, slow_iterations: int = 10, maximum_inelastic_increment: float \| None = None) -> AutomaticIncrementation` | Create inspectable Abaqus-style automatic incrementation. |
| function | `fixed(increments: int) -> FixedIncrementation` | Divide the normalized step interval into exactly ``increments`` parts. |
| function | `at(*load_factors: float) -> FixedIncrementation` | Create a prescribed, nonuniform load-factor path. |
| function | `normalize(value = None, *, increments: int \| None = None, load_factors = None)` | Normalize public and compatibility incrementation inputs. |
| class | `EngineeringStep` | Named inherited activation state, separate from solver controls. |
| function | `engineering_step(name: str, *, previous: EngineeringStep \| None = None, inherit_model_loads: bool = False, inherit_model_constraints: bool = True)` | Public AgentFEM object. |

## `agentfem.time`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `central_difference_predict_displacement(u_next, u, velocity, acceleration, dt: float) -> None` | Predict displacement with the explicit central-difference/Newmark formula. |
| function | `acceleration_from_residual(acceleration, residual, inv_mass: np.ndarray) -> None` | Set acceleration from residual and inverse lumped mass. |
| function | `central_difference_update_midstep_velocity(velocity_mid, velocity, acceleration, dt: float) -> None` | Update the central-difference mid-step velocity. |
| function | `central_difference_correct_velocity(velocity_next, velocity, acceleration, acceleration_next, dt: float) -> None` | Correct velocity with the explicit central-difference/Newmark formula. |
| function | `central_difference_update_velocity(velocity_next, velocity_mid, acceleration_next, dt: float) -> None` | Update whole-step velocity from mid-step velocity and new acceleration. |
| class | `ProgressPrinter` | Rank-zero progress printer controlled by a fixed step interval. |
| class | `TimeStep` | Metadata for one transient-solve step. |
| class | `TimeStepper` | Iterate over transient-solve step metadata. |
| function | `format_duration(seconds: float) -> str` | Format elapsed seconds as ``HH:MM:SS``. |
| class | `GeneralizedAlphaParameters` | Parameters for Newmark/generalized-alpha time integration. |
| function | `generalized_alpha(*, spectral_radius: float = 0.8)` | Second-order generalized-alpha parameters from ``rho_infinity``. |
| function | `newmark(*, beta: float = 0.25, gamma: float = 0.5)` | Average-acceleration Newmark by default. |

## `agentfem.units`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `UnitSystem` | Named consistent base-unit contract attached to a model. |
| function | `consistent(*, length, mass, time, temperature = 'K', name = 'consistent_units')` | Declare the base units used consistently by all model inputs. |
| function | `si(*, temperature = 'K') -> UnitSystem` | Return the SI ``m-kg-s`` engineering contract. |
| function | `n_mm_mpa(*, temperature = 'K') -> UnitSystem` | Return the common ``mm-N-s-MPa`` consistent system. |

## `agentfem.upgrades`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `UpgradeFinding` | One stable, addressable compatibility or migration finding. |
| class | `UpgradeReport` | Dry-run migration plan for one installed-use project. |
| function | `inspect_project(project: ProjectConfig) -> UpgradeReport` | Return a dry-run upgrade report without executing or changing the case. |
| function | `apply_safe_metadata(project: ProjectConfig) -> tuple[Path, ...]` | Apply only deterministic project-metadata migrations, atomically. |

## `agentfem.io`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `ensure_output_dir(path: Path, comm: MPI.Comm) -> None` | Create an output directory once, then synchronize all ranks. |
| class | `CSVLogger` | Rank-zero CSV writer for time histories and scalar diagnostics. |
| class | `XDMFTimeSeries(path: Path, domain, mode: str = 'w') -> None` | Small context manager for writing a mesh and time-dependent fields. |
| class | `ResultWriter(path: Path, domain, fields = (), mode: str = 'w') -> None` | Named result writer for one mesh and a stable field list. |
| function | `interpolate_for_xdmf(field, *, degree: int = 1, name: str \| None = None)` | Interpolate a field to an XDMF-friendly Lagrange output space. |

## `agentfem.diagnostics`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `comm_of(obj = None, default = MPI.COMM_WORLD)` | Return the MPI communicator associated with an object when possible. |
| function | `is_root(obj = None, *, root: int = 0) -> bool` | Return whether the current MPI rank is the selected reporting rank. |
| function | `print_on_root(obj, *args, root: int = 0, flush: bool = True, **kwargs) -> None` | Print a message only on the selected MPI root rank. |
| class | `StandardRunReporter` | Immediate rank-zero progress for long-running analysis steps. |
| class | `SolveEventRecorder` | In-memory structured execution trace shared by every procedure. |
| class | `ReporterGroup` | Fan one solver event out to several independent consumers. |
| function | `compose_reporters(*reporters) -> object \| None` | Compose progress, persistence, and agent observers without coupling. |
| function | `kinetic_energy(mass_lumped: np.ndarray, velocity: fem.Function) -> float` | Global kinetic energy from a lumped mass vector and velocity field. |
| class | `MechanicalEnergy` | Kinetic, recoverable strain, and total mechanical energy. |
| function | `mechanical_energy(*, mass, stiffness, displacement, velocity) -> MechanicalEnergy` | Evaluate ``1/2 v^T M v`` and ``1/2 u^T K u`` from visible operators. |
| class | `LinearStaticEnergy` | Energy closure for a proportional linear-static load path. |
| function | `linear_static_energy(*, stiffness, force, displacement) -> LinearStaticEnergy` | Evaluate energy for loads ramped proportionally from zero to ``force``. |
| class | `MechanicalEnergyMonitor` | Cache visible M/K operators and sample mechanical energy in time. |
| class | `ThermalBalanceMonitor` | Sample discrete heat content, applied rate, outflow, and closure. |
| class | `ThermalContentMonitor` | Backwards-compatible sensible-heat monitor without balance terms. |
| function | `max_abs(function: fem.Function) -> float` | Global max absolute value of a finite-element field. |
| function | `max_magnitude(function) -> float` | Global maximum magnitude of a scalar or vector finite-element field. |
| class | `FieldStats` | Distributed scalar statistics for a finite-element field. |
| class | `ScalarDiagnostic` | Named scalar diagnostic evaluated on demand. |
| class | `DiagnosticSet` | Ordered collection of scalar diagnostics. |
| function | `magnitude_stats(function, *, on = None, name: str \| None = None) -> FieldStats` | Return distributed magnitude statistics for a scalar or vector field. |
| function | `field_stats(function, *, on = None, name: str \| None = None) -> FieldStats` | Alias for ``magnitude_stats`` for application-level diagnostics. |

## `agentfem.ir`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `describe(item)` | Prefer semantic records over display-only summaries. |
| function | `describe_many(items: Iterable[object]) -> tuple[object, ...]` | Describe a collection without retaining backend memory addresses. |
| function | `model_document(model, *, agentfem_version: str, backend: Mapping[str, object] \| None = None, include_validation: bool = True, metadata: Mapping[str, object] \| None = None) -> IRDocument` | Build an experimental AF-IR model document. |
| class | `IRDocument` | Canonical envelope for an AF-IR artifact. |
| class | `IRSerializationError` | Raised when a value cannot be represented without hiding its meaning. |
| function | `to_json_safe(value, *, path: str = '$')` | Convert scientific summaries to deterministic JSON-safe values. |
| function | `write_document(document: IRDocument \| Mapping[str, object], path: str \| Path, *, indent: int = 2) -> Path` | Write one deterministic AF-IR JSON document and return its path. |
| function | `describe_value(value)` | Return a JSON-safe coefficient value or an explicit opaque marker. |

## `agentfem.campaigns`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Campaign(*, name: str, parameter_space: ParameterSpace, outputs: tuple[Quantity, ...], evaluate: Callable[[object], Mapping[str, object] \| CaseOutcome \| SimulationResult], build: Callable[[Mapping[str, object]], object] \| None = None, metadata: Mapping[str, object] \| None = None, execution: ExecutionPolicy \| None = None) -> None` | Build and evaluate a collection of immutable scientific cases. |
| class | `CampaignCase` | One immutable case in a campaign plan. |
| class | `CampaignPlan` | Immutable cases and their design-of-experiment evidence. |
| class | `CampaignReport` | Case-level evidence and the successful scientific dataset. |
| class | `CaseOutcome` | Successful case outputs plus links to scientific evidence. |
| class | `CaseRunRecord` | Execution evidence for one attempted case. |
| class | `ExecutionPolicy` | Declared execution behavior for the current campaign runner. |
| function | `case_id(campaign_name: str, parameters: Mapping[str, object], *, schema_version: str = CAMPAIGN_SCHEMA_VERSION) -> str` | Return a deterministic scientific case identity. |
| function | `create(**kwargs) -> Campaign` | Create a :class:`Campaign` using the public functional spelling. |
| class | `CampaignSpecification` | Validated declarative part of a campaign. |
| function | `load_specification(path: str \| Path) -> CampaignSpecification` | Load a safe JSON campaign specification. |
| function | `specification_from_dict(record: Mapping[str, object]) -> CampaignSpecification` | Validate a dictionary and construct a campaign specification. |
| class | `ChoiceParameter` | Finite categorical or policy parameter. |
| class | `IntegerParameter` | Bounded integer parameter. |
| class | `ParameterSpace` | Ordered scientific input schema for a campaign. |
| class | `RealParameter` | Bounded continuous parameter with optional units and log scaling. |
| class | `SamplingPlan` | Immutable, validated collection of parameter samples. |
| function | `explicit(space: ParameterSpace, samples: Iterable[Mapping[str, object]], *, metadata: Mapping[str, object] \| None = None) -> SamplingPlan` | Create a plan from caller-supplied samples. |
| function | `full_factorial(space: ParameterSpace, levels: int \| Mapping[str, int] = 3) -> SamplingPlan` | Create a full-factorial design in normalized coordinates. |
| function | `latin_hypercube(space: ParameterSpace, count: int, *, seed: int = 0) -> SamplingPlan` | Draw a reproducible Latin-hypercube design. |
| function | `random(space: ParameterSpace, count: int, *, seed: int = 0) -> SamplingPlan` | Draw reproducible independent uniform samples in normalized space. |

## `agentfem.checkpointing`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CheckpointPolicy` | Automatic accepted-increment checkpoint cadence for transient steps. |
| function | `every(increments: int, *, directory = 'checkpoints', final: bool = True, prefix: str \| None = None) -> CheckpointPolicy` | Create an automatic checkpoint policy for accepted time increments. |
| function | `save_transient_checkpoint(path, *, step_kind: str, step_name: str, procedure, dt: float, total_steps: int, completed_steps: int, state: dict[str, object], accepted_times = (), execution_events = (), history_records = ())` | Write one partition-bound transient restart and return its manifest. |
| function | `load_transient_checkpoint(path, *, step_kind: str, step_name: str, procedure, dt: float, total_steps: int, state: dict[str, object]) -> dict[str, object]` | Restore a transient state after validating its scientific identity. |
| function | `function_partition_identity(function) -> dict[str, object]` | Return a JSON-safe identity for one field on one mesh partition. |
| function | `atomic_savez(path, **arrays) -> Path` | Atomically publish one NumPy archive in its destination directory. |
| function | `atomic_write_text(path, content: str) -> Path` | Atomically publish UTF-8 text in its destination directory. |

## `agentfem.datasets`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `DatasetSplit` | Reproducible train/validation partition. |
| class | `ScientificDataset` | A numeric dataset whose columns retain scientific meaning. |
| class | `Quantity` | One scalar, curve, vector, or sampled-field output contract. |
| class | `Sample` | One successful simulation sample and its scientific lineage. |
| function | `decode_quantities(quantities: tuple[Quantity, ...], row) -> dict[str, object]` | Restore one flattened numeric row to declared named quantities. |
| class | `FEMFieldSample` | One FEM field representation with coordinates and scientific encoding. |
| class | `TorchDatasetBundle` | PyTorch dataset plus the schema needed to interpret its columns. |
| function | `fem_field_sample(function, encoding) -> FEMFieldSample` | Export owned nodal coefficients for external neural/PINN tooling. |
| function | `fem_observation_sample(function, grid, *, name: str \| None = None, unit: str \| None = None, role: str = 'output', components = (), outside: str = 'raise', fill_value: float = 0.0) -> FEMFieldSample` | Sample a FEM field on a reusable structured observation grid. |
| function | `to_torch(dataset: ScientificDataset, *, normalized_inputs: bool = True, dtype: str = 'float32', device: str = 'cpu') -> TorchDatasetBundle` | Expose a validated campaign dataset as a PyTorch ``TensorDataset``. |

## `agentfem.surrogates`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Prediction` | One named prediction with source and trust diagnostics. |
| class | `QuantityMetrics` | Error evidence for one declared output quantity. |
| class | `SurrogateValidationReport` | Independent validation metrics and optional acceptance decision. |
| function | `validate_predictions(*, model_kind: str, dataset: ScientificDataset, predictions: np.ndarray, thresholds: Mapping[str, float] \| None = None) -> SurrogateValidationReport` | Compare flattened predictions with a dataset's declared quantities. |
| class | `BoxApplicabilityDomain` | Axis-aligned envelope in normalized scientific parameter space. |
| class | `GuardedSurrogate` | Use a surrogate only inside its declared applicability domain. |
| class | `OutOfDomainError` | Raised when an unguarded surrogate is asked to extrapolate. |
| class | `PODRidgeSurrogate` | Proper-orthogonal-decomposition outputs plus ridge latent dynamics. |
| class | `RidgeSurrogate` | Multi-output ridge regression baseline. |
| class | `TrainedPODRidge` | Fitted POD-ridge field/curve surrogate. |
| class | `TrainedRidge` | Fitted ridge surrogate with named prediction and validation methods. |
| class | `FieldEncoding` | How a physical field becomes a machine-learning tensor. |
| class | `NeuralOperatorSpec` | Function-to-function learning contract for an external trainer. |
| class | `ObservationGrid` | Mesh-independent Cartesian coordinates for field learning and sensing. |
| class | `PhysicsCondition` | Boundary, initial, interface, or observation condition in a loss. |
| class | `PhysicsResidual` | One explicit differentiable residual used in a physics loss. |
| class | `PINNSpec` | Physics-informed training contract for selected explicit residuals. |
| function | `regular_grid(*, bounds, shape, axis_names = None, coordinate_system: str = 'cartesian', order: str = 'C') -> ObservationGrid` | Create an evenly spaced observation grid from physical bounds. |
| class | `TorchMLPSurrogate` | Configurable dense-network baseline for parameter-to-QoI learning. |
| class | `TrainedTorchMLP` | In-memory trained PyTorch adapter. |
| class | `PINNTrainingRecord` | In-memory training evidence without serializing a PyTorch pickle. |
| class | `TorchPINNAdapter` | Bind explicit residual/condition callables to a :class:`PINNSpec`. |
| class | `SurrogateTrainingRun` | A trained model together with its independent validation evidence. |
| function | `train(dataset: ScientificDataset, *, estimator = None, validation_fraction: float = 0.2, seed: int = 0, thresholds = None) -> SurrogateTrainingRun` | Split, fit, and independently validate one surrogate estimator. |

## `agentfem.validation`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ValidationIssue` | One addressable model, numerical, or execution issue. |
| class | `ValidationReport` | Immutable collection of structured validation issues. |
| class | `ModelValidationError(report: ValidationReport)` | Raised when a structured model validation report contains errors. |
| function | `issue(code: str, path: str, message: str, *, severity: Severity = 'error', hint: str \| None = None, **context) -> ValidationIssue` | Concise constructor used by model validators and backend adapters. |

## `agentfem.verification`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `trust_rank(level: str) -> int` | Return the ordered rank of one public trust level. |
| class | `VerificationClaim` | One explicit, machine-readable scientific acceptance claim. |
| class | `VerificationReport` | Trust decision derived from execution state and scientific claims. |
| class | `QualityPolicy` | Low-ceremony acceptance policy for one result or dataset boundary. |
| class | `ConvergenceSample` | One observable evaluated at a declared discretization size. |
| class | `ConvergenceStudy` | A coarse-to-fine mesh or time-step convergence sequence. |
| function | `report(*claims: VerificationClaim, computed: bool = True, converged: bool = True, scope: str = 'simulation') -> VerificationReport` | Concise public constructor for a verification report. |
| function | `quality_policy(value: str \| QualityPolicy) -> QualityPolicy` | Return one named public quality policy. |
| function | `assess(result, quality: str \| QualityPolicy = 'engineering', *, claims: Iterable[VerificationClaim] = (), converged: bool \| None = None, required_quantities: Iterable[str] = (), required_histories: Iterable[str] = (), required_artifacts: Iterable[str] = (), attach: bool = True) -> VerificationReport` | Apply a quality preset and inexpensive deterministic result checks. |
| function | `convergence_study(name: str, observable: str, samples: Iterable[ConvergenceSample], *, discretization: str = 'mesh') -> ConvergenceStudy` | Public AgentFEM object. |
