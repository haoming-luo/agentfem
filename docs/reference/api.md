---
title: Python API
description: Automatically generated public AgentFEM Python API index.
hide:
  - toc
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
| function | `inspect_abaqus_input(path: str \| Path) -> abaqus.AbaqusMigrationReport` | Inventory Abaqus engineering semantics before conversion or solving. |
| function | `inspect_abaqus_source_graph(path: str \| Path) -> abaqus.AbaqusSourceGraph` | Resolve and fingerprint nested Abaqus input sources without flattening. |
| function | `plan_abaqus_migration(path: str \| Path) -> 'AbaqusMigrationPlan'` | Build a scope-aware Abaqus migration plan without solving. |
| function | `create_abaqus_migration_project(source: str \| Path, destination: str \| Path, *, name: str \| None = None, created_with: str = 'unknown', user_material_sources: dict[str, str \| Path] \| None = None) -> dict[str, object]` | Create a fail-closed AgentFEM project from inspected Abaqus sources. |
| function | `assess_abaqus_native_lowering(path: str \| Path)` | Assess whether an Abaqus source fits the reviewed native subset. |
| function | `lower_abaqus_migration_project(project: str \| Path, *, reviewed_by: str, unit_system: str, activate: bool = False, force: bool = False) -> dict[str, object]` | Emit an explicitly reviewed native draft from a migration project. |
| function | `supported_abaqus_element_types(*, family: str \| None = None) -> tuple[str, ...]` | Return Abaqus declarations with explicit AgentFEM import semantics. |
| function | `split_gmsh_physical_interface(*args, **kwargs)` | Lower named Gmsh physical cell/surface groups to a split interface. |
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
| function | `from_geometry_spec(specification: Mapping[str, object], *, resolution: int = 32, comm: MPI.Comm = MPI.COMM_WORLD)` | Create an :class:`agentfem.mesh.FEMMesh` from a public geometry spec. |
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
| function | `model_api(level: str = 'core') -> tuple[str, ...]` | Return the recommended Model vocabulary at one discovery level. |
| function | `model_api_contract(level: str = 'all') -> tuple[dict[str, object], ...]` | Return machine-readable lifecycle metadata for Model methods. |
| class | `Model` | Finite-element model registry for humans and agents. |
| function | `create(*, study, mesh = None, name: str = 'model', units = None) -> Model` | Create a lightweight model registry. |

## `agentfem.fields`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Field` | Tensor-like finite-element field with immediate-value algebra. |
| class | `UnknownField` | Finite-element unknown bundle for application-level workflows. |
| class | `DisplacementPressureUnknown` | Mixed displacement/pressure unknown for hybrid solid mechanics. |
| class | `VelocityPressureUnknown` | Taylor--Hood velocity/pressure unknown for incompressible flow. |
| function | `scalar_unknown(domain, *, name: str = 'Unknown', degree: int = 1, value = 0.0) -> UnknownField` | Create a scalar finite-element unknown. |
| function | `vector_unknown(domain, *, name: str = 'Unknown', degree: int = 1, dim: int \| None = None, value = 0.0) -> UnknownField` | Create a vector finite-element unknown. |
| function | `displacement(domain, *, degree: int = 1, dim: int \| None = None, value = 0.0) -> UnknownField` | Create a displacement unknown for mechanics workflows. |
| function | `displacement_pressure(domain, *, displacement_degree: int = 2, pressure_degree: int = 0, name: str = 'DisplacementPressure') -> DisplacementPressureUnknown` | Create a mixed displacement/pressure unknown. |
| function | `velocity_pressure(domain, *, velocity_degree: int = 2, pressure_degree: int = 1, name: str = 'VelocityPressure') -> VelocityPressureUnknown` | Create a Taylor--Hood incompressible-flow unknown. |
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
| class | `MaterialAssetError` | A project material asset could not be loaded unambiguously. |
| function | `load(source: str \| Path, *, model: str \| None = None, symbol: str \| None = None, role: str = 'mechanical') -> MaterialDefinition` | Load a packaged card by name or an explicitly selected Python asset. |
| function | `load_python(path: str \| Path, *, symbol: str \| None = None, role: str = 'mechanical') -> MaterialDefinition` | Load one trusted project-owned Python material with source provenance. |
| class | `MaterialBehavior` | One named behavior carried by a physical material definition. |
| class | `MaterialCompatibility` | Pre-solve explanation of one material/Study pairing. |
| class | `MaterialDefinition` | Physical material identity plus independently reusable behaviors. |
| function | `define(name: str, behavior = None, *, mechanical = None, thermal = None, behaviors: Mapping[str, object] \| None = None, source: str = 'user_defined', reference_only: bool = False, metadata: Mapping[str, object] \| None = None) -> MaterialDefinition` | Define a named material without coupling it to one Study. |
| class | `MaterialRecord` | Material-library record before conversion to a constitutive law. |
| function | `list_material_models(name: str) -> tuple[str, ...]` | List model names available for one material. |
| function | `list_materials(*, model: str \| None = None) -> tuple[str, ...]` | List available material names, optionally filtered by model. |
| function | `load_material(name: str, model: str \| None = None)` | Load one material model and return a constitutive material object. |
| function | `load_definition(name: str, model: str \| None = None)` | Load a packaged reference card as a named material definition. |
| function | `material_record(name: str) -> MaterialRecord` | Return a validated material record without constructing a model object. |
| function | `register_material(name: str, data: dict, *, overwrite: bool = False) -> None` | Register or override a material record in memory. |
| class | `ElasticAnisotropic2DProperties` | 2D linear-elastic properties using engineering-strain Voigt notation. |
| class | `ElasticIsotropicProperties` | Isotropic linear-elastic material properties. |
| class | `ThermoElasticIsotropicProperties` | Isotropic thermoelastic and heat-conduction properties. |
| class | `TemperatureDependentThermoElasticProperties` | Isotropic thermoelastic properties containing constants or tables. |
| class | `TemperaturePropertyTable` | One material property tabulated against absolute temperature. |
| function | `temperature_property(temperatures, values, **kwargs) -> TemperaturePropertyTable` | Create an inspectable temperature-dependent material property. |
| function | `validate_material_record(name: str, record: dict) -> None` | Validate one material-centered library record. |

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
| class | `ImplicitCreepBatchUpdate` | Vectorized backward-Euler updates for one homogeneous material region. |
| class | `ImplicitCreepState` | Committed small-strain creep state at one integration point. |
| class | `ImplicitCreepUpdate` | Backward-Euler material-point update and consistent tangent. |
| class | `IsotropicPowerLawCreepMaterial` | Isotropic elasticity with an implicit Mises power-law creep branch. |
| class | `KachanovRabotnovCreep` | Classical scalar Kachanov--Rabotnov creep-damage coupling. |
| class | `ModifiedThetaProjection` | Three-parameter modified-theta representation of a creep curve. |
| class | `PowerLawCreep` | Mises time-hardening creep law. |
| class | `SinhCreep` | Stress-sensitive hyperbolic-sine Mises creep law. |
| function | `integrate_stress_history(law: PowerLawCreep, times, interval_stresses) -> CreepHistory` | Integrate a piecewise-constant scalar or tensor stress history. |
| function | `isotropic_power_law(*, young: float \| None = None, poisson: float \| None = None, density: float \| None = None, elastic: ElasticIsotropicProperties \| ThermoElasticIsotropicProperties \| TemperatureDependentThermoElasticProperties \| None = None, coefficient: float, stress_exponent: float, time_exponent: float = 0.0, reference_stress: float = 1.0, reference_time: float = 1.0, name: str = 'isotropic power-law creep') -> IsotropicPowerLawCreepMaterial` | Create one Abaqus-style material record with elastic and creep data. |
| function | `isotropic_arrhenius_power_law(*, young: float \| None = None, poisson: float \| None = None, density: float \| None = None, elastic: ElasticIsotropicProperties \| ThermoElasticIsotropicProperties \| TemperatureDependentThermoElasticProperties \| None = None, coefficient: float, stress_exponent: float, activation_energy: float, reference_temperature: float, time_exponent: float = 0.0, reference_stress: float = 1.0, reference_time: float = 1.0, gas_constant: float = 8.31446261815324, name: str = 'isotropic Arrhenius power-law creep') -> IsotropicPowerLawCreepMaterial` | Create elasticity plus a globally consumable Arrhenius creep law. |
| function | `anisotropic_stress_2d(displacement, properties: ElasticAnisotropic2DProperties, *, study = None)` | 2D anisotropic stress from engineering-strain Voigt stiffness. |
| function | `anisotropic_elastic_2d(*, stiffness_voigt, density: float, name: str = 'anisotropic elastic 2D') -> ElasticAnisotropic2DProperties` | Create 2D anisotropic linear-elastic properties. |
| function | `estimate_elastic_wave_speeds(material) -> tuple[float, float]` | Return approximate ``(pressure_speed, shear_speed)`` for a material. |
| function | `isotropic_stress(displacement, properties: ElasticIsotropicProperties, *, study = None, temperature = None)` | Small-strain isotropic stress, ``sigma(u)``. |
| function | `isotropic_elastic(*, young: float, density: float, poisson: float, name: str = 'isotropic elastic') -> ElasticIsotropicProperties` | Create isotropic linear-elastic properties. |
| function | `thermal_expansion_stress(temperature, properties, *, study = None, dimension = None)` | Return positive ``C:epsilon_thermal`` for an equivalent thermal load. |
| function | `thermal_strain(temperature, properties, *, dimension: int)` | Return isotropic free thermal strain ``alpha (T-T_ref) I``. |
| function | `thermoelastic(*, young: float, density: float, poisson: float, thermal_expansion: float, conductivity: float, specific_heat: float, reference_temperature: float = 293.15, name: str = 'isotropic thermoelastic') -> ThermoElasticIsotropicProperties` | Create one material record for sequential thermal-stress workflows. |
| function | `temperature_dependent_thermoelastic(*, young, density: float, poisson, thermal_expansion, conductivity, specific_heat, reference_temperature: float = 293.15, name: str = 'temperature-dependent isotropic thermoelastic') -> TemperatureDependentThermoElasticProperties` | Create tabulated properties for sequential thermo-mechanics. |
| function | `thermoelastic_stress(displacement, temperature, properties, *, study = None)` | Small-strain isotropic stress including thermal eigenstrain. |
| function | `orthotropic_plane_stress_2d(*, ex: float, ey: float, nuxy: float, gxy: float, density: float, name: str = 'orthotropic plane-stress elastic 2D') -> ElasticAnisotropic2DProperties` | Create 2D orthotropic plane-stress elastic properties. |
| function | `stress(displacement, properties, *, study = None, temperature = None)` | Dispatch to the matching elastic stress relation. |
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
| class | `FiniteStrainJ2Logarithmic` | Multiplicative finite-strain J2 plasticity with Hencky elasticity. |
| function | `finite_strain_j2_logarithmic(*, young: float, poisson: float, yield_stress: float, hardening_modulus: float = 0.0, tangent_relative_step: float = 2e-06) -> FiniteStrainJ2Logarithmic` | Create the logarithmic finite-strain J2 material provider. |
| class | `FiniteStrainKinematics` | Standard total-Lagrangian kinematics derived from one displacement. |
| class | `MixedNeoHookeanProperties` | Isochoric Neo-Hookean solid with an independent pressure field. |
| class | `MooneyRivlinProperties` | Two-parameter isotropic Mooney-Rivlin finite-strain solid. |
| class | `NeoHookeanProperties` | Compressible Neo-Hookean parameters derived from ``E`` and ``nu``. |
| class | `PlaneStressNeoHookeanProperties` | Compressible Neo-Hookean membrane with locally relaxed thickness. |
| function | `kinematics(displacement) -> FiniteStrainKinematics` | Return the standard finite-strain kinematic measures for ``u``. |
| function | `mooney_rivlin(*, shear_modulus: float, first_invariant_fraction: float, bulk_modulus: float, density: float \| None = None, name: str = 'compressible Mooney-Rivlin') -> MooneyRivlinProperties` | Create a three-dimensional compressible Mooney-Rivlin solid. |
| function | `mooney_rivlin_plane_stress(*, shear_modulus: float, first_invariant_fraction: float, density: float \| None = None, name: str = 'incompressible plane-stress Mooney-Rivlin') -> MooneyRivlinProperties` | Create the exact incompressible sheet reduction of Eq. (17). |
| function | `mixed_condensed_energy_value(deformation_gradient, properties: MixedNeoHookeanProperties) -> float` | Evaluate the pressure-eliminated quadratic-volumetric energy. |
| function | `mixed_neo_hookean(*, young: float \| None = None, poisson: float \| None = None, shear_modulus: float \| None = None, bulk_modulus: float \| None = None, density: float \| None = None, name: str = 'mixed Neo-Hookean') -> MixedNeoHookeanProperties` | Create a quadratic-volumetric mixed Neo-Hookean material. |
| function | `neo_hookean(*, young: float, poisson: float, density: float \| None = None, name: str = 'compressible Neo-Hookean') -> NeoHookeanProperties` | Create a compressible Neo-Hookean material. |
| function | `neo_hookean_plane_stress(*, young: float, poisson: float, density: float \| None = None, name: str = 'plane-stress compressible Neo-Hookean') -> PlaneStressNeoHookeanProperties` | Create a finite-strain plane-stress Neo-Hookean membrane material. |
| function | `plane_stress_first_piola_value(deformation_gradient, properties: PlaneStressNeoHookeanProperties) -> np.ndarray` | Return the condensed numerical in-plane first Piola stress. |
| function | `plane_stress_out_of_plane_first_piola_from_gradient(F, properties: PlaneStressNeoHookeanProperties)` | Return the condensed ``P33`` residual for diagnostics and tests. |
| function | `plane_stress_thickness_stretch_value(deformation_gradient, properties: PlaneStressNeoHookeanProperties, *, tolerance: float = 1e-12, maximum_iterations: int = 30) -> float` | Solve the local ``P33=0`` condition for one numerical 2x2 ``F``. |
| function | `plane_stress_uniaxial_deformation_gradient(axial_stretch: float, properties: PlaneStressNeoHookeanProperties \| MooneyRivlinProperties, *, tolerance: float = 1e-12, maximum_iterations: int = 30) -> np.ndarray` | Return homogeneous uniaxial ``F2`` with traction-free lateral faces. |
| function | `supports_hyperelastic_study(properties, *, dimension: int, assumption) -> bool` | Return whether one material has a formulation for the declared Study. |
| class | `MaterialPointBatchResult` | Responses from one atomic integration-point constitutive update. |
| class | `MaterialQuadratureResponse` | Quadrature stress/tangent fields sharing one typed state transaction. |
| function | `update_material_points(material: UserMaterial \| QuadratureMaterialMap, state: MaterialQuadratureState, *, deformation_gradient_old, deformation_gradient_new, time: float, time_increment: float, properties = (), temperature = None, temperature_increment = None, field_variables = None, commit: bool = False) -> MaterialPointBatchResult` | Update every local quadrature point as one rollback-safe transaction. |
| class | `ChabocheCombinedHardening` | Small-strain J2 plasticity with nonlinear combined hardening. |
| class | `ChabocheState` | History for small-strain combined isotropic/kinematic hardening. |
| class | `J2LinearIsotropicHardening` | Rate-independent von Mises plasticity with linear isotropic hardening. |
| class | `J2PlasticState` | History variables for small-strain isotropic J2 plasticity. |
| class | `J2Update` | Result of one radial-return material-point update. |
| class | `UniaxialPlasticState` | History variables for the exact one-dimensional counterpart. |
| function | `chaboche(*, young: float, poisson: float, yield_stress: float, backstresses: Iterable[tuple[float, float]], isotropic_saturation: float = 0.0, isotropic_rate: float = 0.0, name: str = 'Chaboche combined hardening') -> ChabocheCombinedHardening` | Create a combined-hardening material from ``(C, gamma)`` pairs. |
| function | `update_uniaxial(total_strain: float, material: J2LinearIsotropicHardening, state: UniaxialPlasticState \| None = None) -> tuple[float, UniaxialPlasticState]` | Return stress and state for a one-dimensional bilinear material test. |
| function | `von_mises(stress) -> float` | Return ``sqrt(3/2 s:s)`` for a symmetric Cauchy stress. |
| class | `ChabocheQuadratureState` | Committed/trial integration-point state for combined-hardening J2. |
| class | `CreepQuadratureState` | Committed/trial integration-point state for implicit 3D creep. |
| class | `J2QuadratureState` | Committed/trial integration-point state for 3D small-strain J2. |
| class | `MaterialQuadratureState` | Schema-lowered committed/trial state for one material provider. |
| class | `QuadratureField` | A DOLFINx quadrature function with an explicit NumPy point view. |
| class | `QuadratureMaterialMap` | Cell-region material dispatch shared by stateful solid procedures. |
| class | `QuadratureTransaction` | Shared trial/commit/rollback contract for integration-point state. |
| function | `j2_quadrature_state(domain, material, *, degree: int = 2, scheme: str = 'default')` | Create the quadrature state matching one homogeneous J2 family. |
| function | `load_portable_quadrature_state(path, state, *, material = None) -> None` | Collectively restore committed state under a changed MPI partition. |
| function | `save_portable_quadrature_state(path, state, *, material = None) -> Path` | Collectively save committed state by physical cell and point identity. |
| class | `AbaqusUserMaterialBridge` | Truthful capability description for an intended UMAT/UHYPER adapter. |
| class | `MaterialPointInput` | Solver-neutral finite-strain input for one material-point update. |
| class | `MaterialPointOutput` | Constitutive response returned to a nonlinear finite-element driver. |
| class | `MaterialStateSchema` | Named layout for portable, auditable material internal variables. |
| class | `MaterialStateVariable` | One named entry in a solver-neutral material state vector. |
| class | `MaterialTangentCheck` | Numerical-differentiation evidence for one declared material tangent. |
| class | `MaterialTangentConvention` | Declared stress/kinematic pair represented by a material Jacobian. |
| class | `UserMaterial` | Protocol implemented by native or adapted material-point models. |
| function | `check_material_tangent(material: UserMaterial, point: MaterialPointInput, *, relative_step: float = 1e-07, tolerance: float = 1e-05) -> MaterialTangentCheck` | Compare a declared ``dP/dF`` against fixed-state finite differences. |
| function | `validated_material_update(material: UserMaterial, point: MaterialPointInput) -> MaterialPointOutput` | Run one material update and verify the complete solver contract. |

## `agentfem.constraints`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ConstraintCapabilities` | Solver-facing capability contract for one kinematic constraint. |
| class | `DirichletConstraint` | Strong Dirichlet constraint and its optional mutable value object. |
| class | `TimeDependentDirichlet` | Dirichlet constraint driven by an amplitude. |
| class | `RemoteDisplacementConstraint` | Rigid boundary motion prescribed about a named reference point. |
| class | `PrescribedValuePath` | Update ordinary strong boundary values along a normalized step path. |
| function | `prescribed_value_path(constraints) -> PrescribedValuePath` | Create a normalized load-factor driver from registered constraints. |
| function | `dirichlet_constraints(constraints) -> tuple[object, ...]` | Return concrete Dirichlet assets from nested model constraint sets. |
| function | `scalar_dirichlet(V, marker = None, value = 0.0, *, location = None, on = None, name: str = 'dirichlet') -> DirichletConstraint` | Semantic wrapper for scalar essential boundary data. |
| function | `component_dirichlet(V, component: int, marker = None, value = 0.0, *, location = None, on = None, name: str = 'dirichlet') -> DirichletConstraint` | Semantic wrapper for vector-component essential boundary data. |
| function | `axisymmetric_plane_strain(displacement, *, value: float = 0.0, name: str = 'axisymmetric_plane_strain') -> DirichletConstraint` | Constrain ``u_z`` everywhere in an ``(r, z)`` meridian model. |
| function | `axisymmetric_axis(displacement, *, location = None, on = None, value: float = 0.0, name: str = 'axisymmetric_axis') -> DirichletConstraint` | Enforce radial regularity ``u_r=0`` on the revolution axis. |
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
| function | `constraint_capabilities(constraint) -> ConstraintCapabilities \| None` | Return the public capability contract of a known constraint asset. |
| function | `constraint_balance_contract(constraints) -> dict[str, object]` | Describe whether strong-reaction force/work diagnostics are complete. |
| function | `validate_solver_compatibility(*, constraints, analysis: str, procedure: str \| None = None, comm_size: int = 1)` | Validate constraint/procedure compatibility before assembly or solve. |
| class | `PeriodicConstraintSpec` | Geometric description of a periodic constraint. |
| class | `ConstraintSet` | Collection of constraints used by assembly or field updates. |
| class | `AbaqusPeriodicConstraint` | Periodic equations controlled by prescribed or free reference dofs. |
| class | `AffineReduction` | Sparse serial representation of ``u = T q + offset``. |
| class | `DistributedAffineReduction` | Homogeneous correction space for a distributed affine constraint. |
| function | `abaqus_periodic_cell(target, *, nodes: AbaqusNodeTable, equations: AbaqusEquationSet, anchor_node: int, reference_nodes, deformation_gradient = None, control_displacements = None, tolerance: float = 1e-09, name: str = 'abaqus_periodic_cell') -> AbaqusPeriodicConstraint` | Create exact periodic equations and explicit macro-control semantics. |
| class | `RectangularPeriodicMPC` | Exact rectangular periodic relation and construction diagnostics. |
| function | `rectangular_periodic_mpc(target, *, axes = None, bcs = (), tolerance: float \| None = None, name: str = 'rectangular_periodic_mpc') -> RectangularPeriodicMPC` | Constrain maximum faces of a rectangular mesh to minimum faces. |

## `agentfem.amplitudes`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `Amplitude` | Named scalar history function. |
| class | `AmplitudeAudit` | Portable endpoint and range evidence for one amplitude. |
| function | `as_amplitude(value, *, name: str = 'amplitude') -> Amplitude` | Convert a scalar, callable, or ``Amplitude`` into an ``Amplitude``. |
| function | `constant(value: float, *, name: str = 'constant') -> Amplitude` | Create a constant amplitude. |
| function | `ramp(start_value: float = 0.0, end_value: float = 1.0, *, start_time: float = 0.0, end_time: float = 1.0, name: str = 'ramp') -> Amplitude` | Create a clipped linear ramp amplitude. |
| function | `smooth_step(start_value: float = 0.0, end_value: float = 1.0, *, start_time: float = 0.0, end_time: float = 1.0, name: str = 'smooth_step') -> Amplitude` | Create a clipped half-cosine transition with zero endpoint slopes. |
| function | `tabular(times, values, *, name: str = 'tabular', left: float \| None = None, right: float \| None = None) -> Amplitude` | Create a linearly interpolated tabular amplitude. |
| function | `sine(amplitude: float = 1.0, frequency: float = 1.0, *, phase: float = 0.0, offset: float = 0.0, name: str = 'sine') -> Amplitude` | Create a sinusoidal amplitude. |
| function | `gaussian_modulated_sine(amplitude: float, frequency: float, width: float, *, center: float \| None = None, phase: float = 0.0, name: str = 'gaussian_modulated_sine') -> Amplitude` | Create a Gaussian-windowed sinusoidal pulse. |
| class | `AmplitudeBasis` | Named, serializable loading modes with a declared coefficient order. |
| function | `basis(*components: Amplitude, name: str = 'amplitude_basis', coefficient_names: Sequence[str] \| None = None, coordinate_name: str = 'time', coordinate_unit: str \| None = 's', value_unit: str \| None = None) -> AmplitudeBasis` | Create a named basis for control, inverse, and transient studies. |

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
| function | `surface_force(resultant, *, location = None, on = None, reference_measure: float \| None = None, study = None, system = None, name: str = 'surface_force') -> SurfaceResultantLoad` | Distribute a total reference-configuration force over a boundary. |
| function | `distributing_coupling(force, *, moment = None, reference_point = None, location = None, on = None, system = None, name: str = 'distributing_coupling') -> DistributedCouplingLoad` | Distribute force/moment over a surface with tributary-area weighting. |
| function | `remote_force(force, *, reference_point, moment = None, location = None, on = None, system = None, name: str = 'remote_force') -> DistributedCouplingLoad` | Apply a reference-point force/moment through a continuum surface. |
| function | `pressure(value, *, location = None, on = None, normal = None, configuration: str = 'reference', displacement = None, name: str = 'pressure') -> PressureLoad` | Create inward pressure on a reference or current boundary. |
| function | `hydrostatic_pressure(*, density, gravity, reference_point, reference_pressure = 0.0, on = None, location = None, clip_at_zero: bool = True, configuration: str = 'reference', displacement = None, name: str = 'hydrostatic_pressure') -> HydrostaticPressureLoad` | Create ``p = p_ref + rho g dot (x - x_ref)`` on a boundary. |
| function | `heat_flux(value, *, location = None, on = None, name: str = 'heat_flux') -> BoundaryLoad` | Create a prescribed heat flux applied on a boundary region. |
| function | `body_force_form(force, test_function)` | Create a body-force virtual-work form. |
| function | `boundary_traction_form(traction, test_function, ds_measure)` | Create a boundary-traction virtual-work form. |

## `agentfem.project`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `new_run_id(now: datetime \| None = None) -> str` | Return a sortable, collision-resistant identifier for one execution. |
| class | `ProjectConfig` | Operational metadata for an AgentFEM case directory. |
| function | `discover(start: str \| Path \| None = None) -> ProjectConfig` | Find the nearest ``agentfem.toml`` from ``start`` upward. |
| class | `RunContext` | Filesystem and identity contract shared by scripts, CLIs, GUIs, and agents. |
| function | `current_run(*, project_root: str \| Path \| None = None, project_name: str \| None = None) -> RunContext` | Return the CLI-provided context or create one for direct Python use. |

## `agentfem.results`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CheckpointRecord` | One restart asset with an explicit portability boundary. |
| class | `FieldResult` | A named live field or an external field artifact. |
| class | `HistoryResult` | Time, load, or iteration history with a fixed value shape. |
| class | `ResultQuantity` | One scalar or fixed-shape quantity of interest. |
| class | `SimulationResult` | Scientific results and artifacts from one simulation. |
| function | `dof_statistics(field) -> dict[str, float \| int]` | Return global finite dof statistics for a DOLFINx-like field. |
| function | `from_solution(solution, *, name: str = 'result', field_name: str \| None = None, unit: str \| None = None, metadata: Mapping[str, object] \| None = None, scientific_inputs: Mapping[str, object] \| None = None) -> SimulationResult` | Wrap one solved field in a :class:`SimulationResult`. |
| class | `ForceMomentResultant` | Integrated force and moment about an explicit physical point. |
| class | `PathSample` | Values sampled along one straight physical-space path. |
| class | `RectilinearGridSample` | A finite-element field sampled on a Cartesian observation grid. |
| class | `StaticForceBalance` | Global algebraic force equilibrium for one linear static solid. |
| class | `StaticWorkBalance` | Energy closure including proportional prescribed boundary motion. |
| function | `average(expression, *, measure = ufl.dx, comm = None)` | Return the measure-weighted global average of an expression. |
| function | `boundary_resultant(traction, *, on, study = None)` | Integrate traction/flux over a named physical boundary. |
| function | `field_extrema(field, *, magnitude: bool = False, location: bool = False) -> dict[str, object]` | Return MPI-global field extrema, optionally with physical locations. |
| function | `free_body_resultant(*, boundary_tractions = (), body_forces = (), about) -> ForceMomentResultant` | Integrate boundary and volume forces into one free-body resultant. |
| function | `external_force_resultant(problem)` | Return the MPI-global resultant of a linear problem's assembled RHS. |
| function | `integral(expression, *, measure = ufl.dx, comm = None)` | Return the global integral of a scalar, vector, or tensor expression. |
| function | `l2_norm(expression, *, measure = ufl.dx, comm = None) -> float` | Return ``sqrt(integral(inner(value, value)))`` globally. |
| function | `probe(field, *, at, padding: float = 1e-10)` | Return one scalar, vector, or tensor field value at a physical point. |
| function | `quadrature_extrema(expression, domain, *, degree: int = 4) -> tuple[float, float]` | Return global min/max sampled at Basix quadrature points. |
| function | `reaction_resultant(problem, *, on = None, component: int \| None = None, name: str = 'RF')` | Return an MPI-global strong-constraint reaction resultant. |
| function | `region_average(expression, *, on, study = None)` | Return a measure-weighted average over a named mesh region. |
| function | `region_integral(expression, *, on, study = None)` | Integrate over a named region using its declared physical measure. |
| function | `region_measure(*, on, study = None) -> float` | Return the global length, area, or volume of a named region. |
| function | `sample_path(field, *, start, end, count: int = 101, padding: float = 1e-10, missing: str = 'raise') -> PathSample` | Sample a field along the straight segment from ``start`` to ``end``. |
| function | `sample_points(field, points, *, padding: float = 1e-10, missing: str = 'raise') -> np.ndarray` | Evaluate a finite-element field at common physical points under MPI. |
| function | `sample_rectilinear_grid(field, *, bbox, shape, reduction: str \| None = None, component: int \| None = None, padding: float = 1e-10) -> RectilinearGridSample` | Sample a scalar or vector field on a 2D/3D rectilinear grid. |
| function | `section_resultant(stress, *, on, normal = None, about = None) -> ForceMomentResultant` | Integrate section force and moment from a Cauchy/nominal stress field. |
| function | `static_force_balance(problem, *, constraints = ()) -> StaticForceBalance` | Evaluate ``R + F = 0`` for a converged linear static solid. |
| function | `static_work_balance(problem, *, constraints = ()) -> StaticWorkBalance` | Evaluate linear-static work including nonzero strong Dirichlet data. |
| function | `project(expression, *, domain = None, family: str = 'DG', degree: int = 0, name: str = 'ProjectedField', weight = 1.0)` | Return the global L2 projection of a UFL expression. |
| function | `project_piecewise(terms, *, domain = None, family: str = 'DG', degree: int = 0, name: str = 'ProjectedField', weight = 1.0)` | Project region-dependent expressions into one finite-element field. |
| function | `small_strain_cell_fields(displacement, properties, *, study = None, variables = ('S', 'E', 'MISES', 'SENER'), degree: int = 0) -> tuple[object, ...]` | Create standard projected fields for linear small-strain elasticity. |
| function | `small_strain_partition_fields(displacement, assignments, *, study = None, variables = ('S', 'E', 'MISES', 'SENER'), degree: int = 0) -> tuple[object, ...]` | Create standard fields for a complete regional material partition. |
| class | `FieldRecovery` | A reviewable conversion from constitutive evidence to a field. |
| function | `cell_average_recovery() -> FieldRecovery` | Return the standard scientific integration-point recovery policy. |
| function | `recover_integration_point_field(source, *, name: str \| None = None, policy: FieldRecovery \| None = None, unit: str \| None = None, description: str = '') -> FieldResult` | Recover one ``QuadratureField`` without hiding its processing history. |
| function | `add_execution_trace(result, events: Iterable[object]) -> tuple[dict[str, object], ...]` | Attach one complete execution trace and its standard histories. |
| function | `execution_records(events: Iterable[object]) -> tuple[dict[str, object], ...]` | Normalize solver events without depending on a particular procedure. |
| function | `complete_result(step, result, *, output = None, fields = (), strict_output: bool = False, deformation_scale: float = 0.0, metadata: Mapping[str, object] \| None = None)` | Complete output and metadata through one compatibility-safe path. |
| function | `execution_context(step)` | Return the context bound by :meth:`Model.step`, when available. |
| class | `HillMandelIncrement` | Finite-strain macrohomogeneity evidence over one accepted increment. |
| class | `HomogenizedFrame` | Macroscopic response reconstructed from one periodic-cell state. |
| class | `LiveFiniteStrainCellFields` | Derived cell fields refreshed from active Explicit state at output time. |
| class | `StressStateInvariants` | Three-dimensional Cauchy-stress invariants with explicit validity. |
| function | `cauchy_stress_invariants(stress, *, relative_tolerance: float = 1e-12) -> StressStateInvariants` | Return triaxiality and normalized Lode state from a 3D Cauchy tensor. |
| function | `finite_strain_dynamic_cell_fields(displacement, velocity, properties, *, variables = ('SENER', 'KED', 'J'), pressure = None, density = None) -> LiveFiniteStrainCellFields` | Create reusable SED/KED/stress fields for Explicit saved frames. |
| function | `finite_strain_diagnostics(displacement, *, constraint = None, quadrature_degree: int = 4) -> dict[str, object]` | Evaluate reusable physical checks for a finite-deformation solution. |
| function | `finite_strain_cell_fields(displacement, properties, *, variables = ('F', 'E', 'GREEN', 'P', 'S', 'MISES', 'J', 'SENER', 'EVOL'), pressure = None, velocity = None, density = None) -> tuple[object, ...]` | Create requested standard P0 finite-strain cell fields. |
| function | `homogenize_periodic_cell(displacement, properties, *, pressure = None, accepted_fields = None, macro_deformation_gradient, cell_reference_volume: float, load_factor: float) -> HomogenizedFrame` | Return volume-normalized macroscopic finite-strain response. |
| function | `homogenize_periodic_path(snapshots, properties, *, constraint) -> tuple[HomogenizedFrame, ...]` | Homogenize every saved state of an affine periodic-cell analysis. |
| function | `hill_mandel_increment(start_snapshot, snapshot, properties, *, constraint, start_frame: HomogenizedFrame \| None = None, frame: HomogenizedFrame \| None = None) -> HillMandelIncrement` | Compare microscopic and macroscopic first-Piola work increments. |
| function | `hill_mandel_periodic_path(snapshots, properties, *, constraint, frames = None) -> tuple[HillMandelIncrement, ...]` | Evaluate Hill--Mandel evidence between consecutive saved states. |
| function | `write_homogenized_csv(path: str \| Path, frames, *, hill_mandel = (), increment_info = ()) -> Path` | Write flattened macro tensors in a human-readable table. |
| function | `write_homogenized_history(path: str \| Path, frames, *, hill_mandel = (), increment_info = ()) -> Path` | Write an exact, compact NumPy history for plotting and ML reuse. |
| class | `FieldVariable` | Stable public meaning of one result variable. |
| function | `field_variable(name: str, *, finite_strain: bool = False) -> FieldVariable` | Resolve a standard variable, including the context-dependent ``E`` alias. |
| function | `preselected_fields(*, physics: str, finite_strain: bool = False) -> tuple[str, ...]` | Return the engineering-default field set for one physics context. |
| function | `resolve_field_variables(names, *, finite_strain: bool = False) -> tuple[FieldVariable, ...]` | Resolve aliases, preserve request order, and remove duplicates. |
| class | `WeightedFieldStatistics` | Global weighted distribution with explicit field semantics. |
| function | `weighted_field_statistics(values, weights, *, quantiles: Sequence[float] = (0.05, 0.5, 0.95), thresholds: Sequence[float] = (), location: str, representation: str, comm = None) -> WeightedFieldStatistics` | Return exact global statistics from physical sample weights. |
| class | `FiniteStrainDiagnosticRequest` | Record physical admissibility and constraint checks. |
| class | `HistoryRequest` | Evaluate one scientific quantity on every accepted output frame. |
| class | `OutputPlan` | One declarative output contract for a completed finite-strain step. |
| class | `PeriodicCellHistoryRequest` | Record complete tensor histories for a finite-strain periodic cell. |
| class | `ProbeHistoryRequest` | Record a field value at one physical point on every accepted frame. |
| class | `PresentationOutput` | Optional serial rendering from the scientific XDMF/HDF5 series. |
| class | `SolverHistoryRequest` | Record accepted-increment convergence history. |
| class | `SourceNodeHistoryRequest` | Record U and current coordinates using source-mesh node labels. |
| function | `finite_strain_checks(*, constraint = None, quadrature_degree: int = 4) -> FiniteStrainDiagnosticRequest` | Public AgentFEM object. |
| function | `history(name: str, evaluate, *, coordinate = None, unit: str \| None = None, abscissa_name: str \| None = None, abscissa_unit: str \| None = None, description: str = '') -> HistoryRequest` | Create a scalar history evaluated on accepted analysis states. |
| function | `output_plan(directory, *, field: FieldOutput \| None = None, requests = (), presentation: PresentationOutput \| None = None, basename: str = 'results') -> OutputPlan` | Create a complete finite-strain output plan. |
| function | `periodic_cell_history(constraint, *, basename: str = 'homogenized_history') -> PeriodicCellHistoryRequest` | Public AgentFEM object. |
| function | `probe_history(name: str, *, at, field = None, component: int \| None = None, unit: str \| None = None, description: str = '') -> ProbeHistoryRequest` | Create a point-probe history for accepted static or transient states. |
| function | `presentation(*, comparison: bool = True, animation: str \| None = 'gif', scalar: str = 'UMAG', fps: int = 2) -> PresentationOutput` | Public AgentFEM object. |
| function | `solver_history() -> SolverHistoryRequest` | Public AgentFEM object. |
| function | `source_node_history(nodes, **points: int) -> SourceNodeHistoryRequest` | Public AgentFEM object. |
| class | `FieldOutput` | What fields to save, how often, and in which configuration. |
| class | `FieldOutputArtifacts` | Files and final live fields produced by one output plan. |
| class | `ResultFieldArtifacts` | One completed-result field dataset and its explicit layout contract. |
| class | `UnifiedXDMFTimeSeries(path, *, deformation_scale: float = 0.0, store_reference_geometry: bool = True, compression: int = 4) -> None` | Incremental single-grid XDMF/HDF5 writer for serial result histories. |
| function | `field_output(*variables, every: int \| str \| None = None, intervals: int \| None = None, configuration: str = 'deformed', deformation_scale: float = 1.0, backend: str = 'xdmf') -> FieldOutput` | Create a concise, inspectable field-output request. |
| function | `read_unified_xdmf_series(xdmf_path) -> tuple[object, ...]` | Read AgentFEM's compact XDMF/HDF5 frames as PyVista grids. |
| function | `write_deformed_vtk_series(pvd_path, snapshots, cell_fields, *, deformation_scale: float = 1.0) -> tuple[Path, tuple[Path, ...]]` | Write one deformed VTU grid per frame and a ParaView PVD collection. |
| function | `write_parallel_vtk_series(path, snapshots, fields_by_frame) -> Path` | Write collective single-dataset ParaView frames under MPI. |
| function | `write_result_fields(result, path, *, time: float = 0.0, names = (), deformation_scale: float = 0.0) -> ResultFieldArtifacts` | Write the live, visualization-ready fields of one SimulationResult. |
| function | `write_unified_xdmf_series(xdmf_path, snapshots, cell_fields, *, deformation_scale: float = 1.0, store_reference_geometry: bool = True, compression: int = 4) -> Path` | Write one temporal XDMF and one compressed HDF5 heavy-data file. |
| function | `render_deformation_animation(undeformed_path, snapshots, nodes, output_path, *, fps: int = 2) -> Path` | Render scale-one deformation history as GIF or MP4. |
| function | `render_deformation_comparison(undeformed_path, deformed_path, output_path, *, scalar: str = 'DisplacementMagnitude') -> Path` | Render side-by-side undeformed/deformed surfaces with PyVista. |
| function | `render_unified_xdmf_animation(xdmf_path, output_path, *, scalar: str = 'UMAG', fps: int = 2) -> Path` | Render a GIF or MP4 from AgentFEM's single XDMF/HDF5 series. |
| function | `render_unified_xdmf_comparison(xdmf_path, output_path, *, scalar: str = 'UMAG') -> Path` | Render the first and final grids from a unified XDMF series. |
| function | `render_vtk_series_animation(frame_paths, output_path, *, scalar: str = 'UMAG', fps: int = 2) -> Path` | Render a GIF directly from a combined-field deformed VTU series. |

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

## `agentfem.units`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `UnitSystem` | Named consistent base-unit contract attached to a model. |
| function | `consistent(*, length, mass, time, temperature = 'K', name = 'consistent_units')` | Declare the base units used consistently by all model inputs. |
| function | `si(*, temperature = 'K') -> UnitSystem` | Return the SI ``m-kg-s`` engineering contract. |
| function | `n_mm_mpa(*, temperature = 'K') -> UnitSystem` | Return the common ``mm-N-s-MPa`` consistent system. |

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

## `agentfem.assessments`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `SequentialEnergyLedger` | Layered evidence for a one-way heat-to-mechanics workflow. |
| function | `sequential_energy_ledger(thermal_result, mechanical_result, *, field_history = None) -> SequentialEnergyLedger` | Audit thermal and mechanical energy channels across sequential steps. |
| class | `CreepDamageBlock` | One dwell or service block for the time-fraction rule. |
| class | `DwellInterval` | One explicitly declared hold interval in a result time history. |
| function | `creep_blocks_from_result(result, *, stress_history: str, temperature_history: str, dwells: Iterable[DwellInterval], rupture_time: Callable[[float, float], float], rupture_source: str, stress_reducer: str = 'maximum_absolute', temperature_reducer: str = 'maximum') -> tuple[CreepDamageBlock, ...]` | Create source-identified creep blocks from named result histories. |
| class | `CreepDamageAssessment` | Auditable linear time-fraction assessment over service blocks. |
| class | `InteractionDiagram` | Declared creep/fatigue allowable boundary in damage coordinates. |
| class | `CreepFatigueAssessment` | Combined engineering assessment from independent damage consumers. |
| function | `creep_time_fraction(blocks: Iterable[CreepDamageBlock]) -> CreepDamageAssessment` | Evaluate the linear creep time-fraction rule for declared blocks. |
| function | `interaction_diagram(*, points, name: str, source: str) -> InteractionDiagram` | Create an explicit piecewise-linear creep/fatigue interaction curve. |
| function | `linear_interaction() -> InteractionDiagram` | Return the transparent reference boundary ``Dc + Df = 1``. |
| function | `creep_fatigue(*, fatigue: FatigueAssessment, creep: CreepDamageAssessment, interaction: InteractionDiagram \| None = None) -> CreepFatigueAssessment` | Combine existing fatigue and creep assessments against one boundary. |
| function | `creep_fatigue_from_result(result, *, fatigue_history: str, fatigue_curve, stress_history: str, temperature_history: str, dwells: Iterable[DwellInterval], rupture_time: Callable[[float, float], float], rupture_source: str, interaction: InteractionDiagram \| None = None, ultimate_strength: float \| None = None, stress_reducer: str = 'maximum_absolute', temperature_reducer: str = 'maximum') -> CreepFatigueAssessment` | Build the engineering V1 assessment from named result histories. |

## `agentfem.boundary_models`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ElasticFoundation` | Distributed linear spring support on a solid boundary. |
| function | `elastic_foundation(*, on = None, location = None, stiffness, mode: str = 'isotropic', normal = None, name: str = 'elastic_foundation') -> ElasticFoundation` | Public AgentFEM object. |
| class | `ConvectionBoundary` | Linear convection ``-k grad(T).n = h (T - T_inf)``. |
| function | `convection(*, on = None, location = None, coefficient, ambient_temperature, name: str = 'convection') -> ConvectionBoundary` | Create a linear thermal convection boundary condition. |

## `agentfem.campaigns`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ChoiceParameter` | Finite categorical or policy parameter. |
| class | `IntegerParameter` | Bounded integer parameter. |
| class | `ParameterSpace` | Ordered scientific input schema for a campaign. |
| class | `RealParameter` | Bounded continuous parameter with optional units and log scaling. |
| class | `SamplingPlan` | Immutable, validated collection of parameter samples. |
| function | `explicit(space: ParameterSpace, samples: Iterable[Mapping[str, object]], *, metadata: Mapping[str, object] \| None = None) -> SamplingPlan` | Create a plan from caller-supplied samples. |
| function | `full_factorial(space: ParameterSpace, levels: int \| Mapping[str, int] = 3) -> SamplingPlan` | Create a full-factorial design in normalized coordinates. |
| function | `latin_hypercube(space: ParameterSpace, count: int, *, seed: int = 0) -> SamplingPlan` | Draw a reproducible Latin-hypercube design. |
| function | `random(space: ParameterSpace, count: int, *, seed: int = 0) -> SamplingPlan` | Draw reproducible independent uniform samples in normalized space. |
| class | `Campaign(*, name: str, parameter_space: ParameterSpace, outputs: tuple[Quantity, ...], evaluate: Callable[[object], Mapping[str, object] \| CaseOutcome \| SimulationResult], build: Callable[[Mapping[str, object]], object] \| None = None, metadata: Mapping[str, object] \| None = None, scientific_inputs: Mapping[str, object] \| None = None, execution: ExecutionPolicy \| None = None) -> None` | Build and evaluate a collection of immutable scientific cases. |
| class | `CampaignCase` | One immutable case in a campaign plan. |
| class | `CampaignPlan` | Immutable cases and their design-of-experiment evidence. |
| class | `CampaignReport` | Case-level evidence and the successful scientific dataset. |
| class | `CaseOutcome` | Successful case outputs plus links to scientific evidence. |
| class | `CaseRunRecord` | Execution evidence for one attempted case. |
| class | `ExecutionPolicy` | Declared execution behavior for the current campaign runner. |
| function | `case_id(campaign_name: str, parameters: Mapping[str, object], *, schema_version: str = CAMPAIGN_SCHEMA_VERSION) -> str` | Return a deterministic scientific case identity. |
| function | `create(**kwargs) -> Campaign` | Create a :class:`Campaign` using the public functional spelling. |
| function | `local_processes(*, workers: int \| None = None, fail_fast: bool = False, resume: bool = True) -> ExecutionPolicy` | Use spawned local processes for independent campaign cases. |
| class | `CampaignSpecification` | Validated declarative part of a campaign. |
| function | `load_specification(path: str \| Path) -> CampaignSpecification` | Load a safe JSON campaign specification. |
| function | `specification_from_dict(record: Mapping[str, object]) -> CampaignSpecification` | Validate a dictionary and construct a campaign specification. |

## `agentfem.convergence`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ConvergenceAxis` | One refinement coordinate with all other coordinates fixed explicitly. |
| class | `ObservablePolicy` | How one scalar, vector, event, or topology record is compared. |
| class | `ConvergenceCheck` | One observable checked along one explicitly selected refinement axis. |
| class | `ConvergenceCertificate` | Auditable multi-axis convergence decision for one CampaignReport. |
| function | `axis(parameter: str, *, fixed: Mapping[str, object] \| None = None, discretization: str = 'mesh', characteristic: Characteristic = 'value') -> ConvergenceAxis` | Public AgentFEM object. |
| function | `observable(name: str, *, comparison: Comparison = 'relative', tolerance: float \| None = None, source: Source = 'output', path: str \| None = None, minimum_observed_order: float \| None = None, unit: str \| None = None) -> ObservablePolicy` | Public AgentFEM object. |
| function | `audit(report: CampaignReport, *, axes: tuple[ConvergenceAxis, ...], observables: tuple[ObservablePolicy, ...], output: str \| Path \| None = None) -> ConvergenceCertificate` | Build a conservative convergence certificate from campaign evidence. |

## `agentfem.checkpointing`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CheckpointPolicy` | Automatic accepted-increment checkpoint cadence for transient steps. |
| function | `every(increments: int, *, directory = 'checkpoints', final: bool = True, prefix: str \| None = None, keep_last: int \| None = None, portable: bool = False) -> CheckpointPolicy` | Create an automatic checkpoint policy for accepted time increments. |
| function | `save_transient_checkpoint(path, *, step_kind: str, step_name: str, procedure, dt: float, total_steps: int, completed_steps: int, state: dict[str, object], accepted_times = (), execution_events = (), history_records = (), auxiliary_state: dict[str, object] \| None = None, portable: bool = False)` | Write a transient restart, optionally with partition-independent state. |
| function | `load_transient_checkpoint(path, *, step_kind: str, step_name: str, procedure, dt: float, total_steps: int, state: dict[str, object]) -> dict[str, object]` | Restore a transient state after validating its scientific identity. |
| function | `save_portable_state_bundle(path, *, state: dict[str, object]) -> dict[str, object]` | Collectively publish a portable nodal-state bundle. |
| function | `load_portable_state_bundle(path, *, state: dict[str, object], record: dict[str, object], identities: dict[str, object]) -> None` | Collectively restore a bundle written by :func:`save_portable_state_bundle`. |
| function | `checkpoint_file_record(path) -> dict[str, object]` | Describe one checkpoint payload by name, size, and digest. |
| function | `validate_checkpoint_record(directory, record: dict[str, object]) -> Path` | Validate and return a payload referenced by a scientific manifest. |
| function | `function_portable_identity(function) -> dict[str, object]` | Return an MPI-partition-independent identity for a nodal field. |
| function | `mesh_portable_identity(domain) -> dict[str, object]` | Hash cell geometry independently of local numbering and partition. |
| function | `remove_stateful_checkpoint(path, *, comm) -> None` | Collectively remove one manifest and only its declared state payloads. |
| function | `function_partition_identity(function) -> dict[str, object]` | Return a JSON-safe identity for one field on one mesh partition. |
| function | `atomic_savez(path, **arrays) -> Path` | Atomically publish one NumPy archive in its destination directory. |
| function | `atomic_write_text(path, content: str) -> Path` | Atomically publish UTF-8 text in its destination directory. |

## `agentfem.coordinates`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CartesianSystem` | Right-handed orthonormal Cartesian coordinate system. |
| class | `ReferencePoint` | Named engineering point used for remote resultants and kinematics. |
| function | `cartesian(*, origin = None, axes = None, x = None, y = None, z = None, name = 'local') -> CartesianSystem` | Create a Cartesian system from a matrix or named basis vectors. |
| function | `reference_point(coordinates, *, name = 'reference_point', system = None) -> ReferencePoint` | Create a named engineering reference point. |

## `agentfem.datasets`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `DatasetSplit` | Reproducible train/validation partition. |
| class | `ScientificDataset` | A numeric dataset whose columns retain scientific meaning. |
| class | `ExternalDatasetAudit` | Local evidence that downloaded public data matches its manifest. |
| class | `ExternalDatasetManifest` | Versioned public dataset identity, scope, and local audit policy. |
| class | `ExternalFile` | One immutable file identity in a public scientific dataset. |
| class | `SpreadsheetSheet` | Rectangular values from one XLSX worksheet. |
| class | `SpreadsheetWorkbook` | Dependency-free, read-only representation of one XLSX workbook. |
| function | `read_xlsx_workbook(path: str \| Path) -> SpreadsheetWorkbook` | Read values and cached formula results from an XLSX without pandas. |
| function | `science_supershear_dryad_manifest() -> ExternalDatasetManifest` | Return the pinned CC0 Dryad v7 manifest for Science 2023. |
| function | `science_supershear_v5_research_task() -> dict[str, object]` | Return the installed machine-readable V5 research handoff. |
| class | `Quantity` | One scalar, curve, vector, or sampled-field output contract. |
| class | `Sample` | One successful simulation sample and its scientific lineage. |
| function | `decode_quantities(quantities: tuple[Quantity, ...], row) -> dict[str, object]` | Restore one flattened numeric row to declared named quantities. |
| class | `RectilinearObservation` | One scalar field on explicit physical ``x``/``y`` axes. |
| class | `FEMFieldSample` | One FEM field representation with coordinates and scientific encoding. |
| class | `TorchDatasetBundle` | PyTorch dataset plus the schema needed to interpret its columns. |
| function | `fem_field_sample(function, encoding) -> FEMFieldSample` | Export owned nodal coefficients for external neural/PINN tooling. |
| function | `fem_observation_sample(function, grid, *, name: str \| None = None, unit: str \| None = None, role: str = 'output', components = (), outside: str = 'raise', fill_value: float = 0.0, coordinate_map = None, configuration: str = 'reference') -> FEMFieldSample` | Sample a FEM field on a reusable structured observation grid. |
| function | `to_torch(dataset: ScientificDataset, *, normalized_inputs: bool = True, dtype: str = 'float32', device: str = 'cpu') -> TorchDatasetBundle` | Expose a validated campaign dataset as a PyTorch ``TensorDataset``. |

## `agentfem.events`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `FirstPassageEvent` | One threshold event with explicit localization and censoring evidence. |
| function | `first_passage(abscissa, values = None, *, threshold: float, direction: EventDirection = 'rising', localization: str = 'linear', component: int \| tuple[int, ...] \| None = None, name: str = 'first_passage', coordinate_name: str \| None = None, coordinate_unit: str \| None = None, value_name: str \| None = None, value_unit: str \| None = None) -> FirstPassageEvent` | Locate the first threshold crossing in a history or numeric arrays. |

## `agentfem.expressions`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ExpressionError` | Raised when a scientific expression is invalid or unsupported. |
| class | `ScientificExpression` | An inspectable expression that can be lowered to UFL. |
| function | `expression(source: str \| Real \| ScientificExpression) -> ScientificExpression` | Return a validated :class:`ScientificExpression`. |
| function | `as_ufl(source, domain, *, parameters: Mapping[str, object] \| None = None)` | Validate and lower one scalar expression to UFL. |
| function | `vector_as_ufl(sources: Sequence[str \| Real \| ScientificExpression], domain, *, parameters: Mapping[str, object] \| None = None)` | Validate and lower a vector of scalar expressions to UFL. |
| function | `interpolate(target, source, *, parameters: Mapping[str, object] \| None = None) -> object` | Interpolate a validated scalar or vector expression into ``target``. |

## `agentfem.fatigue_fracture`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ForceCycle` | One scalar cyclic-load definition expressed in test parameters. |
| function | `force_cycle(*, minimum: float \| None = None, maximum: float \| None = None, fmin: float \| None = None, fmax: float \| None = None, frequency: float = 1.0, waveform: str = 'sine', hold_minimum_fraction: float = 0.0, hold_maximum_fraction: float = 0.0, table = (), name: str = 'force cycle') -> ForceCycle` | Create a cyclic force from ``minimum/maximum`` or ``fmin/fmax``. |
| class | `CycleJumpDecision` | One inspectable proposal for advancing the independent cycle count. |
| class | `CycleJumpPolicy` | Bound a cycle block by predicted damage and exact output landings. |
| class | `CycleJumpRecord` | Accepted or rejected cycle-block evidence. |
| class | `CycleJumpLedger(*, start_cycle: int = 0)` | Record exact cycle progress and every jump/cutback decision. |
| class | `CyclicCohesiveResponse` | Mode-I response with separated monotonic and fatigue evidence. |
| class | `CyclicCohesiveLaw` | Replaceable power-law range fatigue layered on a bilinear envelope. |
| class | `CyclicCohesiveTransaction(law: CyclicCohesiveLaw, size: int)` | Atomic monotonic trials and cycle-block trials for cohesive points. |
| class | `MixedModeEnergyRange` | Local cohesive-energy driver from one physical peak/valley pair. |
| class | `OrderedJumpCyclePath` | One ordered closed cycle of complete local cohesive jump vectors. |
| class | `MixedModeEnergyPath` | Segment-resolved local cohesive energy evidence for one ordered path. |
| class | `OrderedMixedModeEnergyPathDriver` | Segment-resolved BK/power driver for ordered mixed-mode cycles. |
| class | `MixedModeEnergyRangeDriver` | BK/power interaction for local mixed-mode cyclic energy ranges. |
| class | `MixedModeCyclicCohesiveResponse` | Mixed-mode monotonic response with committed cyclic evidence. |
| class | `MixedModeCyclicCohesiveLaw` | Replaceable cyclic damage layered on a mixed-mode cohesive envelope. |
| class | `MixedModeCyclicCohesiveTransaction(law: MixedModeCyclicCohesiveLaw, size: int)` | Atomic full-vector cycle transaction with mixed-mode energy evidence. |
| class | `FieldStateTransaction(fields, *, assets = None)` | In-memory rollback for bulk fields and other transactional assets. |
| class | `GeneralizedWorkSample` | One named force--displacement pair at an accepted equilibrium point. |
| function | `generalized_work_sample(name, *, force, displacement, role = 'natural_load') -> GeneralizedWorkSample` | Declare one generalized work-conjugate channel. |
| function | `reference_point_work_sample(load, *, translation, rotation = None) -> GeneralizedWorkSample` | Pair a distributed reference load with measured rigid motion. |
| class | `CyclicEnergyFrame` | One accepted or trial cycle-block work--energy closure. |
| class | `CyclicWorkEnergyLedger(*, name = 'cyclic work-energy ledger')` | Transactional generalized-work and cycle-block energy accounting. |
| function | `cyclic_work_energy_ledger(**options) -> CyclicWorkEnergyLedger` | Create a transactional cycle-block work--energy ledger. |
| class | `CyclicEquilibriumPoint` | Evidence returned by one converged cyclic equilibrium solve. |
| class | `CyclicFatigueBlock` | Accepted structure-level cycle block and its error evidence. |
| class | `GlobalCyclicFatigueStep(*, cycle: ForceCycle, stop_cycle: int, interfaces, state, solve_equilibrium, jump: CycleJumpPolicy \| None = None, landing_cycles = (), maximum_opening_feedback: float = 0.02, maximum_energy_balance_error: float \| None = None, energy_ledger: CyclicWorkEnergyLedger \| None = None, ordered_path_phases = (), observe = None, name: str = 'cyclic fatigue')` | Quasi-static cyclic fatigue loop with global rollback and cutback. |
| class | `SurfaceCrackComponent` | One connected failed component in a surface-crack observation. |
| class | `SurfaceCrackObservation` | One cycle's geometric evidence on a triangular cohesive surface. |
| class | `CrackTopologyEvent` | Auditable identity change between two accepted crack observations. |
| class | `TrackedSurfaceCrack` | A connected crack component with identity stable across cycle blocks. |
| class | `SurfaceCrackTrackingFrame` | Persistent component identities and topology events at one cycle. |
| class | `SurfaceCrackTracker(*, interface_name: str, id_prefix: str \| None = None)` | Track cracks on one fixed cohesive surface by physical facet identity. |
| class | `CrackInteractionObservation` | Two-crack geometry and growth evidence at one exact cycle. |
| function | `observe_surface_crack(coordinates, facets, damage, opening, *, cycle: int, name: str = 'surface crack', damage_threshold: float = 0.95, include_boundary_front: bool = False, facet_ids = None) -> SurfaceCrackObservation` | Recover connected failed area and a three-dimensional crack front. |
| function | `surface_crack_interaction(first: SurfaceCrackObservation, second: SurfaceCrackObservation, *, first_single_growth_rate: float \| None = None, second_single_growth_rate: float \| None = None, first_double_growth_rate: float \| None = None, second_double_growth_rate: float \| None = None, coalescence_tolerance: float = 0.0) -> CrackInteractionObservation` | Compare two named fronts without hiding the single-crack baseline. |
| class | `ParisEvidence` | Postprocessed Paris-region evidence; never a crack-growth solver law. |
| function | `paris_evidence(cycles, crack_size, driving_force, *, fit_cycle_range: tuple[float, float] \| None = None, fit_mask = None, derivative_window: int = 3, driving_force_name: str = 'Delta K', crack_size_name: str = 'a', driving_force_unit: str = 'declared', crack_size_unit: str = 'declared') -> ParisEvidence` | Fit a Paris relation after simulation from ``a(N)`` and a driver. |
| function | `cyclic_cohesive(*, monotonic: BilinearCohesiveLaw \| MixedModeBilinearCohesiveLaw, fatigue_coefficient: float, fatigue_exponent: float, range_threshold: float, peak_exponent: float = 0.0, residual_exponent: float = 0.0, driver: MixedModeEnergyRangeDriver \| OrderedMixedModeEnergyPathDriver \| None = None, name: str \| None = None) -> CyclicCohesiveLaw \| MixedModeCyclicCohesiveLaw` | Create a Mode-I or mixed-mode cyclic cohesive law. |
| function | `mixed_mode_energy_range_driver(**options) -> MixedModeEnergyRangeDriver` | Create the first proportional peak/valley mixed-mode fatigue driver. |
| function | `ordered_jump_cycle(phases, jumps, *, name = 'ordered jump cycle')` | Create a closed, station-resolved local cohesive cycle. |
| function | `ordered_mixed_mode_energy_path_driver(**options) -> OrderedMixedModeEnergyPathDriver` | Create a segment-resolved non-proportional mixed-mode driver. |
| function | `field_state(fields = None, *, assets = None, **named_fields) -> FieldStateTransaction` | Create rollback state from a field mapping or named field arguments. |
| function | `global_cyclic_fatigue_step(**kwargs) -> GlobalCyclicFatigueStep` | Create a reusable extrema- or ordered-path fatigue controller. |

## `agentfem.fracture`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `finite_strain_internal_force(displacement, test_function, material, *, measure = ufl.dx, name: str = 'F_internal_finite_strain') -> OperatorForm` | Return the current Total-Lagrangian hyperelastic internal force. |
| class | `FiniteStrainEnergyMonitor` | Accepted-frame kinetic and hyperelastic bulk energy monitor. |
| class | `DofMappedCohesiveForce(assembler, displacement, *, node_to_block_dof)` | Map a serial cohesive facet kernel to vector finite-element dofs. |
| class | `NamedCohesiveResponse` | Responses and aggregate energy from several named interfaces. |
| class | `CohesiveForceCollection(interfaces)` | Atomically compose independent named cohesive-interface forces. |
| function | `named_cohesive_forces(**interfaces) -> CohesiveForceCollection` | Create an atomically managed collection from named cohesive forces. |
| function | `named_mode_i_cohesive_forces(split, displacement, *, laws, normal_hints, thicknesses = None, tangential = 'free', tangential_stiffness = None, tolerance: float = 1e-10) -> CohesiveForceCollection` | Build independent named forces on one atomically split solver mesh. |
| function | `cohesive_forces(split, displacement, *, laws, normal_hints, thicknesses = None, tangential = None, tangential_stiffness = None, tolerance: float = 1e-10) -> CohesiveForceCollection` | Build a recommended force for every named split interface. |
| class | `DistributedDofMappedCohesiveForce(assembler, displacement, *, input_node_to_block_dof, input_node_owned, global_topology, global_facet_indices, local_input_nodes = None)` | MPI assembler for a physical-keyed split interface. |
| function | `p1_input_node_to_block_dof(displacement, *, number_of_input_nodes: int)` | Recover the complete serial input-node to block-DOF map. |
| function | `mode_i_cohesive_force(split: interface_api.SplitInterfaceMesh, displacement, law, *, normal_hint, thickness: float = 1.0, tolerance: float = 1e-10, tangential: str = 'free', tangential_stiffness: float \| None = None) -> DofMappedCohesiveForce \| DistributedDofMappedCohesiveForce` | Build a fixed-path cohesive force from a split mesh contract. |
| function | `cohesive_force(split: interface_api.SplitInterfaceMesh, displacement, law, *, normal_hint, tangential: str \| None = None, tangential_stiffness: float \| None = None, thickness: float = 1.0, tolerance: float = 1e-10)` | Build the recommended full-vector fixed-path interface consumer. |
| class | `FiniteStrainCohesiveResidual(bulk, cohesive)` | Assemble bulk UFL and paired-facet interface forces into one residual. |
| class | `CohesiveNewtonSolveInfo` | Convergence evidence for one native bulk-plus-interface equilibrium. |
| class | `ArcLengthOptions` | Crisfield-style spherical continuation controls. |
| class | `ArcLengthSolveInfo` | Public AgentFEM object. |
| class | `FiniteStrainCohesiveEquilibrium(residual: FiniteStrainCohesiveResidual, tangent, displacement, *, set_load = None, load_parameter = None, reference_load: float = 1.0, bcs = (), solver_options = None, control_displacement = None, reaction = None, bulk_strain_energy = None)` | Native Newton consumer for UFL bulk and zero-thickness interfaces. |
| class | `FiniteStrainCohesiveArcLength(equilibrium: FiniteStrainCohesiveEquilibrium, options: ArcLengthOptions, *, initial_load: float = 0.0)` | Spherical arc-length continuation for cohesive equilibrium paths. |
| class | `MassProportionalDampingResidual(base, *, mass, velocity, coefficient: float, dt: float)` | Add ``alpha M v_mid`` with transactional dissipation accounting. |
| class | `DampingEnergyMonitor` | Add accepted viscous dissipation to an existing mechanical monitor. |
| class | `FiniteStrainCohesiveEnergyMonitor` | Typed accepted-frame energy for bulk plus cohesive dynamics. |
| class | `DynamicEnergyLedger` | Accepted-frame external work and mechanical-energy closure. |
| class | `IsotropicWaveSpeeds` | Reference small-on-zero wave speeds for one isotropic material. |
| class | `IncrementalWaveSpeeds` | Small-on-large bulk-wave modes about one homogeneous deformation. |
| class | `PrincipalSurfaceWaveSpeed` | Reference-coordinate principal surface-wave secular solution. |
| function | `neo_hookean_material_tangent(deformation_gradient, material) -> np.ndarray` | Return ``A[i,J,k,L] = dP[i,J]/dF[k,L]`` for a supported energy. |
| function | `incremental_wave_speeds(deformation_gradient, direction, material, *, direction_configuration: str = 'current') -> IncrementalWaveSpeeds` | Return homogeneous small-on-large bulk-wave speeds. |
| function | `principal_surface_wave_speed(deformation_gradient, material: hyperelasticity.NeoHookeanProperties, *, propagation_axis: int = 0, scan_points: int = 320) -> PrincipalSurfaceWaveSpeed` | Solve the 2D small-on-large principal surface-wave secular problem. |
| function | `isotropic_reference_wave_speeds(material) -> IsotropicWaveSpeeds` | Return unstretched 3D isotropic ``c_d``, ``c_s``, and ``c_R``. |
| class | `StableTimeIncrement` | Visible body/interface estimate for central difference. |
| class | `CohesiveCrackHistory` | Crack-front position and window-fitted speed on a fixed path. |
| class | `CrackPropagationFit` | Representative crack speed fitted across a declared path interval. |
| class | `InterfaceFrontHistory` | Front position and fitted speed for one declared interface signal. |
| class | `CohesiveFrontEnsemble` | Crack-front evidence from multiple thresholds and physical signals. |
| class | `CohesiveInterfaceTrace` | Portable accepted-frame record on one fixed cohesive interface. |
| class | `ScientificComparison` | Common scalar evidence for a simulation-to-observation comparison. |
| class | `PreloadTransferReport` | Evidence for a quasi-static displacement to Explicit state transfer. |
| function | `transfer_preload_to_explicit(preload_displacement, *, state, mass, residual, initial_velocity = None, mode: str = 'equilibrium', force_tolerance: float = 1e-08, acceleration_projection = None, energy_monitor = None, source_energy: float \| None = None, source_step: str \| None = None, destination_step: str \| None = None) -> PreloadTransferReport` | Initialize ``u/v/a`` consistently from a quasi-static preload state. |
| function | `cohesive_crack_tip(path_coordinate, damage, *, threshold: float = 0.95, direction: str = 'increasing') -> float` | Locate the contiguous crack front by interpolating a damage threshold. |
| function | `crack_tip_history(time_values, path_coordinate, damage_frames, *, threshold: float = 0.95, fit_window: int = 5, direction: str = 'increasing') -> CohesiveCrackHistory` | Build a crack history without single-failed-element speed spikes. |
| function | `fit_crack_propagation_speed(history: CohesiveCrackHistory, *, start_position: float, end_position: float, minimum_samples: int = 3) -> CrackPropagationFit \| None` | Fit one representative speed over a fixed physical path interval. |
| function | `interface_front_history(time_values, path_coordinate, signal_frames, *, signal: str, threshold: float, fit_window: int = 5, direction: str = 'increasing') -> InterfaceFrontHistory` | Track a contiguous interface front from any increasing damage signal. |
| function | `cohesive_front_ensemble(trace: CohesiveInterfaceTrace, *, damage_thresholds = (0.5, 0.75, 0.95), opening_thresholds = (), dissipation_thresholds = (), fit_window: int = 5, direction: str = 'increasing') -> CohesiveFrontEnsemble` | Build observer-sensitivity evidence from a portable interface trace. |
| function | `compare_curve(reference_coordinate, reference_values, simulation_coordinate, simulation_values, *, coordinate_name: str = 'coordinate', quantity_name: str = 'value') -> ScientificComparison` | Interpolate a simulated curve onto observed coordinates and compare. |
| function | `compare_mach_cone(*, crack_speed: float, shear_wave_speed: float, observed_angle: float, unit: str = 'radian') -> ScientificComparison` | Compare an observed Mach angle with ``asin(c_s/v)``. |
| function | `compare_rectilinear_field(reference_x, reference_y, reference_values, simulation_x, simulation_y, simulation_values, *, quantity_name: str = 'field', reference_mask = None, simulation_mask = None) -> ScientificComparison` | Compare scalar maps after bilinear interpolation on their overlap. |
| function | `compare_rectilinear_observations(reference, simulation, *, quantity_name: str \| None = None) -> ScientificComparison` | Compare two portable rectilinear observations with semantic checks. |
| function | `mach_cone_angle(*, crack_speed: float, shear_wave_speed: float) -> float` | Return the ideal Mach angle ``asin(c_s / v)`` in radians. |
| function | `separation_regime(*, crack_speed: float, rayleigh_wave_speed: float, shear_wave_speed: float, failed_fraction: float, simultaneous_failed_fraction: float, spall_fraction: float = 0.8, rapid_failed_fraction: float \| None = None, ligament_traction_ratio: float \| None = None, pressure_wave_speed: float \| None = None) -> str` | Classify one frame with explicit crack-speed and spall evidence. |
| function | `estimate_stable_time_increment(*, characteristic_length, dilatational_speed: float, safety_factor: float = 0.8, interface_stiffness: float \| None = None, interface_area: float \| None = None, negative_mass: float \| None = None, positive_mass: float \| None = None) -> StableTimeIncrement` | Estimate explicit stability from body transit and interface oscillator. |
| function | `minimum_cell_nodal_spacing(domain) -> float` | Return an MPI-global conservative spacing from cell geometry nodes. |

## `agentfem.histories`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `FieldHistory` | A sampled scalar or finite-element field over physical time. |
| function | `field_history(source, **kwargs) -> FieldHistory` | Create a generic field-history recorder. |
| function | `temperature(source, *, name: str = 'temperature', unit: str = 'K', **kwargs) -> FieldHistory` | Create a physical-time temperature history. |

## `agentfem.interfaces`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CohesiveResponse` | One Mode-I traction--separation update. |
| class | `VectorCohesiveResponse` | Local-basis response of a two- or three-dimensional interface. |
| class | `MixedModeBilinearCohesiveLaw` | Bilinear mixed-mode cohesive law for proportional loading paths. |
| class | `BilinearCohesiveLaw` | Irreversible bilinear Mode-I cohesive law. |
| class | `CohesiveTransaction(law: BilinearCohesiveLaw, size: int)` | Trial/commit/rollback state for a batch of cohesive points. |
| class | `MixedModeCohesiveTransaction(law: MixedModeBilinearCohesiveLaw, size: int)` | Trial/commit state for :class:`MixedModeBilinearCohesiveLaw`. |
| class | `PairedLineFacets` | Deterministically paired zero-thickness line facets for a 2D mesh. |
| class | `PairedSurfaceFacets` | Deterministically paired zero-thickness triangular facets in 3D. |
| class | `SplitInterfaceMesh` | Array-level result of splitting one conforming interface manifold. |
| class | `NamedSplitInterfaceMesh` | One solver mesh carrying several disjoint named cohesive surfaces. |
| class | `InterfaceRigidModeAudit` | Rigid-body constraint rank of a split-interface model. |
| function | `audit_split_interface_rigid_modes(split: SplitInterfaceMesh \| NamedSplitInterfaceMesh, *, constrained_components, tangential = 'free', active_facets = None, rank_tolerance: float \| None = None, error_if_singular: bool = False) -> InterfaceRigidModeAudit` | Audit rigid translations and rotations before creating a solver mesh. |
| function | `create_dolfinx_split_mesh(split: SplitInterfaceMesh \| NamedSplitInterfaceMesh, *, comm = None, cell_type: str \| None = None, input_order: str = 'counterclockwise')` | Create an executable DOLFINx mesh for an audited split interface. |
| class | `CohesiveFacetResponse` | Trial force, kinematics and energy from paired interface facets. |
| class | `ModeIKinematicsAudit` | Accepted-state check that a declared Mode-I path remains Mode-I. |
| function | `audit_mode_i_kinematics(response: CohesiveFacetResponse, *, ratio_limit: float = 0.1, absolute_tolerance: float = 1e-12, error_if_exceeded: bool = False) -> ModeIKinematicsAudit` | Check tangential jump without changing cohesive history. |
| class | `CohesiveElementTangents` | Element-node layouts and consistent scalar-dof tangent matrices. |
| class | `ModeICohesiveFacetAssembler(topology: PairedLineFacets, law, *, number_of_nodes: int, thickness: float = 1.0, tangential: str = 'free', tangential_stiffness: float \| None = None)` | Two-point line integration for a fixed-path 2D interface. |
| class | `ModeICohesiveSurfaceAssembler(topology: PairedSurfaceFacets, law, *, number_of_nodes: int, tangential: str = 'free', tangential_stiffness: float \| None = None)` | Three-point integration of linear triangular interfaces in 3D. |
| function | `pair_coincident_surface_facets(coordinates, negative_facets, positive_facets, *, normal_hint, tolerance: float = 1e-10) -> PairedSurfaceFacets` | Pair coincident three-node triangular facets in 3D. |
| function | `pair_coincident_line_facets(coordinates, negative_facets, positive_facets, *, normal_hint, tolerance: float = 1e-10) -> PairedLineFacets` | Pair coincident two-node line facets with a declared normal direction. |
| function | `split_conforming_line_interface(coordinates, cells, interface_facets, *, positive_cells) -> SplitInterfaceMesh` | Duplicate nodes on a declared conforming 2D cell interface. |
| function | `split_conforming_surface_interface(coordinates, cells, interface_facets, *, positive_cells) -> SplitInterfaceMesh` | Duplicate nodes on a declared conforming triangular surface in 3D. |
| function | `split_conforming_named_interfaces(coordinates, cells, named_interfaces) -> NamedSplitInterfaceMesh` | Atomically split several disjoint conforming cohesive manifolds. |
| function | `split_conforming_cell_interface(coordinates, cells, *, positive_cells) -> SplitInterfaceMesh` | Split the internal facet separating two declared cell partitions. |
| class | `CohesiveSurface` | Public description of a fixed-path zero-thickness interface. |
| function | `bilinear_cohesive(*, strength: float, fracture_energy: float, initial_stiffness: float, compression_stiffness: float \| None = None, name: str = 'bilinear Mode-I cohesive law') -> BilinearCohesiveLaw` | Create a bilinear Mode-I cohesive law. |
| function | `mixed_mode_bilinear_cohesive(*, normal_strength: float, shear_strength: float, normal_fracture_energy: float, shear_fracture_energy: float, normal_stiffness: float, tangential_stiffness: float, interaction: str = 'bk', interaction_exponent: float = 1.45, compression_stiffness: float \| None = None, residual_tangential_fraction: float = 0.0, friction_coefficient: float = 0.0, friction_regularization: float = 1e-08, name: str = 'bilinear mixed-mode cohesive law') -> MixedModeBilinearCohesiveLaw` | Create a quadratic-initiation, energy-evolution mixed-mode law. |
| function | `cohesive_surface(*, law, mode: str = 'normal', name: str = 'cohesive surface') -> CohesiveSurface` | Declare a fixed-path zero-thickness cohesive interface. |
| function | `cohesive_characteristic_length(*, young: float, fracture_energy: float, strength: float) -> float` | Return the declared scale ``E * Gamma / strength**2``. |

## `agentfem.learning`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ConditionSpec` | A physical condition and the declared way it enters an objective. |
| class | `IntegrationEvidence` | Independent objective re-integration and refinement evidence. |
| class | `IntegrationPlan` | Training, held-out validation, and optional refinement integration. |
| class | `IntegrationRule` | One inspectable numerical-integration point set. |
| class | `NeuralFieldSpec` | Provider-neutral contract for PINN, DEM, XDEM, and related solvers. |
| class | `NeuralRepresentation` | How one neural function represents one or more unknown fields. |
| class | `ObjectiveTerm` | One named contribution to a neural-field optimization objective. |
| class | `SamplingPlan` | Inspectable coordinates or integration samples for one physical set. |
| class | `TrainableParameter` | A physical parameter inferred jointly with one or more fields. |
| function | `integration_consistency_check(plan: IntegrationPlan, *, training_value: float, validation_value: float, refinement_values = (), balance_error: float \| None = None, relative_tolerance: float = 0.05) -> IntegrationEvidence` | Compare optimized and held-out integration without trusting loss alone. |
| class | `NeuralFieldExecutionRequest` | Immutable input supplied to a user- or package-owned executor. |

## `agentfem.mechanics`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `CreepEnergyFrame` | Accepted work and energy evidence for one physical-time increment. |
| class | `CreepIncrementInfo` | Public AgentFEM object. |
| class | `CreepPathInfo` | Public AgentFEM object. |
| class | `ImplicitCreepStep` | Adaptive backward-Euler creep step with global Newton equilibrium. |
| function | `implicit_creep_step(*, displacement, material, duration: float, external_force, constraints = (), study = None, incrementation = None, solver_options = None, quadrature_degree: int = 2, creep_strain_error_tolerance: float \| None = None, time_unit: str \| None = None, progress = True, status_file = None, amplitude = None, temperature = None, name: str = 'implicit_creep', _experimental_distributed: bool = False) -> ImplicitCreepStep` | Build global 3D or axisymmetric implicit power-law creep. |
| class | `ExperimentalFiniteStrainPlasticityStep` | Total-Lagrangian Newton path consuming a neutral material provider. |
| class | `FiniteStrainJ2AffineTransaction` | Provider-owned trial/commit state for affine and distributed MPC Newton. |
| class | `FiniteStrainPlasticityIncrementInfo` | Public AgentFEM object. |
| function | `experimental_finite_strain_j2_step(*, displacement, material: FiniteStrainJ2Logarithmic, external_force = None, constraints = (), incrementation = None, solver_options = None, quadrature_degree: int = 2, amplitude = None, name: str = 'finite_strain_j2_experimental') -> ExperimentalFiniteStrainPlasticityStep` | Build the gated 3D global patch for logarithmic finite-strain J2. |
| function | `finite_strain_j2_affine_problem(*, displacement, material: FiniteStrainJ2Logarithmic \| QuadratureMaterialMap, constraint, external_force = None, incrementation = None, solver_options = None, quadrature_degree: int = 2, output_every: int \| None = 1, output_factors = (), progress = True, status_file = None, checkpoint_policy = None, name: str = 'finite_strain_j2')` | Build stateful finite-strain J2 under exact affine/MPC kinematics. |
| class | `J2IncrementInfo` | Public AgentFEM object. |
| class | `J2LoadPathInfo` | Public AgentFEM object. |
| class | `J2PlasticityStep` | Incremental global equilibrium for 3D small-strain J2 plasticity. |
| function | `j2_plasticity_step(*, displacement, material, external_force, constraints = (), study = None, incrementation = None, solver_options = None, quadrature_degree: int = 2, progress = True, status_file = None, amplitude = None, name: str = 'j2_plasticity', _experimental_distributed: bool = False) -> J2PlasticityStep` | Build a global 3D or axisymmetric J2 step. |

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
| function | `force_vector(target, loads = None, *, load = None, study = None) -> OperatorForm` | Create a total force/source vector from one or more load objects. |
| function | `form_arity(expression) -> int \| None` | Return the number of UFL arguments, or ``None`` for opaque backends. |
| function | `flux_vector(flux, target, *, measure = None, location = None) -> OperatorForm` | Create a prescribed scalar boundary-flux vector. |
| function | `heat_capacity_operator(temperature, capacity, *, measure = ufl.dx) -> OperatorForm` | Create a heat-capacity operator ``C`` for transient heat problems. |
| function | `heat_capacity_vector(previous_temperature, temperature, capacity, *, measure = ufl.dx) -> OperatorForm` | Create the known heat-capacity vector ``C * T_previous``. |
| function | `heat_conduction_operator(temperature, conductivity, *, measure = ufl.dx) -> OperatorForm` | Create a heat-conduction operator ``K`` for ``-div(k grad(T))``. |
| function | `heat_source_vector(source, temperature, *, measure = ufl.dx) -> OperatorForm` | Create a heat-source vector ``Q`` for a temperature unknown. |
| function | `inertial_force_vector(acceleration, target, density = 1.0, *, measure = ufl.dx) -> OperatorForm` | Create the inertial virtual-work vector ``F_inertia = M a``. |
| function | `load_vector(target, loads = None, *, load = None, study = None) -> OperatorForm` | Create a total external-load vector ``F`` for a target unknown. |
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
| function | `stiffness(field, properties = None, *, law = None, study = None, temperature = None, measure = ufl.dx) -> OperatorForm` | Create the primary stiffness-like operator ``K`` for an unknown field. |
| function | `quadratic_form(operator, field) -> float` | Return the algebraic scalar ``field^T operator field``. |
| function | `xtmx(field, operator) -> float` | Cast3M-style alias for ``field^T operator field``. |
| function | `xtmy(left, operator, right) -> float` | Cast3M-style alias for ``left^T operator right``. |
| function | `elastic_stiffness(displacement, properties, *, study = None, temperature = None, measure = ufl.dx) -> OperatorForm` | Create an elastic stiffness operator ``K`` from a displacement unknown. |
| function | `internal_force_vector(displacement, test_function = None, properties = None, *, study = None, measure = ufl.dx) -> OperatorForm` | Create an elastic internal-force vector contribution. |
| function | `stiffness_operator(displacement, test_function = None, properties = None, *, study = None, temperature = None, measure = ufl.dx) -> OperatorForm` | Create an elastic stiffness/internal virtual-work operator ``K``. |
| function | `thermal_expansion_vector(target, temperature, properties, *, study = None, measure = ufl.dx, name: str = 'F_thermal') -> OperatorForm` | Equivalent nodal load produced by isotropic thermal expansion. |
| function | `convective_momentum_operator(advecting_velocity, transported_velocity, test_velocity, *, measure = ufl.dx, name: str = 'N_convection') -> OperatorForm` | Return ``((w . grad) u, v)`` for vector momentum transport. |
| function | `incompressibility_operator(velocity, test_pressure, *, measure = ufl.dx, name: str = 'D_incompressibility') -> OperatorForm` | Return the symmetric saddle-point term ``-(q, div(u))``. |
| function | `pressure_coupling_operator(pressure, test_velocity, *, measure = ufl.dx, name: str = 'G_pressure') -> OperatorForm` | Return the pressure contribution ``-(p, div(v))``. |
| function | `viscous_flow_operator(velocity, test_velocity, viscosity, *, measure = ufl.dx, name: str = 'K_viscous') -> OperatorForm` | Return ``nu (grad(u), grad(v))`` for incompressible momentum. |
| function | `auxiliary_laplacian_boundary(boundary_expression)` | Return ``-Delta(g)`` for the auxiliary field ``w=-Delta(u)``. |
| function | `split_laplacian_operator(trial, test, *, measure = ufl.dx, name: str = 'K_split_laplacian') -> OperatorForm` | Return one second-order block of a mixed biharmonic split. |
| class | `FirstOrderSystem` | First-order transient system, ``C x_dot + K x = F``. |
| class | `LinearSystem` | Engineering-level static system, usually ``K x = F``. |
| class | `SecondOrderSystem` | Engineering-level second-order system, ``M a + C v + K u = F``. |
| function | `first_order_system(C, K, F = None, *, name: str = 'Cxdot_plus_Kx_eq_F')` | Create ``C x_dot + K x = F`` for heat/diffusion-like evolution. |
| function | `linear_system(K, F = None, *, name: str = 'Kx_eq_F') -> LinearSystem` | Create a static linear system in engineering notation, ``K x = F``. |
| function | `second_order_system(M, K, C = None, F = None, *, name: str = 'Ma_plus_Cv_plus_Ku_eq_F')` | Create ``M a + C v + K u = F`` with optional damping and force. |
| function | `advection_operator(trial, test, velocity, *, measure = ufl.dx, name: str = 'A_advection') -> OperatorForm` | Return the Galerkin advection operator ``(v . grad(u), w)``. |
| function | `as_velocity(velocity)` | Normalize a public velocity sequence without hiding UFL expressions. |
| function | `burgers_convection_operator(advecting_scalar, transported_scalar, test, *, direction = None, measure = ufl.dx, name: str = 'N_burgers') -> OperatorForm` | Return scalar Burgers transport ``u_adv (d . grad(u))``. |
| function | `intrinsic_time_scale(domain, velocity)` | Return the standard cellwise advective SUPG scale ``h/(2 \|v\|)``. |
| function | `reaction_expression(value, law: str \| Mapping[str, object], **parameters)` | Lower a named scalar reaction law to a UFL expression. |
| function | `streamline_upwind_operator(strong_residual, test, velocity, *, tau = None, domain = None, measure = ufl.dx, name: str = 'A_supg') -> OperatorForm` | Return a SUPG contribution ``tau R(u) (v . grad(w))``. |

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
| function | `cyclic_fatigue() -> SolutionProcedure` | Quasi-static peak/valley equilibrium with independent cycle blocks. |
| function | `for_step(*, analysis: str, method: str \| None = None, stateful: bool = False)` | Resolve a procedure without coupling ``Study`` to one solver route. |
| function | `resolve(*, analysis: str, requested: SolutionProcedure \| str \| None = None, preferred: str \| None = None, stateful: bool = False) -> SolutionProcedure` | Resolve and validate the numerical procedure for one analysis request. |

## `agentfem.responses`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ResponseReport` | A finite-difference Jacobian and the cases that support it. |
| class | `FiniteDifferenceResponse` | A method-neutral response contract with a finite-difference provider. |
| function | `finite_difference(**kwargs) -> FiniteDifferenceResponse` | Create a campaign-backed finite-difference response experiment. |

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
| class | `PreparedLinearProblem(bilinear_form, linear_form, solution, *, bcs = None, options: LinearSolverOptions \| None = None)` | A linear problem whose constant matrix and KSP are assembled once. |
| function | `prepare_linear_problem(bilinear_form, linear_form, solution, *, bcs = None, options: LinearSolverOptions \| None = None) -> PreparedLinearProblem` | Prepare one constant linear operator for repeated right-hand sides. |
| function | `solve_matrix_system(A, b, x, options: LinearSolverOptions \| None = None, *, raise_on_failure: bool \| None = None) -> LinearSolveInfo` | Solve ``A x = b`` and return explicit PETSc convergence evidence. |
| function | `solve_linear_problem(bilinear_form, linear_form, solution, *, bcs = None, options: LinearSolverOptions \| None = None, return_info: bool = False)` | Assemble and solve a standard linear variational problem. |
| function | `solve_nonlinear_problem(residual_form, solution, *, bcs = None, jacobian_form = None, options: NonlinearSolverOptions \| NewtonSolverOptions \| None = None, petsc_options_prefix: str = 'agentfem_nonlinear_') -> tuple[object, NonlinearSolveInfo]` | Solve ``R(u; v) = 0`` with the current DOLFINx PETSc/SNES interface. |
| function | `solve_affine_nonlinear_path(residual_form, jacobian_form, solution, constraint, *, load_factors = None, incrementation = None, output_factors = (), options: AffineNewtonOptions \| NewtonSolverOptions \| None = None, on_increment = None, on_accepted_boundary = None, on_acceptance_failure = None, acceptance_check = None, state_transaction = None, stop_factor: float = 1.0, accepted_history = (), attempted_history = (), next_increment_size: float \| None = None, reporter = None, step_name: str = 'affine_nonlinear', step_number: int = 1) -> tuple[object, AffineLoadPathInfo]` | Solve a nonlinear path under ``u = T q + u_bar`` constraints. |

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
| class | `AffineCoordinateMap` | Explicit affine map from observation coordinates to model coordinates. |
| class | `FieldEncoding` | How a physical field becomes a machine-learning tensor. |
| class | `NeuralOperatorSpec` | Function-to-function learning contract for an external trainer. |
| class | `ObservationGrid` | Mesh-independent Cartesian coordinates for field learning and sensing. |
| class | `PhysicsCondition` | Boundary, initial, interface, or observation condition in a loss. |
| class | `PhysicsResidual` | One explicit differentiable residual used in a physics loss. |
| class | `PINNSpec` | Physics-informed training contract for selected explicit residuals. |
| function | `regular_grid(*, bounds, shape, axis_names = None, coordinate_system: str = 'cartesian', order: str = 'C', coordinate_unit: str \| None = None) -> ObservationGrid` | Create an evenly spaced observation grid from physical bounds. |
| class | `TorchMLPSurrogate` | Configurable dense-network baseline for parameter-to-QoI learning. |
| class | `TrainedTorchMLP` | In-memory trained PyTorch adapter. |
| class | `PINNTrainingRecord` | In-memory training evidence without serializing a PyTorch pickle. |
| class | `TorchPINNAdapter` | Bind explicit residual/condition callables to a :class:`PINNSpec`. |
| class | `SurrogateTrainingRun` | A trained model together with its independent validation evidence. |
| function | `train(dataset: ScientificDataset, *, estimator = None, validation_fraction: float = 0.2, seed: int = 0, thresholds = None) -> SurrogateTrainingRun` | Split, fit, and independently validate one surrogate estimator. |

## `agentfem.assembly`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `make_form(ufl_form)` | Compile a UFL form for assembly. |
| function | `assemble_vector(form)` | Assemble a vector and accumulate ghost contributions to owned entries. |
| function | `assemble_matrix(form, bcs = None)` | Assemble a matrix and apply optional strong Dirichlet BC structure. |
| function | `assemble_lumped_operator(V, coefficient = 1.0, measure = ufl.dx) -> np.ndarray` | Assemble a diagonal/lumped operator vector on ``V``. |
| function | `assemble_lumped_mass(V, density = 1.0, measure = ufl.dx) -> np.ndarray` | Assemble a lumped mass vector for a scalar or vector space. |
| function | `inverse_diagonal(diagonal: np.ndarray) -> np.ndarray` | Return a safe inverse for a diagonal vector. |

## `agentfem.backends`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `BackendAdapter` | Minimal interface used by operator compilation and assembly. |
| class | `BackendDescriptor` | Inspectable backend identity and capability statement. |
| class | `FEniCSxBackend` | Current production backend for AgentFEM operator forms. |
| function | `available_backends() -> tuple[str, ...]` | Return registered backend names without importing their dependencies. |
| function | `backend_descriptors() -> tuple[BackendDescriptor, ...]` | Return descriptors for all registered backends. |
| function | `default_backend_name() -> str` | Public AgentFEM object. |
| function | `get_backend(name: str \| None = None) -> BackendAdapter` | Return a lazily constructed backend adapter. |
| function | `register_backend(name: str, factory: BackendFactory, *, overwrite: bool = False) -> None` | Register a lazy backend factory. |
| function | `set_default_backend(name: str) -> None` | Select the process-local default backend by registered name. |

## `agentfem.benchmarks`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `BenchmarkSpec` | One verification obligation and its executable evidence. |
| function | `benchmark(identifier: str) -> BenchmarkSpec` | Return one benchmark by stable identifier. |
| function | `list_benchmarks(*, capability: str \| None = None) -> tuple[BenchmarkSpec, ...]` | Return all benchmarks or those for one capability. |
| class | `CapabilityEvidence` | Evidence supporting one declared constitutive maturity boundary. |
| function | `audit_capability_evidence() -> tuple[CapabilityEvidence, ...]` | Return a stable, machine-readable audit for the whole catalog. |
| function | `capability_evidence(capability: ConstitutiveCapability, *, benchmarks: tuple[BenchmarkSpec, ...] \| None = None) -> CapabilityEvidence` | Audit one catalog capability against the benchmark registry. |
| class | `GoldenBenchmark` | A named collection of numerical observables from a benchmark card. |
| class | `GoldenQuantity` | One expected physical observable with explicit numerical tolerances. |
| function | `golden_benchmark(identifier: str) -> GoldenBenchmark` | Load a numerical contract by stable benchmark-card identifier. |
| class | `DelaminationBenchmarkAssessment` | Acceptance evidence for one structural cohesive benchmark. |
| class | `DelaminationBenchmarkSpec` | Geometry and evidence contract for DCB, ENF or MMB verification. |
| class | `DelaminationEnergyReleaseCurve` | Compliance-derived structural GI/GII evidence versus crack length. |
| class | `MixedModeBendingComparison` | Curve-level errors under explicitly declared scientific tolerances. |
| class | `MixedModeBendingCurve` | One traceable load/displacement/mode-mix curve versus crack length. |
| function | `assess_delamination_benchmark(spec, predicted, reference, *, energy_release_relative_tolerance, minimum_process_zone_elements, required_process_zone_elements = 3.0, artificial_dissipation = 0.0, internal_energy = 1.0) -> DelaminationBenchmarkAssessment` | Apply curve, cohesive-zone resolution and dissipation guardrails. |
| function | `beam_theory_energy_release_curve(spec, *, crack_length, load)` | Return a DCB/ENF analytical oracle through the same public contract. |
| function | `compliance_energy_release_curve(spec: DelaminationBenchmarkSpec, *, crack_length, load, displacement = None, compliance = None, mode_i_fraction = None, source: str \| None = None) -> DelaminationEnergyReleaseCurve` | Recover structural energy release by the compliance derivative. |
| function | `compare_mixed_mode_bending_curves(reference: MixedModeBendingCurve, predicted: MixedModeBendingCurve, *, load_relative_tolerance: float, displacement_relative_tolerance: float, mode_i_fraction_absolute_tolerance: float) -> MixedModeBendingComparison` | Compare a computed curve on the reference crack-length coordinates. |
| function | `dcb_beam_compliance(spec, crack_length)` | Euler--Bernoulli DCB compliance for two arms of thickness ``h``. |
| function | `delamination_benchmark_spec(kind, **geometry) -> DelaminationBenchmarkSpec` | Create a DCB, ENF or MMB numerical-verification specification. |
| function | `enf_beam_compliance(spec, crack_length)` | Classical simple-beam ENF compliance with support half-span ``L``. |
| class | `CohesiveEnergyBenchmark` | Energy closure for one uniformly separating cohesive interface. |
| class | `ClassicalCrackBenchmark` | Fixed-path Mode-I crack propagation evidence for the V3 guardrail. |
| class | `ThinThreeDimensionalCrossCheck` | Plane-stress condensation versus an affine thin-3D FEM patch. |
| class | `WaveArrivalBenchmark` | Measured and acoustic-tensor wave speed in reference coordinates. |
| class | `WeakInterfaceConvergenceStudy` | Two-dimensional mesh and time-step evidence for one V4 mechanism. |
| class | `WeakInterfaceTransitionBenchmark` | One prestressed thin-sheet case in the JMPS V4 mechanism ladder. |
| class | `WeakInterfaceTransitionSuite` | Auditable crack-like to supershear to spall-like V4 mechanism gate. |
| function | `cohesive_energy_balance(*, dt: float = 0.001, loading_time: float = 0.2, opening: float = 0.08) -> CohesiveEnergyBenchmark` | Open one split interface through a smooth prescribed-motion history. |
| function | `classical_cohesive_crack(*, cells: int = 60, length: float = 3.0, precrack_length: float = 0.5, opening: float = 0.0135, loading_time: float = 0.15, hold_time: float = 0.15, time_step_scale: float = 0.8, damping: float = 0.0) -> ClassicalCrackBenchmark` | Propagate a precracked cohesive strip below the classical limit. |
| function | `finite_strain_wave_arrival(*, prestrain: float = 0.0, cells: int = 80, courant: float = 0.3, length: float = 2.0, source_position: float = 0.25, receiver_positions = (0.75, 1.25), pulse_width: float = 0.1) -> WaveArrivalBenchmark` | Measure a small longitudinal pulse about a held homogeneous stretch. |
| function | `jmps_weak_interface_transition_v4(*, cells: int = 30, total_time: float = 0.1, history_every: int = 5) -> WeakInterfaceTransitionSuite` | Run the first fixed, executable JMPS-inspired V4 mechanism ladder. |
| function | `jmps_weak_interface_convergence_v4(*, history_every: int = 20, spatial_speed_tolerance: float = 0.1, temporal_speed_tolerance: float = 0.02) -> WeakInterfaceConvergenceStudy` | Run the opt-in two-dimensional V4 supershear convergence contract. |
| function | `plane_stress_thin_3d_crosscheck(*, axial_stretch: float = 1.12, reference_thickness: float = 0.02, cells = (2, 2, 1), young: float = 1000000.0, poisson: float = 0.49, density: float = 1000.0, tolerance: float = 1e-09) -> ThinThreeDimensionalCrossCheck` | Compare condensed 2D membrane response with a thin 3D FEM patch. |
| function | `prestressed_weak_interface_separation(*, label: str = 'v4_candidate', cells: int = 60, transverse_cells: int = 2, length: float = 3.0, height: float = 1.0, precrack_length: float = 0.5, axial_strain: float = 0.12, strength: float = 10.0, fracture_energy: float = 0.1, initial_stiffness: float = 10000.0, young: float = 1000.0, poisson: float = 0.49, density: float = 1.0, total_time: float = 0.2, time_step_scale: float = 0.8, damping: float = 0.0, history_every: int = 1, impact_displacement: float = 0.0, impact_rise_time: float \| None = None, speed_fit_length: float \| None = None, bulk_material = None, retain_trace: bool = False) -> WeakInterfaceTransitionBenchmark` | Drive a precrack through a prestressed plane-stress weak interface. |
| function | `creep_thick_cylinder_benchmark(*, comm = MPI.COMM_WORLD, radial_cells: int = 4, angular_cells: int = 8, axial_cells: int = 1, increments: int = 300, duration: float = 1000.0, creep_strain_error_tolerance: float = 0.0005, progress: object = False, formulation: str = 'axisymmetric') -> InelasticStructuralBenchmark` | Run the NAFEMS R0027 Test 7 secondary-creep benchmark. |
| class | `InelasticStructuralBenchmark` | Compact, rank-independent evidence from one structural benchmark. |
| function | `j2_plane_strain_first_yield_pressure(*, inner_radius: float, outer_radius: float, poisson: float, yield_stress: float) -> float` | Lamé plane-strain pressure at first Mises yield on the inner wall. |
| function | `j2_thick_cylinder_benchmark(*, comm = MPI.COMM_WORLD, radial_cells: int = 4, angular_cells: int = 8, axial_cells: int = 1, increments: int = 24, formulation: str = 'three_dimensional_sector') -> InelasticStructuralBenchmark` | Run the Comet-FEniCSx thick-cylinder first-yield benchmark. |
| function | `power_law_creep_cylinder_stress(radius, *, inner_radius: float, outer_radius: float, pressure: float, stress_exponent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Return the NAFEMS R0027 Test 7 steady-state cylinder stresses. |
| function | `thick_cylinder_sector_mesh(*, inner_radius: float, outer_radius: float, thickness: float, radial_cells: int, angular_cells: int, cell_type: str = 'tetrahedron', comm = MPI.COMM_WORLD)` | Create a one-layer 3D quarter-cylinder benchmark mesh. |
| class | `CenterCrackLEFMBenchmark` | One solved center-crack model and its independently extracted evidence. |
| function | `center_crack_lefm_mesh(*, half_crack_length: float = 1.0, half_width: float = 8.0, half_height: float = 8.0, comm = MPI.COMM_SELF)` | Build the serial, conforming split mesh used by the LEFM benchmark. |
| function | `center_crack_mode_i_benchmark(*, young_modulus: float = 1000.0, poisson_ratio: float = 0.25, half_crack_length: float = 1.0, half_width: float = 8.0, half_height: float = 8.0, remote_strain: float = 0.001, relative_tolerance: float = 0.05) -> CenterCrackLEFMBenchmark` | Solve and verify a finite-plate Mode-I crack with the public workflow. |

## `agentfem.dependencies`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `OptionalDependencyError(*, package: str, extra: str, capability: str)` | Raised when a requested optional capability is not installed. |
| class | `DependencyStatus` | Inspectable availability record for one optional integration. |
| function | `require(package: str, *, extra: str, capability: str)` | Import an optional package or raise an installation-specific error. |
| function | `status(package: str, *, extra: str, capability: str) -> DependencyStatus` | Return package availability without importing compiled extensions. |

## `agentfem.diagnostics`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `PerformanceLedger` | Low-overhead, rank-local timing evidence for one solver lifecycle. |
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
| class | `StateDependentThermalBalanceMonitor` | Heat ledger for nonlinear conductivity and heat-capacity models. |
| class | `ThermalContentMonitor` | Backwards-compatible sensible-heat monitor without balance terms. |
| function | `max_abs(function: fem.Function) -> float` | Global max absolute value of a finite-element field. |
| function | `max_magnitude(function) -> float` | Global maximum magnitude of a scalar or vector finite-element field. |
| class | `FieldStats` | Distributed scalar statistics for a finite-element field. |
| class | `ScalarDiagnostic` | Named scalar diagnostic evaluated on demand. |
| class | `DiagnosticSet` | Ordered collection of scalar diagnostics. |
| function | `magnitude_stats(function, *, on = None, name: str \| None = None) -> FieldStats` | Return distributed magnitude statistics for a scalar or vector field. |
| function | `field_stats(function, *, on = None, name: str \| None = None) -> FieldStats` | Alias for ``magnitude_stats`` for application-level diagnostics. |

## `agentfem.elements`

This package exposes its public objects through focused submodules.

## `agentfem.extensions`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ExtensionError` | An installed extension could not be discovered or activated safely. |
| class | `ExtensionSpec` | Identity and compatibility contract published by one extension. |
| class | `Extension` | One loadable extension and its side-effect-free registration callback. |
| class | `ExtensionDescriptor` | Package metadata visible without importing extension code. |
| class | `ExtensionContext` | Staging area exposed to an extension during activation. |
| class | `LoadedExtension` | Activated identity and the capabilities registered into this process. |
| function | `discover_extensions() -> tuple[ExtensionDescriptor, ...]` | Return installed extension metadata without importing extension code. |
| function | `extension_status() -> dict[str, object]` | Return the machine-facing installed and activated extension inventory. |
| function | `loaded_extensions() -> tuple[LoadedExtension, ...]` | Return activated extensions in stable name order. |
| function | `missing_extensions(names) -> tuple[str, ...]` | Return required names that are not advertised by installed packages. |
| function | `load_extension(name: str) -> LoadedExtension` | Explicitly import, validate, and activate one installed extension. |
| function | `load_extensions(names) -> tuple[LoadedExtension, ...]` | Activate required extensions in declaration order. |

## `agentfem.forms`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `stiffness_form(stress, strain_test, measure = ufl.dx)` | Internal stiffness/virtual-work form, ``sigma : epsilon(test)``. |
| function | `mass_form(density, trial_function, test_function, measure = ufl.dx)` | Consistent mass form, ``rho * trial . test``. |
| function | `damping_form(coefficient, trial_function, test_function, measure = ufl.dx)` | Viscous damping form, ``c * trial . test``. |
| function | `diffusion_form(conductivity, trial_function, test_function, measure = ufl.dx)` | Diffusion/conduction form, ``k * grad(trial) . grad(test)``. |
| function | `inertial_form(density, acceleration, test_function, measure = ufl.dx)` | Inertial virtual-work form, ``rho * acceleration . test``. |
| function | `body_load_form(force, test_function, measure = ufl.dx)` | Body-force/source virtual-work form, ``force . test``. |
| function | `boundary_load_form(load, test_function, measure)` | Boundary flux/traction virtual-work form, ``load . test``. |
| function | `scalar_flux_form(flux, test_function, measure)` | Scalar flux weak form, ``flux * test`` on a boundary or domain measure. |
| function | `robin_form(coefficient, trial_function, test_function, measure)` | Robin/impedance bilinear form, ``coefficient * trial * test``. |
| function | `internal_virtual_work(stress, strain_test)` | Compatibility wrapper for ``stiffness_form``. |
| function | `inertial_virtual_work(density, acceleration, test_function)` | Compatibility wrapper for ``inertial_form``. |
| function | `body_force_virtual_work(force, test_function, measure = ufl.dx)` | Compatibility wrapper for ``body_load_form``. |
| function | `boundary_flux_virtual_work(flux, test_function, ds_measure)` | Compatibility wrapper for ``boundary_load_form``. |

## `agentfem.io`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `ensure_output_dir(path: Path, comm: MPI.Comm) -> None` | Create an output directory once, then synchronize all ranks. |
| class | `CSVLogger` | Rank-zero CSV writer for time histories and scalar diagnostics. |
| class | `XDMFTimeSeries(path: Path, domain, mode: str = 'w') -> None` | Small context manager for writing a mesh and time-dependent fields. |
| class | `ParaViewTimeSeries(path: Path, domain, mode: str = 'w') -> None` | Collective VTK/PVD series with one geometry carrying all fields. |
| class | `ResultWriter(path: Path, domain, fields = (), mode: str = 'w') -> None` | Named result writer for one mesh and a stable field list. |
| function | `interpolate_for_xdmf(field, *, degree: int = 1, name: str \| None = None)` | Interpolate a field to an XDMF-friendly Lagrange output space. |

## `agentfem.integrations`

This package exposes its public objects through focused submodules.

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

## `agentfem.platforms`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `PlatformSupport` | One operating-system support decision with explicit limitations. |
| class | `RuntimeReport` | Compact runtime inventory for bug reports and agent inspection. |
| function | `support_for(system: str, *, wsl: bool = False, wsl_version: int \| None = None) -> PlatformSupport` | Return the first-release support tier for an operating-system route. |
| function | `current_support() -> PlatformSupport` | Detect the current OS, including Windows Subsystem for Linux. |
| function | `runtime_report() -> RuntimeReport` | Return versions and optional integrations useful in issue reports. |

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
| function | `affine_nonlinear(residual, solution, *, jacobian, constraint, load_factors = None, incrementation = None, solver_options: AffineNewtonOptions \| NewtonSolverOptions \| None = None, output_every: int \| None = 1, output_factors = (), state_transaction = None, checkpoint_policy = None, acceptance_check = None, progress = True, status_file = None, name: str = 'affine_nonlinear', procedure = None) -> AffineNonlinearVariationalProblem` | Create a nonlinear problem reduced by an affine constraint map. |
| class | `LoadIncrementSnapshot` | A copied solution state at one nonlinear load factor. |
| function | `first_order_transient(*, capacity, stiffness, history, source = None, dt: float, study = None, unknown = None, solution = None, constraints = None, bcs = None, solver_options: LinearSolverOptions \| None = None, name: str = 'first_order_transient_step', method: str = 'implicit_euler') -> AnalysisStep` | Create a first-order transient step. |
| function | `first_order_transient_run(*, capacity, stiffness, history, current, previous, dt: float, steps: int, source = None, study = None, constraints = None, bcs = None, solver_options: LinearSolverOptions \| None = None, update_load = None, save_every: int \| None = None, print_every: int \| None = None, progress = True, status_file = None, checkpoint_policy = None, name: str = 'first_order_transient') -> FirstOrderTransientStep` | Create an executable implicit-Euler time step and loop. |
| function | `nonlinear_first_order_transient_run(*, residual, jacobian, current, previous, dt: float, steps: int, study = None, constraints = None, bcs = None, solver_options: NonlinearSolverOptions \| NewtonSolverOptions \| None = None, update_load = None, save_every: int \| None = None, print_every: int \| None = None, progress = True, status_file = None, checkpoint_policy = None, history_monitor = None, name: str = 'nonlinear_first_order_transient', petsc_options_prefix: str = 'agentfem_nonlinear_transient_') -> FirstOrderTransientStep` | Create a nonlinear implicit-Euler step with the shared lifecycle. |
| function | `explicit_dynamics(*, state, integrator, residual, stiffness = None, dt: float, steps: int, study = None, prescribed = (), constraints = (), update_load = None, save_every: int \| None = None, print_every: int \| None = None, history_every: int = 1, progress = True, status_file = None, checkpoint_policy = None, history_monitor = None, stability = None, name: str = 'explicit_dynamics') -> ExplicitDynamicsStep` | Create a second-order explicit dynamics step. |
| function | `implicit_dynamics(*, state, mass, stiffness, force, damping = None, dt: float, steps: int, parameters = None, study = None, constraints = (), bcs = None, solver_options: LinearSolverOptions \| None = None, update_load = None, progress = True, status_file = None, checkpoint_policy = None, save_every: int \| None = None, print_every: int \| None = None, name: str = 'implicit_dynamics') -> ImplicitDynamicsStep` | Create a linear Newmark or generalized-alpha dynamics step. |

## `agentfem.provenance`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `content_fingerprint(record: object) -> str` | Return a canonical content identity for one JSON-safe scientific record. |
| function | `scientific_input_manifest(value: object, *, label: str = 'scientific_inputs', require_nonempty: bool = False) -> dict[str, object]` | Describe and fingerprint scientific inputs without hiding opaque parts. |
| function | `seal_manifest(manifest: Mapping[str, object], *, base: str \| Path, producer_version: str) -> dict[str, object]` | Return a deterministic integrity seal for an unsealed manifest. |
| function | `runtime_manifest() -> dict[str, object]` | Capture runtime evidence and a stable compatibility identity. |
| function | `freeze_runtime(path: str \| Path) -> Path` | Atomically write the current runtime lock for a frozen campaign. |
| class | `RuntimeComparison` | Compatibility decision between a frozen and current runtime. |
| function | `compare_runtime(expected: str \| Path \| Mapping[str, object], *, actual: Mapping[str, object] \| None = None) -> RuntimeComparison` | Compare a stored runtime identity with the current or supplied one. |
| function | `require_runtime(expected: str \| Path \| Mapping[str, object], *, policy: str = 'error') -> RuntimeComparison` | Enforce or warn about a frozen runtime before a scientific campaign. |
| class | `SealVerification` | Outcome of checking one stored provenance seal. |
| function | `verify_manifest(path: str \| Path) -> SealVerification` | Verify a result manifest and every artifact recorded in its seal. |

## `agentfem.spaces`

| Kind | Public object | Purpose |
| --- | --- | --- |
| function | `lagrange_space(domain, degree: int = 1)` | Create a scalar Lagrange function space. |
| function | `scalar_space(domain, degree: int = 1)` | Create a scalar Lagrange function space. |
| function | `vector_lagrange_space(domain, degree: int = 1, dim: int \| None = None)` | Create a vector Lagrange function space. |
| function | `vector_space(domain, degree: int = 1, dim: int \| None = None)` | Create a vector Lagrange function space. |
| function | `velocity_pressure_space(domain, *, velocity_degree: int = 2, pressure_degree: int = 1)` | Create a Taylor--Hood velocity/pressure mixed space. |
| function | `displacement_pressure_space(domain, *, displacement_degree: int = 2, pressure_degree: int = 0)` | Create the mixed ``H1`` displacement / discontinuous-pressure space. |
| function | `test_function(V)` | Create a UFL test function for a function space. |
| function | `trial_function(V)` | Create a UFL trial function for a function space. |
| function | `named_function(V, name: str, value = 0.0)` | Create a named finite-element function and optionally initialize it. |

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

## `agentfem.upgrades`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `UpgradeFinding` | One stable, addressable compatibility or migration finding. |
| class | `UpgradeReport` | Dry-run migration plan for one installed-use project. |
| function | `inspect_project(project: ProjectConfig) -> UpgradeReport` | Return a dry-run upgrade report without executing or changing the case. |
| function | `apply_safe_metadata(project: ProjectConfig) -> tuple[Path, ...]` | Apply only deterministic project-metadata migrations, atomically. |
| function | `migrate_cohesive_checkpoint(snapshot: dict[str, object], *, tangential: str, tangential_stiffness: float \| None = None, acknowledge_physics_change: bool = False) -> dict[str, object]` | Explicitly promote a physical-keyed scalar checkpoint to schema v5. |

## `agentfem.validation`

| Kind | Public object | Purpose |
| --- | --- | --- |
| class | `ValidationIssue` | One addressable model, numerical, or execution issue. |
| class | `ValidationReport` | Immutable collection of structured validation issues. |
| class | `ModelValidationError(report: ValidationReport)` | Raised when a structured model validation report contains errors. |
| function | `issue(code: str, path: str, message: str, *, severity: Severity = 'error', hint: str \| None = None, **context) -> ValidationIssue` | Concise constructor used by model validators and backend adapters. |
