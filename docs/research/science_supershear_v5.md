# Science Supershear V5 Research Protocol

This protocol hands publication comparison to a research analyst while keeping
the reusable solver and evidence contracts in AgentFEM. It targets Wang, Shi,
and Fineberg, *Science* 381 (2023), and uses the authors' public Dryad dataset.
The JMPS 2025 model remains a complementary mechanism reference because its
complete computational input deck is not public.

## What the software now provides

AgentFEM pins Dryad version `235603` as a 26-file CC0 manifest. Each file has
an official size and SHA-256 identity. The base installation can inspect XLSX
values and cached formula results without adding pandas or openpyxl.

The dynamic-fracture route can optionally retain a compact interface trace:

```python
case = benchmarks.prestressed_weak_interface_separation(
    ...,
    retain_trace=True,
)
case.trace.write("interface_trace.npz")

fronts = fracture.cohesive_front_ensemble(
    case.trace,
    damage_thresholds=(0.50, 0.75, 0.95),
    opening_thresholds=(opening_threshold,),
    dissipation_thresholds=(energy_threshold,),
    fit_window=fit_window,
)
```

The trace contains accepted frame times and facet-center opening, traction,
damage, and dissipated-energy density. It records whether a quantity is a
quadrature mean or maximum. Global cohesive energy remains a separately
integrated history channel; density and integrated energy are not conflated.

Bulk energy maps can be refreshed by the existing Explicit writer without a
private research time loop:

```python
live = results.finite_strain_dynamic_cell_fields(
    step.state.u,
    step.state.v,
    material,
    variables=("SENER", "KED", "J"),
)
step.run(output="fields.xdmf", fields=(step.state.u, step.state.v, live))
```

Common comparisons use:

```python
speed_fit = fracture.compare_curve(x_exp, speed_exp, x_fem, speed_fem)
cone_fit = fracture.compare_mach_cone(
    crack_speed=v,
    shear_wave_speed=cs,
    observed_angle=angle,
    unit="degree",
)
field_fit = fracture.compare_rectilinear_field(
    x_exp, y_exp, sed_exp,
    x_fem, y_fem, sed_fem,
    quantity_name="SED",
)
```

These functions report sample count, RMSE, range-normalized RMSE, correlation
when defined, overlap, and interpolation convention. They are evidence tools,
not acceptance thresholds chosen by the software.

Publication panels and FEM meshes need not share an origin, axis orientation,
or length scale. The registration is now an explicit scientific object rather
than a plotting transform:

```python
registration = surrogates.AffineCoordinateMap(
    matrix=observation_to_model_axes,
    offset=observation_to_model_origin,
    source_coordinate_system="publication_panel",
    target_coordinate_system="reference_mesh",
    source_unit="mm",
    target_unit="m",
)
# observation_grid records coordinate_unit="mm"; the matrix carries the
# reviewed mm-to-m scale and any axis/origin registration.
sample = datasets.fem_observation_sample(
    SED,
    observation_grid,
    coordinate_map=registration,
    configuration="reference",
    outside="mask",
)
fem_map = datasets.RectilinearObservation.from_field_sample(sample)
field_fit = fracture.compare_rectilinear_observations(experiment_map, fem_map)
```

The comparison rejects mismatched units or reference/current configurations.
Masked void and outside-domain values are excluded only when all bilinear
contributors are valid, so a convenient fill value cannot silently improve a
field error.

One condition can be handed to another researcher or agent as a sealed package:

```python
bundle = fracture.DynamicFractureEvidenceBundle(
    benchmark_id=condition_id,
    trace=case.trace,
    wave_speeds=wave_speeds,
    energy_history=energy_columns,
    comparisons=(speed_fit, cone_fit, field_fit),
    artifacts={"fields": "fields.xdmf", "SED": "sed_map.npz"},
)
manifest = bundle.write("evidence/condition_07")
```

The package copies its declared artifacts and seals their bytes together with
trace, energy channels, comparison records, and metadata. Integrity is checked
before readback. This makes transfer reproducible; it is deliberately not an
automatic validation decision.

## Obtain and audit the public data

Download the files from the official Dryad landing page. Repository bot
protection must not be bypassed by an embedded scraper. Then run:

```bash
python examples/science_supershear_v5_protocol.py path/to/downloaded/files
```

The command verifies the minimum V5 evidence roles and produces a workbook
inventory. A failed hash, wrong size, or missing file stops the evidence
chain. The full versioned identity is in
`knowledge/external_data/science_supershear_dryad_v7.json`.

## Research sequence

1. Build a data dictionary before fitting anything. Record workbook, sheet,
   cell range, symbol, units, normalization, reference/current coordinates,
   and uncertainty.
2. Freeze a calibration/prediction split. Reserve at least one material or
   weak-layer family and one SED/KED field map. Do not revisit that split after
   seeing prediction error.
3. Calibrate independently measurable bulk behavior, density, wave speeds,
   and fracture energy before interface parameters.
4. Establish mesh, time-step, cohesive-zone, boundary-reflection, energy, and
   crack-observer convergence. Keep one physical speed-fit length across mesh
   levels.
5. Compare crack histories, Mach angle, SED/KED maps, displacement/strain
   maps, radial strain profiles, and critical-prestrain trends.
6. Evaluate the reserved prediction conditions once parameters and processing
   rules are frozen.
7. Attribute discrepancies. Distinguish ambiguous data semantics, parameter
   identifiability, discretization, bulk/cohesive models, loading assumptions,
   two-dimensional reduction, and actual software defects.

## Scientific gates

A V5 result is not one boolean. Every reported condition should carry:

- exact AgentFEM commit and public-data manifest identity;
- geometry, mesh, time increment, stability factor, and cohesive resolution;
- calibration data and untouched prediction data;
- crack-front observer thresholds and fitted-speed spread;
- prestrain-dependent wave-speed convention;
- complete energy ledger and declared numerical damping;
- curve/field residuals, not only selected matching images;
- an explicit conclusion: agreement, partial agreement, disagreement,
  insufficient public information, or software/model gap.

The machine-readable assignment and exact deliverables are in
`knowledge/research_tasks/science_supershear_v5.json`.

## Current boundary

This infrastructure makes an independent public-data study reproducible and
reviewable. It does not create missing JMPS dimensions, loading histories,
cohesive parameters, or mesh specifications. A homogeneous affine
plane-stress/thin-three-dimensional FEM cross-check and physical-keyed
cross-rank-count cohesive state and Explicit force/restart contracts are
implemented. Full thin-3D fracture, general near-incompressible validation,
direct imported internal surfaces, and extreme-scale neighborhood-collective
profiling remain software roadmap items. The 2D route now derives a conforming
path from cell partitions and exchanges only scheduled interface traces and
forces across MPI. The first publication-image registration is
implemented for reviewed affine maps; nonlinear optical calibration, image
segmentation, and uncertainty propagation remain research processing rather
than inferred software behavior.
