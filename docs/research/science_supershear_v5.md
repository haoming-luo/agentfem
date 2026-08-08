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
cohesive parameters, or mesh specifications. Thin-three-dimensional and
general near-incompressible cross-checks, MPI cohesive ownership, and a
reviewed publication-image coordinate map remain software roadmap items.
