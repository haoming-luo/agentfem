# Engineering feedback decision record: 3D brake-caliper study

This record turns one external engineering trial into reusable product
decisions. The case used a quadratic tetrahedral mesh, a P2 displacement field,
tagged pressure and clamp surfaces, and a three-dimensional linear-elastic
solve. It is evidence about workflow reliability, not a request to copy
case-specific code into the library.

## Decisions made now

| Feedback | Decision | Product consequence |
| --- | --- | --- |
| a boundary carried both a marker and an imported physical tag | accepted as P0 | `BoundaryRegion.selection` names the source of truth; imported tagged and hybrid regions use facets topologically for strong constraints and `ds(tag)` for weak terms |
| marker and tag could disagree silently | accepted as P0 | `BoundaryRegion.audit(strict=...)` and `Model.audit_boundaries(...)` report facet counts, measure, midpoint bounds, integrated normal, and set differences |
| a pressure area required `ufl.as_ufl(1)` in the case | accepted | `results.region_measure(on=region)` is the public application-level operation |
| one DOLFINx XDMF grid was produced per field | accepted as P0 for serial product output | ordinary `AnalysisStep.solve_result(output=...)` uses AgentFEM's one-grid XDMF/HDF5 layout; `U`, point fields, and DG0 cell fields are Attributes of the same Uniform Grid |
| a CG1 result could not be written on P2 output geometry | accepted | continuous auxiliary fields are represented on the solution's nodal output grid before unified XDMF storage; unsupported higher-order discontinuous fields fail with the field name and a remediation |
| optional output failure erased the meaning of a successful solve | accepted as P0 | output failure produces `completed_with_output_errors`, retains the live result, records the exception, and only raises with `strict_output=True` |
| extrema lacked a location | accepted as P1 | `field_extrema(..., location=True)` reports coordinates, rank, global dof, sampling method, and DG0 cell identity |
| stress display meaning was unclear | already addressed and reinforced | default `S/E/MISES` fields remain explicitly documented DG0 cell-average projections without nodal extrapolation or smoothing |

The intended application code is now:

```python
pressure_surface = mesh.tagged_boundary_region(
    domain, facet_tags, tag=102, name="pressure_surface"
)
bolt_holes = mesh.tagged_boundary_region(
    domain, facet_tags, tag=101, name="bolt_holes"
)

model.pressure(16.0e6, on=pressure_surface)
model.clamp(displacement, on=bolt_holes)

boundary_evidence = model.audit_boundaries(strict=True)
pressure_area = results.region_measure(on=pressure_surface)
simulation = model.step(target=displacement).solve_result(
    output="outputs/result.xdmf",
    field_variables=("S", "E", "MISES", "SENER"),
)
peak = results.field_extrema(simulation.fields["MISES"], location=True)
```

This removes the need for case-owned boundary-area UFL and ordinary XDMF
plumbing. Expert UFL remains available for new physics and nonstandard
quantities, but it is no longer required for this standard workflow.

## Deliberately not disguised as a quick fix

AgentFEM does not rename a global continuous projection as SPR or PPR. A smooth
scalar `MISES` projection is useful for presentation, but it is not equivalent
to recovering stress tensor components and then computing the invariant.
Material-aware nodal recovery therefore remains a reviewed mechanics feature,
with linear patch, bending, material-interface, and stress-concentration
benchmarks required before it becomes a standard representation.

The same discipline applies to broader NASTRAN CTETRA10/CTRIA6 support and a
mesh-study object. Both are valuable, but they require preserved PID semantics,
node-order and Jacobian evidence, and convergence measures beyond a single
maximum. They are not allowed to displace boundary identity and output
reliability work.

## Visualization and checkpoint roles

In serial, `solve_result(output=...)` writes one temporal Uniform Grid per
frame. `U` and other continuous fields are point data; `S`, `E`, `MISES`, and
other DG0 fields are cell data. The geometry is the reference configuration,
so ParaView's **Warp By Vector** uses `U` once and cannot reveal a second
unwarped copy hidden inside the same reader.

The low-level `io.XDMFTimeSeries` intentionally remains a thin DOLFINx writer.
DOLFINx writes multiple Functions as separate XDMF Grids, so it is not the
recommended ParaView product for a multi-field serial analysis. Existing
files can be inspected with **Extract Block**, followed by **Append
Attributes**, then **Warp By Vector**.

Under MPI, AgentFEM keeps DOLFINx's collective XDMF/HDF5 scientific path rather
than gathering a large distributed mesh to rank zero. The result lifecycle now
also writes and registers a collective PVD/PVTU presentation artifact. Each
time value has one distributed geometry carrying mixed point and cell fields,
so the user-facing ParaView path no longer depends on multi-block extraction.
VTX/BP remains a future option for selected high-order workflows, but it is not
used to hide incompatible element-location semantics.

Checkpoint/state output and visualization output are separate contracts:
checkpoints preserve continuation state and identity; visualization products
optimize field discovery and presentation. Neither may silently substitute for
the other.

## Next evidence-bearing increments

1. Couple boundary geometry evidence with applied-load resultant, solved
   reaction, and relative equilibrium error in `SimulationResult`.
2. Define `nodal_l2` as an explicitly named presentation representation, then
   implement tensor-first, material-aware recovery separately.
3. Add CTETRA10/CTRIA6 NASTRAN fixtures that verify PID separation, node order,
   positive Jacobians, area, and bounding boxes.
4. Design a mesh-study result around displacement, energy, reaction, path and
   regional measures, hotspot drift, and singularity warnings.
5. Add a collective visualization backend only after mixed-field and MPI
   ownership tests define what “one dataset” means in distributed output.
6. Extend `agentfem doctor` from dependency versions to the exact Python
   executable, imported package path, installed distribution path, and a clear
   warning when a source checkout shadows another installation.

## Feedback intake rule

A feedback item is promoted when it supplies a reproducible case, identifies a
scientific or usability consequence, and can be guarded by a stable test. It is
then classified as: correctness risk, evidence gap, workflow friction, new
capability, or presentation preference. Correctness and silent-failure risks
come first. A useful suggestion can still be deferred when its honest
implementation needs benchmarks or a larger semantic design.
