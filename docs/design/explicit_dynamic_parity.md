# Explicit dynamics migration: nodes, histories and enriched tetrahedra

Branch: `feat/explicit-dynamic-parity`. This development is driven by the blade
impact brief, but contains no blade coordinates, material values or sensor
labels in platform implementation.

## Product boundary

The public modelling path remains Study → mesh → Model → material/constraints/
loads → Step → SimulationResult. Examples describe physics; they do not own a
private time integrator, load assembly, history sampler or element basis.

Three separate statements must not be confused:

1. Reading a C3D10M connectivity record preserves source topology.
2. An independently named positive-mass element provides a numerical method.
3. Matching Abaqus C3D10M histories requires an actual cross-solver experiment.

Only (1) and independent alternatives under (2) are implemented here. No
C3D10M auxiliary displacement/hourglass formula has been implemented. The
original blade INP and Abaqus histories were not present in the supplied
case directory; external Gates D/E remain open.

## Public use

```python
from agentfem import mesh, fields, elements, results, amplitudes
from agentfem.mesh.abaqus import read_dynamic_input

# A lossless flat-deck inventory; not an automatic whole-deck solver.
deck = read_dynamic_input("original.inp", unit_system="mm-tonne-s")
audit = deck.summary()
impact_nodes = deck.nodeset("IMPACT_NODE")

# The imported geometry must retain deck.nodes source order/input identities.
model.nodal_force((0, -1230, 0), nodes=impact_nodes,
                  amplitude=amplitudes.half_sine(25e-6), name="impact")

# Existing ordinary P2: an explicit, separately named HRZ inertia option.
u = fields.displacement(domain, degree=2)
mass = model.lumped_mass(u, method="tet10_hrz")

# Alternative published enriched degree-two space, NOT C3D10M.
u = fields.displacement(domain, element=elements.tet15_mass_lumped())
mass = model.lumped_mass(u, method="tet15_nodal")

step = model.step(target=u, mass=mass, dt="auto", steps=1000,
                  history_every=1, save_every=50)
history = results.dynamic_history("sensor", at=(10, 2, 3),
                                  length_unit="mm", time_unit="s")
result = step.solve_result(history=history, output="fields.xdmf")
results.write_dynamic_history(result, "sensor.csv")
```

The snippets show alternative fields, not successive registrations in one
model. A complete executable case is `examples/explicit_nodal_pulse/case.py`.

### Nodal forces

`model.nodal_force` accepts `at=...`, `labels=..., source=...`, or a
`mesh.nodal_selection` including an imported `deck.nodeset`. The vector is a
**force per selected node**, not a total force distributed over the set.
Only global coordinates and small-strain central difference are initially
admitted. Other lowering routes fail explicitly, rather than reinterpret the
load as pressure. Spatial selection requires a unique node within tolerance.
Coincident seam nodes must be selected by source identity.

Owned node assembly prevents MPI double counting. Nodal-force amplitude
identity is part of the existing time-input checkpoint contract. A zero-time
nonzero force initializes acceleration before the first displacement update.
The external-work ledger integrates the actual force and displacement at every
accepted increment, even when retained histories are sparse. Moving-support
reaction work is not implemented by this monitor.

### Histories

`results.dynamic_history` produces U/V/A components, magnitude and maximum of
recorded samples. Named multi-node selections expand into individual stations.
For fixed reference-mesh physical coordinates, the platform freezes a deterministic FE interpolation
stencil. JSON sidecars retain physical and reference coordinates, source cell,
owner rank, global DOF blocks and basis weights. No nearest-node substitution
is used. At a coincident crack face use node identity for a definite side.

History cadence and full-field cadence are independent. Cumulative maxima are
of the **retained samples**, not an inferred extremum between samples. Declare
length/time units explicitly; the platform does not infer units from values.

### INP audit

`read_dynamic_input` retains raw keyword data, line numbers, node/element labels,
sets, surfaces, load/amplitude/output cards, units and source SHA256. Its
`converted` list is intentionally empty: preservation is not execution.
Part/Instance/INCLUDE scope is rejected by this flat-deck helper and must pass
through the existing reviewed migration route first. Unsupported formulations
are listed separately from successfully read topology.

### Zero-thickness seams

`mesh.split_tetrahedral_seam` accepts Tet4 or Tet10 **Abaqus-order** connectivity,
matching internal faces and an explicit positive-side cell set. All selected
corner and midside nodes are duplicated at exactly coincident coordinates.
No contact or cohesive law is inserted. Source labels, duplication map, parents
and normals are retained. For a finite crack, `front_nodes` explicitly names
shared crack-front nodes. Unselected neighboring faces may not accidentally
open. Degenerate/nonmatching/nonmanifold/repeated splits are rejected.

## Numerical design

### Why ordinary Tet10 row sums are not admissible

The P2 tetrahedral shape functions integrate to negative corner masses. A
positive tolerance or clipping cannot repair that formulation. HRZ scales the
positive consistent diagonal locally and conserves each element's total mass
and centroid. It does **not** reproduce the consistent mass spectrum or rigid
rotation inertia. Independent evidence records these differences.

For symmetric quadrature using only the 4 vertices and 6 edge midpoints,
matching quadratic moments and total volume forces negative vertex weights.
This explains why a positive 10-node diagonal cannot simultaneously reproduce
all these moments just by changing its weights.

### Published enriched Tet15 route

Use the space `P2 + four cubic face bubbles + one quartic interior bubble`.
Basix constructs the basis, orientation and interpolation; FEniCSx assembles
stress divergence and full-integration stiffness. Its 15 scalar nodes are the
4 vertices, 6 edge midpoints, 4 face centers and 1 cell center. Reference-volume
weights are respectively `17/5040`, `2/315`, `9/560` and `16/315` (volume 1/6).
These are positive and integrate the quadratic inertia moments exactly.

The route is straight-sided/affine, constant-density, full-domain and initially
validated for small-strain linear elastic explicit dynamics. There is no
reduced integration or hourglass stabilization. Tests check that the stiffness
has exactly six rigid null modes. Higher mode frequencies still differ from
consistent-mass frequencies; the complete spectra are retained.

This adds degrees of freedom and is **not** an INP-preserving C3D10M substitute.
It is a useful independently named high-order explicit capability and a
research comparison route for the eventual Abaqus experiments.

### Stable increment and energy

For positive diagonal M and symmetric linear-elastic K, the absolute row-sum
bound of `M^-1 K` gives a conservative spectral upper bound. Automatic dt uses
`0.8 * 2 / sqrt(bound)`. No mass scaling is enabled. A supplied fixed dt is
recorded together with the estimate; the estimate is not an exact eigenvalue.
Expert residuals retain responsibility for their own stability declaration.

The nodal-load route records kinetic/strain energy, trapezoidal external work,
initial energy and balance error. Unforced validation includes energy drift.
Displacement, velocity and acceleration restart with the work endpoint.

### Result storage

Ordinary Tet10 uses the existing unified XDMF/HDF5 writer. The enriched Tet15
serial writer retains **all native nodal coefficients** plus SourceCellDofs in
HDF5. For viewing it subdivides each source cell into 24 linear tetrahedra;
this approximation is labeled in both XDMF and HDF5. DG0 attributes inherit
the source parent cell. This display subdivision does not change the solver
basis. Distributed enriched XDMF delivery is not yet admitted; MPI histories
and checkpoint/restart are separately tested.

### Cross-solver reporting

`results.read_velocity_csv`, `compare_dynamic_traces` and
`write_dynamic_comparison` use declared units and three raw velocity components.
They report speed/max, peak/time, absolute-threshold first arrival, RMS,
normalized time-integrated vector L2, correlation and component-vector spectra.
Peak/RMS ratios also support a separately identified cracked/healthy comparison.
No automatic smoothing, scaling, time shift or extrapolation is applied.
Different sampling grids require explicit `align="interpolate_candidate"`;
original raw data are always written as well. The spectrum uses the three
components rather than the magnitude (which can double a sinusoid's frequency).

## Evidence and remaining work

See `validation/dynamic_parity/REPORT.md`. The independent analytical waves,
MPI/restart results and element checks are not substituted for Abaqus data.
The blade must not be declared matched until Gates C–E and its original input
identity are resolved. Contact, plasticity, curved geometry, heterogeneous
Tet15 density, C3D10M auxiliary/hourglass behavior and distributed enriched
field output are outside this first implementation.

## Sources consulted

- Abaqus solid elements: https://docs.software.vt.edu/abaqusv2026/English/SIMACAEELMRefMap/simaelm-c-solidcont.htm
- Abaqus interpolation/integration: https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-tritetwedge.htm
- Geevers, Mulder & van der Vegt, SIAM J. Sci. Comput. 40 (2018), Table 1: https://doi.org/10.1137/18M1175549
- Dispersion analysis of explicit tetrahedral methods: https://doi.org/10.1007/s10915-018-0709-7
- Higher-order HRZ accuracy and stability: https://doi.org/10.1002/nme.7499
- Precise mid-node mass research: https://doi.org/10.6052/0459-1879-23-241

Implementation uses independently written mathematical expressions, not copied
solver code or a proprietary Abaqus implementation.

## Direction after the first external comparison

The [Abaqus 2026 continuum-element documentation](https://docs.software.vt.edu/abaqusv2026/English/SIMACAEELMRefMap/simaelm-c-solidcont.htm)
now states that C3D10 has better accuracy and performance than C3D10M in
Abaqus/Explicit. Matching the supplied C3D10M experiment and choosing the
platform's longer-term high-order explicit discretization are therefore
separate engineering decisions. This statement is not evidence that either
AgentFEM alternative matches Abaqus C3D10 or C3D10M. Retain the original
reference for parity and compare alternative formulations as separately named
methods on a common physical problem.
