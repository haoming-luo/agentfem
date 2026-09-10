# Result-field semantics: raw values, projection, and smoothing

Finite-element result names are incomplete without a location and a processing
history. A stress tensor at integration points, a discontinuous cell average,
an extrapolated element-nodal value, and a nodally averaged contour can share
the label `S` while having different numerical values. AgentFEM therefore
treats result processing as scientific metadata rather than a hidden viewer
setting.

## What established CAE systems display

Abaqus commonly stores element variables such as stress at integration points.
For ordinary contour display, Abaqus/CAE can extrapolate element values to
nodes and average contributions according to the current averaging criteria.
The displayed extrema can consequently change with those criteria. Abaqus also
distinguishes a stored stress tensor from invariants computed from it: the
order of extrapolation, invariant calculation, and averaging can change the
plotted Mises field. Its documentation explicitly notes that extrapolated
nodal Mises values can exceed the integration-point yield stress.

COMSOL likewise distinguishes Gauss-point evaluation from presentation. Its
`gpeval` operator constructs an approximate smooth field from Gauss-point data
by least-squares fitting. Result plots expose smoothing policies such as none,
inside material domains, inside geometry domains, and everywhere; the usual
material-domain policy avoids blending values across different materials.

ANSYS Mechanical defaults to averaged contours for many element-nodal
quantities but also exposes unaveraged contours, nodal differences, and nodal
fractions. The discontinuity between neighboring element contributions is
therefore available as mesh-quality evidence rather than being treated only as
a visual defect.

These systems demonstrate two useful principles:

1. smooth contours are a presentation choice, not the constitutive truth;
2. material boundaries and extrapolation order are part of result semantics.

## AgentFEM default

For small-strain elasticity, one-call static output uses the engineering field
set `U/S/E/MISES`:

| variable | role | default representation |
| --- | --- | --- |
| `U` | primary displacement unknown | continuous finite-element solution |
| `S` | Cauchy stress | discontinuous cell-average L2 projection |
| `E` | infinitesimal strain | discontinuous cell-average L2 projection |
| `MISES` | immediately useful invariant of stress | invariant evaluated from the constitutive stress, then discontinuously projected |
| `SENER` | strain-energy density | available but opt-in diagnostic field; for finite-strain J2, `ELENER + HARDENER` |
| `ELENER` | elastic or condensed mixed stored-energy representation | provider-owned finite-strain J2 quadrature field; mixed-route point values require the local pressure constraint for equivalence to the primal volumetric energy |
| `HARDENER` | isotropic-hardening stored-energy density | provider-owned finite-strain J2 quadrature field |
| `PDENER` | cumulative irrecoverable plastic-dissipation density | provider-owned finite-strain J2 quadrature state field |
| `MEAN_KIRCHHOFF_STRESS` | independent mixed J2 volumetric unknown, positive in tension | discontinuous primary field for the experimental 3D P2/DG0 and 2D Q2/DPC1 finite-strain J2 routes |
| `MIXED_POTENTIAL` | mixed J2 saddle variational density | provider-owned quadrature diagnostic; not pointwise stored energy |
| `V`, `A` | velocity and acceleration | nodal transient state fields |
| `KED` | kinetic-energy density per reference volume | cell field computed as $\tfrac12\rho_0\mathbf{v}\cdot\mathbf{v}$ when velocity and density are supplied |

`MISES` is deliberately materialized even though it can be derived from `S`:
it gives users an immediate deformed stress contour in ordinary visualization
tools. `SENER` is not preselected because a full energy-density field is less
universally useful than total strain energy and energy-balance histories.
For finite-strain J2, `SENER` remains backward compatible and has the precise
meaning `ELENER + HARDENER`: recoverable Hencky elastic free energy plus the
stored linear-isotropic-hardening free energy. It is not plastic dissipation.
`PDENER` is reported separately as committed cumulative material dissipation
for the declared rate-independent linear-hardening law. It does not by itself
close the structural energy balance: external work for every load and
constraint remains provider-owned evidence.

The default `DG0` result is a cell average. It is discontinuous, performs no
nodal extrapolation, and does not average across elements or material
interfaces. For first-order displacement elements in linear elasticity this
also preserves the elementwise constant strain and stress exactly. For
higher-order fields, `DG0` is a compact average rather than a complete record
of within-element variation.

The Q2/DPC1 mixed-J2 route preserves the exact
`MEAN_KIRCHHOFF_STRESS` field as three discontinuous cell moments. XDMF's
ordinary `Center="Cell"` attribute cannot encode those moments as one scalar
value per cell. Final-result output therefore keeps the exact DPC1 field in
`SimulationResult`, omits only that incompatible representation from the
ParaView dataset, and adds an explicitly named
`MEAN_KIRCHHOFF_STRESS_CELL` DG0 physical cell average. Its processing record
states `global_l2_projection`, retains the source-field name, and forbids any
interpretation as nodal extrapolation or smoothing. Explicit requests for the
unrecovered DPC1 field fail closed instead of silently relabeling a reduced
field. Portable checkpoint identity preserves the exact cell moments by
original physical cell and local mode. Fresh-Step checkpoint/continue is
verified for both serial mixed routes: 3D tetrahedral P2/DG0 and 2D plane-strain
quadrilateral Q2/DPC1. The generic DPC state primitive independently has
one-to-two and two-to-one-rank acceptance coverage, while the mixed J2
equilibrium provider itself remains serial; serializer portability does not
establish a mixed MPI solve or cross-rank mixed-Step restart.

Every generated `FieldResult` records a `processing` mapping containing the
projection method, result space, and explicit false flags for nodal
extrapolation, interelement smoothing, and material-boundary averaging. This
metadata is retained in the result manifest.

An analysis can request diagnostic fields without changing the global default:

```python
result = step.solve_result(
    output="solid_with_energy.xdmf",
    field_variables=("S", "E", "MISES", "SENER"),
)
```

## Scientific and presentation layers

AgentFEM should ultimately expose three related but distinct products:

1. **constitutive evidence** — integration/quadrature-point state for
   path-dependent materials and verification;
2. **scientific fields** — discontinuous fields with explicit projection or
   recovery semantics, suitable for quantitative queries and learning data;
3. **presentation fields** — optional material-aware nodal recovery or
   smoothing for readable contours, always labeled and never overwriting the
   scientific field.

The current release implements the second layer for elasticity. Small-strain
J2 results retain committed `S/PE/PEEQ` and pointwise `MISES` on the
constitutive quadrature. The experimental public ordinary, displacement-only
affine/MPC, and mixed affine finite-strain J2 providers retain the same
provider-owned accepted `F/P/S/MISES/SENER/ELENER/HARDENER/PDENER/FP/PEEQ` at
the same quadrature identity; the mixed route additionally retains
`MIXED_POTENTIAL`. The output layer does not recompute them from a history-free
constitutive expression. Explicit `*_CELL` products never
overwrite a same-named raw quadrature field. J2 and implicit creep also expose
separately named `*_CELL` fields through
`results.recover_integration_point_field(...)`. These fields use the actual
quadrature weights to form a DG0 cell average and record the source position,
point count, target space, and explicit absence of extrapolation, smoothing,
or material-boundary averaging. This is material-aware in the strict sense
that values never cross an element or material interface; it is not yet a
smooth nodal contour recovery.

```python
cell_peeq = results.recover_integration_point_field(
    step.state.equivalent_plastic_strain,
    name="PEEQ_CELL",
)
```

Direct general quadrature-file export and reviewed material-domain nodal
recovery remain roadmap items. A naive global continuous projection is
intentionally not presented as a standard smoothing method because it can
erase real jumps at material interfaces and obscure singular or poorly
converged regions.

The two mixed finite-strain families intentionally use different public field
names because their scalar unknowns have different conventions. In mixed
Neo-Hookean output, `PRESSURE` is the independent cellwise Cauchy-pressure
unknown and is positive in compression. In mixed finite-strain J2 output,
`MEAN_KIRCHHOFF_STRESS` is \(\operatorname{tr}\boldsymbol\tau/3\) and is
positive in tension. `P` remains the first-Piola tensor derived from the
coupled solution in both families. Software and users must not rename either
scalar field to a generic `PRESSURE` and silently erase its stress measure or
sign convention.

For mixed J2, `ELENER` contains the deviatoric elastic storage plus the
nonnegative condensed mixed term \(p^2/(2\kappa)\). This term is pointwise
equivalent to the primal volumetric energy \(\kappa(\ln J)^2/2\) only where the
local constraint \(p=\kappa\ln J\) holds. The weak discrete equation does not
make that pointwise identity automatic, so the channel must be described with
its mixed discretisation when compared externally. The distinct
`MIXED_POTENTIAL` field carries the saddle density used to derive the pressure
equation; it is never an alias for `ELENER` or `SENER`.

## References

- [Abaqus: understanding contour limits](https://docs.software.vt.edu/abaqusv2024/English/SIMACAECAERefMap/simacae-c-conconceptlimits.htm)
- [Abaqus: selecting field output variables and output position](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-reportfieldsetuptabbtn.htm)
- [Abaqus: integration-point output variables and stress invariants](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEOUTRefMap/simaout-c-std-elementintegrationpointvariables.htm)
- [Abaqus: extrapolation, averaging, and Mises contours](https://abaqus-docs.mit.edu/2017/English/SIMACAEGSARefMap/simagsa-c-matpostprocess2.htm)
- [COMSOL: Gauss-point evaluation](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.224.html)
- [COMSOL: stress evaluation and smoothing](https://www.comsol.com/blogs/how-to-evaluate-stresses-in-comsol-multiphysics/)
- [ANSYS Mechanical: averaged and unaveraged contour results](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/wb_sim/ds_Unaveraged_Results.html)
