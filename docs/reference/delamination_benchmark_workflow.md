# Readable DCB, ENF and MMB finite-element benchmarks

The executable providers in `benchmarks/delamination_fem.py`, `enf_fem.py`
and `mmb_fem.py` keep the physical model visible. Beam-theory comparisons
remain independent analytical oracles; they are not used to produce the
finite-element response.

## What a case specifies

Read each provider in this order:

1. **Geometry and discretization:** length, arm thickness, Q1 cell counts,
   plane-stress/plane-strain assumption and represented precrack.
2. **Material and interface:** elastic properties, cohesive law, strengths,
   fracture energies, initial stiffness, thickness and precrack selection.
3. **Loading and restraints:** DCB equal/opposite crack-mouth forces or
   opening, ENF three-point bending, MMB rigid-lever kinematics.
4. **Equilibrium:** bulk internal force, cohesive contribution, consistent
   tangent, solver tolerances and displacement/load control.
5. **Measurements:** opening/compliance, damage-front and process-zone
   definitions, mode partition, reaction and work–energy balance.

These choices remain in the benchmark. In particular the MMB lever relation,
VCCT calculation and damage-weighted mode fractions are not replaced by a
single opaque `run_specimen(...)` call. The DCB opening remains the original
nodal average, not a silently substituted boundary integral.

## Reusable platform capabilities

### Split rectangular interface mesh

```python
split = interfaces.split_rectangle(
    length=length, lower_height=h, upper_height=h,
    cells_along=nx, cells_below=ny, cells_above=ny,
)
domain = interfaces.create_dolfinx_split_mesh(
    split, comm=MPI.COMM_SELF, cell_type="quadrilateral",
)
```

The array-level builder also accepts unequal layer heights and cell counts.
It specifies no material, crack, loading or solver. Interface node duplication
and connectivity belong to the platform; crack location belongs to the case.
For curved interfaces or imported geometries use the existing general
`split_conforming_*` interfaces instead.

### Named boundary loads

`mesh.partition_boundaries` creates disjoint named facet regions with a common
meshtag table. Multiple loads can therefore be assembled in a single form.
Overlapping facets are rejected; sharing corner vertices is allowed. Separate
`mesh.boundary_region` calls still describe independent selections and should
not be assumed to share integration tags.

DCB uses `loads.traction`, `loads.with_amplitude` and
`operators.boundary_load_vector`: the equal and opposite forces and their
normalization by arm height remain explicit. The three providers no longer
import UFL or dolfinx directly. They still expose finite-element operators;
this is an operator-first specialist workflow, not a claim that every
cohesive configuration has a general `model.step(...)` provider.

### Accepted cohesive states

```python
path = fracture.MonotonicCohesivePath(
    equilibrium, residual, cohesive, observe=observe_damage,
)
advance = steps.advance_monotonic_targets(
    targets, try_accept=path.try_accept,
    initial_coordinate=path.coordinate,
    # Specify increment limits and cutbacks for the physical case.
)
```

`observe_damage(response)` defines specimen-specific measurements in the case.
The platform accepts converged states, commits history, integrates trapezoidal
work and subtracts the first accepted cohesive-dissipation baseline. A failed
nonlinear trial changes none of those totals. Existing equilibrium solvers
own trial rollback; the existing monotonic target driver owns cutbacks.

Scope: nonnegative monotonic displacement coordinates and the magnitude of
its conjugate reaction. This helper is **not** a signed cyclic-work integrator,
an arc-length method or a replacement for the nonlinear equilibrium solver.

## Regression evidence (2026-11-06)

Baseline: `origin/main` at `2c9ddbb7`. Before editing, the existing 29 DCB/ENF/MMB
tests passed. Six representative paths were recorded, then recomputed:

| Path | Recorded points | Changed scalar fields |
| --- | ---: | ---: |
| DCB compliance | 3 | 0 |
| DCB propagation | 31 | 0 |
| ENF compliance | 3 | 0 |
| ENF propagation | 16 | 0 |
| MMB compliance | 3 | 0 |
| MMB propagation | 16 | 0 |

All 72 point records were exactly equal in the tested environment, including
reaction, energy, damage and iteration counts. Existing analytical tolerances
were not relaxed. The final targeted suite passed **82 tests**; Ruff and
`git diff --check` also passed. Independent tests additionally cover asymmetric mesh area,
paired interface coordinates, multi-boundary force resultants, rejected
increments, accepted-state work and programming-error propagation.
