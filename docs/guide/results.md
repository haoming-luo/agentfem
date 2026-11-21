---
title: "Finite-element results, histories and visualization"
description: "Inspect displacement and stress fields, record probes and reaction histories, and export results while preserving units, provenance and verification status."
---

# Results and post-processing

A useful result is more than an array written after a solver exits. AgentFEM's
result lifecycle brings fields, histories, artifacts, progress, checkpoints,
quality policy, verification, and failure state under one contract.

## Result layers

1. **Scientific fields** preserve finite-element meaning on the reference
   configuration and distinguish integration-point/discontinuous evidence from
   optional nodal presentation fields.
2. **Engineering histories** record reactions, energies, balance errors,
   resultants, paths, probes, and other quantities over load or time.
3. **Visualization output** makes the field set convenient to inspect without
   changing the scientific source of truth.
4. **Structured result metadata** records status, trust, quality acceptance,
   files, provenance, and restart identity.

For long runs, [binary history storage](history_storage.md) keeps complete
binary64 curves in a companion archive rather than expanding every sample
into JSON. It is opt-in and does not change the field-frame schedule.

Accepted-frame histories use one reusable request rather than one function per
analysis type:

```python
output = results.output_plan(
    "output",
    requests=(
        results.probe_history("tip_U2", at=tip, component=1, unit="mm"),
        results.history(
            "section_force",
            lambda frame, context: evaluate_section(frame.solution),
            unit="N",
        ),
    ),
)
```

The same request contract can represent probes, integrals, resultants,
energies, or application-defined quantities. The abscissa is taken from the
accepted physical time or normalized load factor; custom frame types must
provide an explicit coordinate instead of falling back to an arbitrary index.

Transient steps evaluate the same vocabulary online, without keeping every
field frame in memory:

```python
result = step.solve_result(
    output="results.xdmf",
    history=(
        results.probe_history(
            "sensor_temperature",
            at=(0.5, 0.1),
            unit="K",
        ),
        results.history(
            "mean_temperature_dof",
            lambda accepted_step, time: np.mean(
                accepted_step.current.x.array
            ),
            unit="K",
        ),
    ),
)
```

For dynamics, an omitted `field=` selects displacement; for first-order heat
it selects current temperature. Pass a live field or callback for another
state variable. Vector probes require `component=...` because one history
channel has one scalar engineering meaning.

## Points, paths and planar sections

Use the same field samplers for displacement, recovered cell stress or scalar
temperature. Every MPI rank calls with identical coordinates. The sampler
evaluates the finite-element basis in an owned containing cell and returns the
same ordered values on all ranks:

```python
tip = results.probe(displacement, at=(1.0, 0.1))
points = results.sample_points(stress_field, [(0.2, 0.1), (0.8, 0.1)], return_info=True)
path = results.sample_path(
    stress_field, vertices=[(0.0, 0.1), (0.8, 0.1), (0.8, 0.2)], count=101,
)
path.add_to(result, name="stress_along_path", unit="Pa", distance_unit="m")
```

`start=...`, `end=...` retain the straight-path API. `vertices=...` instead
describes a polyline: stations are uniformly spaced in cumulative arc length,
including both endpoints. Increase `count` to resolve corners or supply explicit
point coordinates when specific stations must be retained. A discontinuous
field evaluated exactly on a cell interface uses the deterministic containing
cell choice documented by `sample_points`; sample inside the desired side for
a one-sided stress trace.

On a 3D mesh, use a physical section without extracting DOLFINx cell arrays:

```python
section = results.sample_section(
    stress_field,
    origin=(0.5, 0.0, 0.0), axis_u=(0.0, 1.0, 0.0), axis_v=(0.0, 0.0, 1.0),
    u=np.linspace(0.0, 0.2, 21), v=np.linspace(0.0, 0.1, 11),
    missing="nan",
)
```

The section maps `x=origin+u*axis_u+v*axis_v`. Its axes must be orthonormal;
`u` and `v` are physical length coordinates. Arrays have `(len(v), len(u))`
grid order followed by the field value shape. `located` and `finite` masks
preserve holes and distinguish missing material from nonfinite values.
`missing="raise"` is the default. Section samples are point observations, not
cut-cell quadrature: use `section_resultant(stress, on=mesh_boundary, ...)`
for an integrated mesh-boundary force and moment.

## Reactions, field comparisons and accepted frames

For a solved strong-Dirichlet problem, sum the discrete residual forces and
their moment about an explicit reference point:

```python
support = results.reaction_force_moment(
    step.problem, on=support_boundary, about=(0.0, 0.0),
)
support.force       # vector
support.moment      # scalar out-of-plane in 2D, vector in 3D
support.as_dict()
```

This uses owned nodal residual entries and reference mesh coordinates; no
stress smoothing or boundary extrapolation enters the calculation. Scope
matches `reaction_resultant`: converged Cartesian solid problems with strong
constraints. Contact/MPC reaction distributions use their provider-owned
definitions. Two-dimensional results retain the selected problem's thickness
convention. To move the reference point from A to B,
`M_B=M_A-(B-A) cross F`.

Compare fields at common physical locations, including different meshes:

```python
comparison = results.compare_fields(reference_field, candidate_field, at=points.coordinates)
comparison.difference  # candidate minus reference
comparison.summary()
```

The result retains both sample arrays and a common validity mask. Reported
maximum, RMS and relative errors refer to the requested unweighted samples,
not a volume norm or an automatic convergence claim. A zero reference norm
produces an unavailable (`None`) relative error; absolute errors remain
available. `missing="nan"` explicitly permits partial coverage, which the
summary reports. Compare matching units, components and stress measures.

Select retained accepted fields by stored index or a nearby load/time value:

```python
frame = results.accepted_snapshot(step, at=0.5, coordinate="load_factor", tolerance=0.01)
tip_at_half_load = results.probe(frame.solution, at=(1.0, 0.1))
stored_fields = frame.fields  # when supplied by the owning procedure
```

Use `index=...` instead when a cyclic path visits the same load factor more
than once. The returned frame is borrowed from the step and should be treated
as read-only. Only saved fields are available: no interpolation, material
reintegration or live-state rewind occurs. A missing frame, coordinate outside
tolerance or ambiguous repeated coordinate raises an addressable error. Select
field retention before solving when this analysis is required; scalar histories
alone do not contain past spatial fields.

## Periodic plastic-contact energy evidence

For the primal periodic/affine finite-strain J2 route, the result also contains
`metadata["mechanical_energy_ledger"]` and accepted scalar histories for stored
material energy, plastic dissipation and, when present, recoverable contact
potential and friction losses. The contact potential includes both normal and
tangential penalty storage. The reported balance subtracts **changes** from
the initial accepted state, not absolute final energy, and combines affine
prescribed-motion work with moving-tool work.

`status="complete"` means the required histories cover the accepted path; it
does not mean the balance meets a scientific tolerance. Work uses accepted-path
trapezoidal integration, so increment refinement is still required. Augmented
contact and mixed saddle-point routes are not assigned a fictitious penalty
energy. Older checkpoints without the scalar energy history may resume, but
the ledger remains unavailable rather than reconstructing missing past values.
Scalar histories share rollback/restart with the solve and do not force a
full-field copy at each increment or change the selected field-output schedule.

## Standard questions

- Did the requested step converge or complete stably?
- Were all required fields and histories produced?
- Are equilibrium, energy, and conservation errors within policy?
- Is the result inside the method's applicability and benchmark envelope?
- Can the run be reproduced or restarted from its recorded state?

## Go deeper

- [Results and campaigns](../results_and_campaigns.md)
- [Result-field semantics](../result_field_semantics.md)
- [Result provenance seal](../provenance_seal.md)
- [Simulation to learning](simulation_to_learning.md)
