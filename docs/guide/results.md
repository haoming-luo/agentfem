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
