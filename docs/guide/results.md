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
