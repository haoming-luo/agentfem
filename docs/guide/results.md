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
