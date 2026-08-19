# ADR 0003: Scientific trust is separate from execution status

## Status

Accepted for the 0.1 release line.

## Context

AI-generated CAE workflows can produce syntactically valid input and a
converged solver run while encoding the wrong orientation, inadequate local
mesh resolution, or an inapplicable analytical comparison. Existing result
status fields described process completion but could not represent this gap.

## Decision

AgentFEM uses an ordered trust vocabulary: `not_computed`, `computed`,
`converged`, `verified`, and `validated`. `VerificationClaim` records an
observable, reference, criterion, applicability domain, evidence, and a
passed/failed/inconclusive status. `SimulationResult` carries the report but
does not overwrite execution status. Campaigns may require a minimum trust
level before admitting data to a learning dataset.

A fixed Golden, convergence study, metamorphic invariant, cross-solver result,
or experiment is represented as a claim with its own declared scope. No one
evidence type is silently promoted into another.

## Consequences

- Successful execution is never described as scientific verification by
  default.
- Inapplicable theories remain visible as inconclusive evidence.
- Verification policy is deterministic and machine-readable; an AI agent may
  explain or repair it but cannot redefine the recorded acceptance rule after
  seeing the answer.
- Release examples need both small automated contracts and honest limitations.
- Future service, MCP, or agent interfaces consume the same result and claim
  records rather than scraping logs.
