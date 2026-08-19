# ADR 0001: Treat scientific knowledge as versioned code-adjacent assets

Status: accepted

## Context

Finite-element APIs carry more meaning than their Python signatures. A
stiffness operator, time integrator, material law, dataset, or learned model is
only scientifically usable when its equation, assumptions, units, sign
conventions, applicability, limitations, examples, and evidence can be
inspected.

Keeping that information independently in source docstrings, tutorials, agent
prompts, and papers leads to drift. Generating prose from source code alone is
also insufficient because implementation syntax does not contain the full
scientific contract.

## Decision

AgentFEM will maintain one versioned Scientific Function Card for each public
scientific capability. Cards are machine-readable JSON validated by
`build_knowledge.py`. Human reference documentation and the compact agent
catalog are generated from the same cards.

Source docstrings remain concise API guidance. Examples remain executable
teaching assets. Tests remain executable behavioral contracts. Benchmark Cards
record physical and numerical evidence. Skills route agent behavior to the
canonical cards instead of duplicating their full contents.

## Consequences

- A feature can be rejected as incomplete even when its implementation works.
- Scientific changes require reviewing code, card, tests, and evidence
  together.
- Generated reference files must remain deterministic and are checked in CI.
- Cards must distinguish code-enforced conditions from assumptions that are
  only documented.
- The initial JSON representation favors stability and tooling over authoring
  elegance; a later schema version may add richer references without silently
  changing existing meaning.
