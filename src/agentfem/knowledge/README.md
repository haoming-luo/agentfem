# AgentFEM scientific knowledge assets

This directory is the versioned scientific memory of AgentFEM. It complements
source code rather than paraphrasing it.

## Asset types

- `cards/`: one machine-readable Scientific Function Card per public
  scientific capability;
- `benchmarks/`: executable or reviewable physical/numerical evidence;
- `decisions/`: architecture decision records explaining why durable choices
  were made;
- `external_data/`: immutable identities for public scientific datasets;
- `research_tasks/`: machine-readable handoffs that separate software
  capability from a named research interpretation;
- `schema/`: formal interchange schemas;
- `catalog.json`: generated compact index for agents and tools.

`docs/reference/scientific_function_reference.md` is generated from the cards
for human readers. Do not edit the generated catalog or reference manually.
Card equations are TeX source, not ASCII mathematical pseudocode. The builder
rejects common ambiguous forms such as bare `Delta`, `delta`, `sum`, `max`,
`dot`, parenthesized subscripts, and double-pipe norms before they reach the
website.

## Definition of done

A new public scientific feature is not complete merely because code executes.
It should provide:

1. a typed public API;
2. a Scientific Function Card;
3. equations or an explicit non-equational scientific contract;
4. assumptions, conventions, units, applicability, and limitations;
5. a minimal example;
6. automated tests;
7. physical or numerical benchmark evidence where applicable;
8. validation and AF-IR/capability consequences;
9. Skill routing when an agent must make a scientific choice.

Publication-facing work additionally needs an external-data manifest, a
calibration/prediction split, and a research-task handoff. Public repository
data is referenced and audited; it is not silently copied into the package.

For analysis procedures, completion also requires structured execution
evidence. Progress text, result histories, status files, and agent monitoring
must consume the same event stream; an implementation must not hide failed
attempts or silently omit increments that were not printed.

Run:

```bash
python build_knowledge.py
python build_knowledge.py --check --check-imports
```

The first command regenerates human and machine views. The second command is a
CI-safe consistency and public-API check.
