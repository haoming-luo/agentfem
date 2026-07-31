# AgentFEM scientific knowledge assets

This directory is the versioned scientific memory of AgentFEM. It complements
source code rather than paraphrasing it.

## Asset types

- `cards/`: one machine-readable Scientific Function Card per public
  scientific capability;
- `benchmarks/`: executable or reviewable physical/numerical evidence;
- `decisions/`: architecture decision records explaining why durable choices
  were made;
- `schema/`: formal interchange schemas;
- `catalog.json`: generated compact index for agents and tools.

`docs/reference/scientific_function_reference.md` is generated from the cards
for human readers. Do not edit the generated catalog or reference manually.

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

Run:

```bash
python build_knowledge.py
python build_knowledge.py --check --check-imports
```

The first command regenerates human and machine views. The second command is a
CI-safe consistency and public-API check.
