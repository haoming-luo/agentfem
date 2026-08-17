# Develop and extend AgentFEM

AgentFEM keeps a concise engineering surface without closing the finite-element
stack. Extensions should enter at the narrowest reusable layer that owns their
scientific meaning.

## Common extension routes

| Need | Primary route |
| --- | --- |
| New material law | `constitutive/` or `mechanics/`, with state and tangent contract |
| New load or boundary behavior | `loads.py` or `boundary_models/` |
| New solution procedure | `procedures.py`, step provider with `StepOptionContract`, result lifecycle |
| New field or engineering quantity | `results/` with scientific/presentation semantics |
| New mesh format | `mesh/` conversion plus provenance and fixtures |
| New agent/GUI client | public Python API or structured CLI only |
| Private product or workflow pack | separate package using `agentfem.extensions` |

## Extension rule

A reusable function should have a real case, governing formula or algorithm,
tests, expected output, failure behavior, and at least one consumer. Avoid
placing a one-off example helper inside the core package.

## Continue

- [API style](../api_style.md)
- [Extension rules](../extension_rules.md)
- [Extension packages and private products](../extensions_and_private_products.md)
- [Module map](../module_map.md)
- [Agent and GUI integration](../agent_gui_integration.md)
- [Abaqus user-material bridge](../abaqus_user_material_bridge.md)
