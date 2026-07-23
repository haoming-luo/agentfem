# AgentFEM Extension Rules

Use these rules before adding new public functions or modules.

## Add to Core Modules When

- The function represents a standard FEM workflow operation.
- The function can reasonably serve more than one problem class.
- The name is conventional for finite-element researchers.
- The input and output types are stable and easy for agents to inspect.

## Add to `constitutive/` When

- The code defines a material law or local response relation.
- The code maps strain, gradient, state, or internal variables to stress, flux,
  tangent, or related quantities.

## Add to `boundary_models/` When

- The code defines a reusable weak boundary physics model.
- The boundary term is more than a simple Neumann load.

## Keep in the Application When

- The code encodes a specific geometry.
- The code encodes a specific source waveform.
- The code encodes one paper, one benchmark, or one translation from another
  software package.
- The code is not yet stable enough to generalize.

## Documentation Rule

When adding a new concept-level function, update the relevant docs and skill
references in the same change.
