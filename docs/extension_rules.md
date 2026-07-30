# AgentFEM Extension Rules

Use these rules before adding new public functions or modules.

## Add to Core Modules When

- The function represents a standard FEM workflow operation.
- The function can reasonably serve more than one problem class.
- The name is conventional for finite-element researchers.
- The input and output types are stable and easy for agents to inspect.

## Add to `constitutive/` When

- The code defines a local response relation.
- The code maps strain, gradient, state, or internal variables to stress, flux,
  tangent, or related quantities.
- The code should not primarily store reusable material constants.

## Add to `materials/` When

- The change is a reusable material-property record or a loader/validator for
  records.
- The record can be expressed as SI-unit data with a model name and source note.
- The code defines typed parameter containers used by constitutive relations.
- The change should not introduce weak forms, stresses, fluxes, or solver logic.

## Add to `boundary_models/` When

- The code defines a reusable weak boundary physics model.
- The boundary term is more than a simple Neumann load.

## Add to `elements/` When

- The code names a reusable interpolation, quadrature, or integration policy.
- The concept applies across multiple applications and is not only a DOLFINx
  one-line constructor.

## Add to `operators/` When

- An assembled operator family has enough reusable behavior to outgrow
  `assembly.py`.
- The code describes mass, stiffness, damping, projection, block, or
  preconditioner-ready operators.
- The API should be understandable to users who think in `K x = F` or
  `M a + C v + K u = F`.

## Add to `benchmarks/` When

- The case has expected quantities, tolerances, and repeatable validation
  commands.
- The goal is verification, not teaching workflow usage.

## Add to `ir/` When

- The code defines versioned scientific records, references, migrations, or
  serialization behavior.
- The representation preserves domain meaning independently of live backend
  object identity.
- The change has round-trip or golden-document tests.

## Add to `backends/` When

- The code describes backend capability or lowers a supported semantic object.
- Unsupported behavior is rejected explicitly.
- The adapter has numerical evidence for every advertised capability.

## Keep in the Application When

- The code encodes a specific geometry.
- The code encodes a specific source waveform.
- The code encodes one paper, one benchmark, or one translation from another
  software package.
- The code is not yet stable enough to generalize.

## Documentation Rule

When adding a new concept-level function, update the relevant docs and skill
references in the same change.
