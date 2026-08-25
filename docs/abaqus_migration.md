# Migrating Abaqus Projects

AgentFEM treats an Abaqus input deck as an engineering asset, not merely a
container of node coordinates.  Migration therefore starts with a
side-effect-free inventory before mesh conversion or solving:

```bash
agentfem inspect-abaqus model.inp
agentfem inspect-abaqus model.inp --json --write migration.json
```

The same operation is available from Python:

```python
from agentfem import mesh

report = mesh.inspect_abaqus_input("model.inp")
print(report.text())
```

When the inventory is understood, create a reviewable migration project:

```bash
agentfem migrate-abaqus model.inp ./agentfem-model
cd agentfem-model
agentfem check
```

The generated project is intentionally fail closed. It copies the complete
source graph, writes machine-readable `migration.json` and a compact
`migration.md` review, and creates ordinary `case.py` plus `agentfem.toml`; it
does not run until the engineer or agent has reviewed the scientific lowering
decisions.

The report records the source fingerprint, recursive include graph, element
declarations, node and element counts, NSET/ELSET/SURFACE semantics, equation
count, keyword inventory, and migration warnings. It does not write a
converted mesh and it does not infer that execution success proves formulation
equivalence.

## Three different meanings of support

An Abaqus element name combines geometric topology with numerical choices.
For example, `C3D8`, `C3D8R`, and `C3D8H` share eight-node hexahedral
connectivity, but reduced integration, hourglass control, and hybrid pressure
variables are different scientific formulations.

AgentFEM reports these levels separately:

1. **Declaration and topology** — the source type, connectivity, sets, and
   face semantics can be retained.
2. **Neutral conversion** — the tested meshio 5.3.x reader maps the
   declaration to a neutral cell topology; later dependency versions are
   checked again during release validation.
3. **Native analogue** — AgentFEM has a corresponding public finite-element
   route, with its own documented variational formulation.
4. **Verification evidence** — an explicit benchmark supports a bounded
   equivalence or accuracy claim.

No suffix is silently discarded.  `C3D8R` may be imported as hexahedral
topology while the report still states that Abaqus hourglass control was not
reproduced.  A direct `C3D10H` workflow selects AgentFEM's documented P2/DG0
mixed analogue rather than entering a displacement-only solve.

These states appear independently as `import_capability`,
`neutral_conversion`, and `solver_capability`. This is important for types
such as higher-order bricks, shells, and cohesive elements: AgentFEM may know
their source connectivity even when the current neutral converter or a
dedicated native formulation is not yet available.

Query the semantic catalog with:

```python
mesh.supported_abaqus_element_types()
mesh.supported_abaqus_element_types(family="continuum_solid")
mesh.abaqus.describe_element_type("C3D8R").summary()
```

The first broad catalog covers common 2D/3D continuum solids, axisymmetric
solids, heat-transfer topologies, cohesive declarations, and selected
truss/beam/shell declarations.  Dedicated beam, shell, cohesive, hybrid, and
reduced-integration lowering remains governed by its own formulation and
verification work; catalog presence is not a solver claim.

## Current preserved semantics

The focused adapter currently preserves:

- source node and element labels;
- element declarations and formulation-relevant suffixes;
- NSET, ELSET, and explicit SURFACE entries;
- continued homogeneous EQUATION definitions;
- source fingerprints and conversion choices;
- selected three-dimensional solid face reconstruction;
- direct C3D4 internal-surface lowering for the existing cohesive kernel.

The inspector recognizes familiar material, section, assembly, load,
amplitude, and Step keywords but marks them `recognized_not_lowered`.  This is
deliberate: a future project migrator can consume the same report and request
an explicit decision instead of silently substituting a different material or
procedure.

Part and instance labels are scoped in Abaqus. The inspector can count decks
with repeated labels across Parts without flattening them; the report marks
that an instance-aware lowering decision remains. Nested `*INCLUDE`
dependencies are resolved relative to the declaring file and recorded as a
content-addressed graph. Missing files and recursive cycles make the graph
incomplete, while scoped semantics remain unflattened until an explicit
instance-aware migration stage is selected.

## Scope-aware migration plan

`mesh.plan_abaqus_migration(...)` reads resolved include files in their
declared order while retaining the original file and line number of every
engineering object. The plan distinguishes:

- model, Part, Assembly, and Instance scopes;
- same-named NSET/ELSET declarations belonging to different scopes;
- Part definitions from the instances that reuse them;
- section declarations from their effective material assignments;
- material identity from its behavior keyword blocks;
- recognized native candidates from behaviors requiring scientific review;
- scoped element declarations with topology and formulation-relevant suffixes;
- Step procedures, loads, boundary conditions, amplitudes, interactions, and
  output requests that are preserved but not yet lowered.

For a conventional isotropic `*ELASTIC` card accompanied by `*DENSITY`, the
plan records an `isotropic_elastic` candidate and its source values. If density
is absent, the material remains review-required because AgentFEM does not
invent the missing property. Even a complete candidate is not executed
automatically: units, analysis assumptions, element formulation, loads, and
verification remain project decisions.
Composite sections retain flags and layer rows for review rather than being
misrepresented as one homogeneous material.

`*USER MATERIAL` and user-defined `*HYPERELASTIC` declarations receive a
dedicated review status. Constants, `*DEPVAR`, and source locations remain in
the plan, but the input deck alone cannot supply the Fortran source, compiler
ABI, stress/tangent convention, or validation evidence required to execute a
UMAT, VUMAT, or UHYPER. Migration therefore points toward AgentFEM's user
material contract instead of pretending to translate arbitrary subroutines.

Part-level section assignments are projected onto every matching Instance in
`effective_assignments`. Instance positioning data remain explicit and receive
a review finding until the corresponding mesh transform has been lowered.
Missing Part, ELSET, or material references receive stable error codes and
block native execution.

The `pending_assets` section is equally important: it keeps source rows and
locations for procedures, loads, boundary conditions, amplitudes,
interactions, and output requests. Their presence in the plan is evidence of
preservation, not a claim that Abaqus execution semantics have already been
reproduced.

## Target migration project

The intended generated project remains readable and keeps the source deck:

```text
project/
├── case.py
├── agentfem.toml
├── AGENTS.md
├── migration.md
├── migration.json
├── mesh/
├── materials/
├── source/
│   ├── model.inp
│   └── included-files...
└── outputs/
```

The copied input graph is authoritative migration evidence. Derived XDMF/HDF5 is a cached
solver artifact whose manifest is invalidated when the source or conversion
policy changes.

## Design lineage

Abaqus defines a material independently of a Step, combines relevant material
behaviors, and assigns it to regions through sections.  AgentFEM retains that
separation while expressing the final model in readable Python.  See the
[Abaqus material definition](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-c-prppropmaterial.htm),
[combining material behaviors](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-matmodels.htm),
and [section assignment](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-solidsection.htm)
documentation for the source concepts.
