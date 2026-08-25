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

The report records the source fingerprint, element declarations, node and
element counts, NSET/ELSET/SURFACE semantics, equation count, keyword
inventory, and migration warnings.  It does not write a converted mesh and it
does not infer that execution success proves formulation equivalence.

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
that an instance-aware lowering decision remains. `*INCLUDE` dependencies are
listed and missing files are diagnosed, but a single-file inspection does not
silently execute or expand them.

## Target migration project

The intended generated project remains readable and keeps the source deck:

```text
project/
├── case.py
├── mesh/
├── materials/
├── source/model.inp
└── migration.json
```

The original input is authoritative evidence.  Derived XDMF/HDF5 is a cached
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
