# Materials and Constitutive Behaviors

A material name, a constitutive equation, and an analysis procedure are not
the same concept.

- `Study` states the physical problem and modeling assumptions.
- a constitutive behavior states the local response equation;
- a material definition retains physical identity, source, and active
  behaviors;
- a model assignment attaches the resolved behavior to a region;
- an element formulation states how the fields are discretized.

This separation lets the same physical material participate in a static,
dynamic, thermal, or sequential analysis without copying a material card into
every Step.

## Concise mechanical material

Existing direct constitutive objects remain the shortest route:

```python
model.material(
    constitutive.isotropic_elastic(
        young=210e9,
        poisson=0.3,
        density=7850.0,
    )
)
```

Use a named definition when identity, source, reuse, or multiple physics
roles matter:

```python
steel = materials.define(
    "laboratory steel",
    constitutive.isotropic_elastic(
        young=210e9,
        poisson=0.3,
        density=7850.0,
    ),
    source="project calibration 2026-08",
)

model.material(steel, region=solid)
```

`isotropic` describes the symmetry of the elastic behavior; it is not the
identity of the material.  Likewise J2 plasticity, Chaboche cyclic plasticity,
Neo-Hookean hyperelasticity, and creep are behavior choices rather than names
of physical substances.

## One identity, separate physics roles

```python
alloy = materials.define(
    "alloy specimen A",
    mechanical=mechanical_behavior,
    thermal=thermal_behavior,
    source="reviewed project dataset",
)
```

When registered, `model.material(alloy)` resolves the role required by the
model's Study.  It checks minimum compatibility before assembly: explicit or
implicit structural dynamics requires density; transient heat transfer
requires conductivity and heat capacity.  The material is not chosen by the
Study and the Study does not rewrite its constitutive equation.

```python
report = alloy.compatibility(study)
print(report.as_dict())
```

There is intentionally only one active executable model per role in one
definition.  Choosing between an elastic approximation and a calibrated J2 or
creep model is a scientific modeling decision, not an automatic consequence
of the Step name.

## Packaged reference cards

```python
steel_reference = materials.load_definition("steel_generic")
```

Packaged cards are marked `reference_only=True`.  Generic and template values
support examples and tests; they are not design allowables, certified
industrial parameters, or substitutes for project calibration.  A material
summary retains the card identity, source note, selected model, and unit
system.

The older `materials.load_material(...)` API remains available and returns the
underlying constitutive object directly.  `load_definition(...)` is preferred
when provenance and material identity should remain attached to the model.

## User and private materials

For one project, keep an executable material outside the main case:

```python
# materials/active.py
from agentfem import constitutive, materials

material = materials.define(
    "project alloy",
    constitutive.isotropic_elastic(
        young=210e9,
        poisson=0.3,
        density=7850.0,
    ),
    source="reviewed calibration",
)
```

The model then changes material without changing its analysis structure:

```python
active_material = materials.load("materials/active.py")
model.material(active_material, region=solid)
```

`materials.load(...)` accepts either a packaged card name or an explicit
Python path. A Python asset publishes `material` or a zero-argument
`create_material()` factory. AgentFEM records the absolute source path,
selected symbol, and SHA-256 fingerprint. Loading is explicit because the
file is trusted executable code; downloaded or unreviewed files should not be
loaded merely to inspect them.

Project-owned Python models, future MFront/MGIS adapters, and bounded
UMAT/UHYPER migration bridges should resolve to the same constitutive behavior
boundary.  Confidential material records belong in a separately installed
extension package rather than in a main simulation file or the open core.

AgentFEM does not require TOML for material equations.  TOML may describe an
extension or select an active material asset; executable Python, compiled
libraries, and large temperature/history tables remain in their natural
files and are referenced by an explicit adapter.
