# Extension packages and private products

AgentFEM's Apache-2.0 core and an organization-specific product should share a
public scientific contract without sharing confidential data or source code.
The extension boundary is intended for validated workflow packs, private
material libraries, alternative backends, and domain products that remain in
their own repositories and Python distributions.

## The package boundary

An extension is an independently installed package advertising one standard
Python entry point. AgentFEM can inspect its package name and version without
importing its code. Activation occurs only when a user requests it or a project
declares it as required.

```toml
# company-agentfem/pyproject.toml
[project]
name = "company-agentfem-solids"
version = "1.0.0"
dependencies = ["agentfem>=0.2,<0.3"]

[project.entry-points."agentfem.extensions"]
company-solids = "company_agentfem:extension"
```

The private package exposes a small registration object:

```python
# company_agentfem/__init__.py
from agentfem import extensions


def register(context):
    context.add_step_provider(company_step_provider)
    context.add_material("company-alloy-a", private_material_record)


extension = extensions.Extension(
    spec=extensions.ExtensionSpec(
        name="company-solids",
        version="1.0.0",
        api_version=extensions.EXTENSION_API_VERSION,
        capabilities=("company.creep-assessment",),
    ),
    register=register,
)
```

The extension package owns `company_step_provider` and
`private_material_record`. They are not copied into the open repository or the
AgentFEM wheel.

## Project declaration

A project can make the dependency operationally explicit:

```toml
[project]
name = "hot-component-assessment"
entrypoint = "case.py"
schema_version = "0.2.0"

[extensions]
required = ["company-solids"]
```

`agentfem check` reports a missing package without importing third-party code.
`agentfem run` activates required extensions before executing `case.py`.
Every execution record preserves the extension name, extension version,
distribution name, distribution version, declared capabilities, and registered
assets. This makes a result reproducible without publishing confidential
parameters.

Installed packages can also be inspected or activated directly:

```bash
agentfem extensions
agentfem extensions --json
agentfem extensions --load company-solids
```

## Safety and ownership rules

- Discovery is lazy. Merely importing AgentFEM does not execute installed
  extension code.
- Activation is explicit. Installing an unrelated package cannot silently
  replace a solver or material.
- Registrations are staged and checked for name conflicts before publication.
- Replacing a core registration requires an explicit `replace=True` decision.
- An extension API version mismatch fails before registration.
- AgentFEM records extension identity, but it does not certify third-party
  scientific validity.

Installed extensions are trusted executable Python packages. They should be
reviewed and distributed with the same care as any solver or compiled material
library.

## What belongs where

| Open AgentFEM core | Separate extension or product |
| --- | --- |
| General FEM concepts and public contracts | Organization-specific workflow policy |
| Reference constitutive equations | Confidential calibrated parameters |
| Synthetic and publishable benchmarks | Proprietary component meshes and service data |
| Common result/provenance schema | Certified report templates and approval workflow |
| Extension API and compatibility checks | GUI, hosted service, licensing and account system |

The private product should depend on released AgentFEM versions. It should not
be maintained as a long-lived private branch of the open repository: a branch
shares history and is easy to merge or publish accidentally, while a separate
repository has an independent license, release cycle, access policy, and CI.

## Compatibility policy

The entry-point name is the stable operational identifier. The private package
should pin an AgentFEM version range and test its extension against the oldest
and newest supported core versions. A future incompatible registration API
increments `EXTENSION_API_VERSION`; it must not be guessed from the package
version.

This boundary is deliberately smaller than a general arbitrary-hook system.
Additional registration kinds should enter the context only when they have a
stable public consumer and conflict semantics.
