# Licensing

AgentFEM is licensed under the Apache License, Version 2.0.

This means the open-source core can be used, modified, redistributed, and used
commercially, provided the license conditions are followed. The license also
defines explicit patent terms and requires preservation of copyright,
attribution, license, and notice information.

## Why Apache-2.0

AgentFEM aims to be useful in research, education, industrial prototyping, and
agent-oriented CAE workflows. Apache-2.0 is a good fit because it is accepted by
the scientific Python ecosystem, compatible with conda-forge packaging, and
clear enough for institutional and company review.

## Project Strategy

The intended structure is:

- AgentFEM open-source core: Apache-2.0.
- Optional commercial services: support, consulting, training, and validation.
- Optional proprietary products: hosted workflows, GUI tools, industrial
  workflow packs, curated databases, and private plugins.

The open-source license cannot be withdrawn from versions that have already
been released. Future separate products or modules can use different licensing
when they are kept outside the Apache-2.0 core.

## Optional dependencies

The Apache-2.0 license describes AgentFEM's own source and release artifacts;
it does not relicense third-party packages. In particular, Gmsh is distributed
separately under GPL-2.0-or-later with the exception published by the Gmsh
authors.

AgentFEM therefore treats Gmsh as an optional adapter:

- the `agentfem` wheel and source distribution do not contain Gmsh;
- Gmsh is absent from AgentFEM's core dependencies;
- `agentfem[gmsh]` requests the separately distributed package only for users
  who need direct Gmsh model or `.msh` import;
- structured DOLFINx meshes, XDMF, and meshio-based Abaqus/NASTRAN conversion
  do not require Gmsh.

Optional separation makes the product boundary and installation choice clear;
it is not a substitute for legal review when another distributor bundles a
combined software system. See the [official Gmsh licensing page](https://gmsh.info/#Licensing)
and [license text](https://gmsh.info/LICENSE.txt).

## Contributions

Unless explicitly agreed otherwise in writing, contributions are accepted under
Apache-2.0. See `CONTRIBUTING.md` for contribution expectations.
