# AgentFEM documentation site

AgentFEM documentation is a scientific-software manual, not a product landing
page. Its first responsibility is to let an engineer install the software, run
a model, inspect the equations and assumptions, find an output definition, and
reproduce an example.

## Information architecture

The public manual follows a stable progression:

1. **Introduction** states the software scope and includes one complete model.
2. **Getting Started** covers installation, project creation, execution, and
   files produced by a run.
3. **User Guide** is organized by analysis procedure, model definition, and
   result workflow rather than Python module name.
4. **Examples** provides executable cases and their numerical maturity.
5. **Theory and Reference** contains equations, conventions, output variables,
   scientific function cards, interoperability, and the Python API.
6. **Extending AgentFEM** supports contributors, agent/GUI integration, and
   custom scientific components.
7. **Project** preserves the current release, trust policy, compatibility, and
   roadmap without competing with the primary user journey. Historical release
   notes and engineering records remain directly linkable but stay outside the
   main sidebar.

This structure is informed by mature engineering software documentation:
Gmsh separates overview, tutorials, scripting/API, options, and file formats;
Abaqus separates analysis, constraints, elements, materials, output, theory,
benchmarks, and verification; scientific Python manuals combine a practical
user guide with a precise API reference.

## Visual contract

The site uses Material for MkDocs as a reliable documentation engine, but its
visual language is deliberately that of a reference manual:

- persistent left-hand document tree;
- readable central text column and right-hand page contents;
- restrained header, typography, color, borders, and status labels;
- code, equations, tables, and result definitions as primary visual objects;
- no full-width product hero, marketing card grid, or hidden home navigation;
- usable narrow-screen navigation and horizontally scrollable technical tables;
- print styles that remove site navigation and preserve the technical page.

The project logo identifies the manual; it is not used as a decorative hero.

## Page and navigation boundaries

Pages are divided by the question a reader is trying to answer, not by an
arbitrary word count. A tutorial remains one continuous page when its steps
form one executable workflow; a reference page is split when unrelated
domains, an unscannable contents list, or slow rendering makes lookup harder.

The primary navigation therefore exposes short task-oriented labels and groups
theory separately from function and API lookup. Old releases, internal audits,
and design records remain searchable and linkable without occupying the main
reader path. The generated scientific-function and Python API references retain
stable single-page indexes because tests, knowledge cards, and external links
address their entries directly. When those indexes are split by domain, the
original anchors must remain available through stable redirects.

The sidebar uses one visual grammar throughout: chapter rows are clickable and
expandable, optional groups use the same chevron convention, and ordinary page
rows remain regular weight. Only the active branch is expanded, so a new reader
sees the manual structure before its details. Section index pages make the
chapter title itself the landing page; redundant `Overview` rows are not shown.
The desktop root label is hidden, while the mobile drawer retains its project
header and home control.

Footer previous/next links provide sequential reading. Numbered pagination is
not added to ordinary manual pages because it hides context and makes technical
search and citation less predictable.

## Scientific page contract

A mature material, analysis procedure, element, load, constraint, or output
page should state, where applicable:

1. purpose and engineering applicability;
2. governing equation and variable definitions;
3. dimensional, kinematic, and constitutive assumptions;
4. required inputs, units, and defaults;
5. discretization, integration, and solution algorithm;
6. available output variables and their locations;
7. supported and incompatible combinations;
8. tests, external benchmarks, primary references, and known limitations;
9. a minimal public-API example.

Beginner tutorials may defer derivations, but they must link to the responsible
theory or scientific-reference page. Formula-rich reference pages should not
force beginner pages to repeat the same equations.

## Build contract

The site uses Material for MkDocs. `build_docs.py` is the canonical build entry:

```bash
python build_docs.py
```

It performs three tasks before the MkDocs build:

- generates the Python API index from declared public workflow modules;
- refreshes `/llms.txt` and `/agentfem.json` for AI-agent discovery;
- synchronizes the reviewed project logo into the documentation assets.

The generated site is written to `site/`. Use `python build_docs.py --check` in
CI to ensure generated references are current. `mkdocs build --strict` checks
navigation and internal links. Mathematical notation is rendered through
Arithmatex and a pinned, self-hosted MathJax runtime from ordinary Markdown
source. Keeping the runtime and its fonts in the documentation artifact avoids
a network-dependent first render while preserving instant navigation.

## Local preview

```bash
python build_docs.py
mkdocs serve
```

Install the optional documentation tools with:

```bash
python -m pip install -e '.[docs]'
```

## Version policy

Material's Mike provider separates development documentation from immutable
release snapshots. Ordinary CI validates the site; publishing a version is an
explicit release action. See [Documentation versions](project/versioning.md).

## Documentation rule

Human-facing and agent-facing material must share the same public concepts.
When a workflow concept changes, update the responsible guide, scientific card,
machine manifest, or skill rather than copying an inconsistent explanation
into several unrelated pages.
