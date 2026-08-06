# AgentFEM documentation site

AgentFEM documentation uses one knowledge base with progressive disclosure:

1. the home page explains the product and routes each visitor;
2. Start and Guides organize work by user goal and physical problem;
3. Examples provide executable evidence;
4. Reference supports precise lookup;
5. Project preserves trust, release, audit, and design history.

Internal audits remain public and searchable but do not compete with the first
user journey.

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
CI to ensure generated references are current.

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
release snapshots. Ordinary CI only validates the site; publishing a version is
an explicit release action. See [Documentation versions](project/versioning.md).

## Documentation rule

Human-facing and agent-facing material must share the same public concepts.
When a workflow concept changes, update the responsible guide, scientific card,
machine manifest or skill rather than copying an inconsistent explanation into
several unrelated pages.
