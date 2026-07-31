# AgentFEM Documentation Site

AgentFEM documentation is written as linked Markdown files. The current
recommended static-site layer is MkDocs because it is simple, friendly to
research software, and easy for agents to inspect.

MkDocs builds the engineering design pages under `docs/`. The dependency-free
`build_docs.py` builder additionally collects the repository README,
installation/workflow/concept guides, examples, and project skill into the
complete local site.

## Local Preview

From the repository root:

```bash
cd agentfem
mkdocs serve
```

Then open the local address printed by MkDocs.

## Static Build

```bash
cd agentfem
mkdocs build
```

The generated site is written to `agentfem/site/`.

## Dependencies

For a polished local site:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
```

For a dependency-free static site:

```bash
python build_docs.py
```

This writes a simple HTML site to `agentfem/site/`.

## Documentation Rule

Human-facing docs and agent-facing docs should share the same concepts. When a
workflow concept changes, update:

- `README.md`
- `WORKFLOW.md`
- `CONCEPTS.md`
- `docs/module_map.md`
- `skills/agentfem/`
