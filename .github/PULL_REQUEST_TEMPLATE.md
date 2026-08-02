## Purpose

Describe the engineering problem or developer need this change addresses.

## Public workflow

Show the user-facing code before and after, or state why no public API changes.

## Scientific and software evidence

- [ ] Relevant unit, symbolic, benchmark, and misuse tests are included.
- [ ] Governing equations, assumptions, units, and validity limits are documented when applicable.
- [ ] The maturity claim matches the evidence; no unsupported industrial or universal claim is added.
- [ ] MPI-sensitive behavior was checked with at least two ranks, or is not applicable.
- [ ] Optional dependencies remain lazily imported with an actionable installation message.

List the tests and benchmarks run:

```text
python -m pytest -q ...
```

## Knowledge and documentation

- [ ] Public documentation and examples are updated, or no update is needed.
- [ ] A knowledge card and benchmark contract are updated when scientific behavior changes.
- [ ] Failure modes and downstream consumers are recorded.
- [ ] Generated knowledge and documentation artifacts are current.

## Compatibility

State supported platforms, backend assumptions, checkpoint/schema effects, and migration notes.

## Authorship

- [ ] Commits include a `Signed-off-by` line under the project DCO rule.
