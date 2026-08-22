# Public Evidence

This directory retains compact, versioned evidence behind externally reported
AgentFEM results. It is separate from ignored local verification workspaces and
from ordinary test fixtures.

Each evidence bundle contains the raw upstream summaries, a normalized report,
the exact AgentFEM and upstream benchmark revisions, explicit evaluation-mode
metadata, and SHA-256 hashes. The current and historical PDEAgent-Bench
snapshots are indexed in [`pdeagent_bench/README.md`](pdeagent_bench/README.md).
Verify the current 558/645 snapshot with:

```bash
python tools/freeze_pdeagent_bench_evidence.py \
  --verify evidence/pdeagent_bench/2026-08-22-fixed-adapter-3d-flow
```

The `fixed_adapter` mode identifies the frozen numerical execution: the runner
loaded the already-written AgentFEM integration instead of invoking a model for
each case. Codex (GPT-5.6-sol) was used in the preceding AgentFEM development
workflow. Runner labels, development tools, and evaluation-time inference are
recorded separately so none is silently presented as another.
