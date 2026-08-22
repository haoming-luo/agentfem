# Public Evidence

This directory retains compact, versioned evidence behind externally reported
AgentFEM results. It is separate from ignored local verification workspaces and
from ordinary test fixtures.

Each evidence bundle contains the raw upstream summaries, a normalized report,
the exact AgentFEM and upstream benchmark revisions, explicit evaluation-mode
metadata, and SHA-256 hashes. Verify the current PDEAgent-Bench snapshot with:

```bash
python tools/freeze_pdeagent_bench_evidence.py \
  --verify evidence/pdeagent_bench/2026-08-22-fixed-adapter
```

The `fixed_adapter` evaluation mode means that the already-written AgentFEM
adapter produced the numerical result. No language model was called during the
evaluation, regardless of labels required by an upstream runner or retained in
its directory layout.
