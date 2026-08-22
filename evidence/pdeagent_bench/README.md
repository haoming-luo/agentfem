# PDEAgent-Bench Evidence Index

## Current snapshot

[`2026-08-22-fixed-adapter-3d-flow/`](2026-08-22-fixed-adapter-3d-flow/)
is the canonical AgentFEM v2 development snapshot:

- 558/645 final gates passed (**86.5%** case-weighted micro-average);
- **87.3%** unweighted macro-average across all eleven PDE families;
- **65.6%** minimum complete-family pass rate;
- 645/645 cases executed and returned schema-valid output;
- 553/586 two-dimensional cases and 5/59 three-dimensional cases passed.

The bundle pins AgentFEM commit
`5a92b5de8b4953e94f0214e77a20bdb71b9fcaba`, PDEAgent-Bench commit
`0ba9853f82a78196796fa4eeaf0951eb4c000a00`, the adapter and catalog hashes,
all upstream summaries, the normalized report, and every artifact digest.

Verify it without rerunning the numerical cases:

```bash
python tools/freeze_pdeagent_bench_evidence.py \
  --verify evidence/pdeagent_bench/2026-08-22-fixed-adapter-3d-flow
```

This is a fixed scientific-platform development result. Codex
(GPT-5.6-sol) participated in the development workflow; after the source was
frozen, the 645-case numerical execution did not invoke a model per case. It is
not a `gpt-5.1` score and is not currently an official leaderboard entry.

Public discussion with the benchmark maintainers is retained in
[PDEAgent-Bench issue #12](https://github.com/YusanX/pde-agent-bench/issues/12).

## Historical snapshot

[`2026-08-22-fixed-adapter/`](2026-08-22-fixed-adapter/) records the preceding
555/645 snapshot before the reusable three-dimensional block-Stokes upgrade.
It remains immutable development history and must not be quoted as the current
AgentFEM result.
