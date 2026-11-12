---
name: agentfem-performance
description: Improve AgentFEM runtime and performance portability through profiling, algorithm and architecture experiments, and proportional evidence. Use for simulation bottlenecks, acceleration research, fast-path migration and reproducible case delivery.
license: Apache-2.0
---

# AgentFEM performance engineering

<!-- SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This is revisable development guidance—not a solver, runtime
dependency, architecture mandate or proof of correctness. A project must run
without an AI supplying missing executable behavior. Follow the user's scope
and repository instructions; explain when a local convention no longer fits.

Read AGENTS.md and the relevant existing evidence. Inspect numerical processes
before launching work; preserve private assets and unrelated changes. Reuse
trustworthy measurements rather than repeatedly rebuilding the same baseline.

## Select the work, not a ritual

Start with the requested outcome and time budget. Optimization-only work does
not require a delivery bundle, release checklist or full simulation. Read only
the references needed for the current decision:

- [Experiments](references/experiments.md): noisy timings, concurrent execution,
  architectural hypotheses, or a new numerical method.
- [Migration parity](references/migration-parity.md): private-to-core execution
  migration or an unexplained regression after migration.
- [Delivery](references/delivery.md): installed release, project portability,
  checkpoint/export changes, or a completed-run performance claim.

Identify the actual route and the dominant source of elapsed time: increment
count, cost per increment, rejected work, initialization, or output. In parallel
paths distinguish work from waits, exclusive from inclusive timing, and the
critical path from the sum of overlapping timers. A costly function is not
necessarily the place where an optimization will shorten the solve.

For the leading hypothesis, state compactly: the mechanism, supporting evidence,
plausible end-to-end benefit and assumptions, cheapest discriminating experiment,
and what would change the decision. Unknown benefit is a reason for a bounded
probe, not an automatic rejection. Do not invent a success probability.

Consider work reduction and execution/dataflow redesign before polishing a
minor kernel. Large changes may need a coherent prototype: two overlapping
branches can require improvement together. A component with no standalone gain
may still enable the combined design; verify that dependency rather than using
it to justify indefinite complexity. Do not add isolated speedup percentages.

## Choose the comparison that fits the change

| Change | Appropriate comparison |
| --- | --- |
| Equivalent execution | Same scientific inputs and relevant accepted state; compare affected fields, history, work and failures. Byte equality is optional diagnostic evidence when arithmetic and environment support it. |
| New algorithm, precision or discretization | Common physical problem, declared accuracy targets and physical stations; assess convergence, events and path-dependent history. Different steps, meshes and operation order are allowed. |
| Hardware/backend/scheduling | Reselect operational budgets; report capabilities and runtime provenance. Do not make one machine's worker count a scientific requirement. |

An intentionally altered mass, penalty, contact scope or loading history is a
separate scientific approximation study, not an unchanged-model speedup. It may
be researched within scope, but must not silently replace the production model.
Stable execution alone does not establish adequate accuracy. Numerical parity
with an old implementation does not independently validate its physical model.

If a checkpoint rejects a changed implementation, do not bypass its identity.
Use a read-only replay with validated input or a common initial problem when
appropriate, and separately design versioned compatibility. An overly broad
identity contract must not become a permanent ban on operational tuning.

## Scale evidence to the decision

1. Explore: inspect code, model the cost, or run a bounded diagnostic/prototype.
   Check the scientific boundary the experiment touches before trusting its
   output. A clear correctness defect can reject this implementation immediately;
   one noisy timing cannot establish that an algorithmic direction is useless.
2. Select: exercise representative states and affected failure paths. Use enough
   timing evidence to distinguish a useful effect from uncertainty. Classify the
   result as supported, unfavorable for the tested scope, or inconclusive.
   Inconclusive candidates need not ship or receive endless repetitions.
3. Promote: test the selected combined route and its relevant installed use.
   Full-run timing is needed for a full-run claim, not every exploratory edit.
   Reuse unaffected evidence with its tested identity and scope stated.

Stop a line when its plausible payoff is too small for the task, its measured
mechanism fails, or its complexity is unjustified. Preserve a compact finding
including the tested scope and conditions for reconsideration. A failed local
implementation does not disprove fusion, caching, parallelism or another entire
family of methods. A cheap repeatable small improvement can be worthwhile, but
must not displace the user's major bottleneck indefinitely.

Choose scientific tolerances from physical/numerical reasoning, not the desired
speedup. If an inherited gate is demonstrably inappropriate, document the old
failure, justify a revised criterion independently, then confirm it on fresh
evidence. Do not silently relabel a failed comparison as passing.

## Preserve invariants, not historical implementation boundaries

Model expresses the problem; Operator contributes physics; Procedure controls
execution; State owns acceptance/rollback; Result exposes evidence. Their
contracts do not require a separate array copy, scan, Python call or traversal
at every boundary. Replace redundant work when ownership and invalidation prove
the same invariant. A preferred immutable-buffer implementation is not the only
valid transaction design.

At promotion, reusable execution belongs in the owning library/backend, not
duplicated solver logic in case.py. Keep confidential material and geometry
outside public core; a local prototype does not require premature generalization.

Readers must see valid state and failed trials must not corrupt accepted state.
Acceptance awaits successful completion of its dependencies; buffers cannot be
released/reused while another operation owns them. Completed immutable chunks
may feed downstream work before global acceptance when dependencies and failure
containment are explicit: safety does not mandate a barrier after every stage.
Cached geometric candidates are not cached forces; justify coverage, current
geometry and invalidation. Preserve required diagnostics and material semantics. Trusted
native callbacks are executable code, not passive model data. Unsupported
providers need an explicit fallback or capability rejection, not silent misuse.

Retain only decision-relevant evidence and reconstructable references; ask
before destructive cleanup. Do not accumulate full archives and delivery bundles
for every failed experiment. End a sprint with an implemented improvement,
a discriminating result, or a narrowed uncertainty—not another unchanged plan.
