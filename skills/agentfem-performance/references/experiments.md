<!-- SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experiments that change the engineering decision

Read for performance research, noisy comparisons, interacting optimizations or
new numerical methods. This is guidance, not a mandatory benchmark suite.

## Diagnose the mechanism

Use a representative installed execution route when import selection, native
builds or migration are in question. For a local algorithm prototype, a focused
harness is sufficient if its boundary and exclusions are explicit. Do not make
wheel building or release packaging a prerequisite for every idea.

Separate CPU work, synchronization, interpreter wait, data movement and I/O.
Timers around native calls can include worker join and interpreter reacquisition.
Nested stage percentages cannot be summed; overlapping branches can hide a
local gain. A simplified `outside + max(body, contact)` model is useful only
with its scheduling and contention assumptions stated. Use timelines or a
bounded causal intervention when an observational profile is ambiguous.

Time completed work, not only asynchronous submission. Use completion events
or the actual dependency boundary; inserting global barriers for every stage
can destroy the overlap being measured. Confirm performance without intrusive
instrumentation once it has identified the mechanism.

Account for simultaneously active worker pools and nested BLAS/OpenMP threads,
not each pool's budget in isolation. Test process-local budgets on the actual
machine and workload; more workers need not improve throughput.

A diagnostic ablation can test a mechanism or conditional optimistic bound,
but changing contention, caches or state can invalidate even that interpretation.
It cannot produce valid simulation results if it omits physics. Keep such output
non-deliverable; a reduced diagnostic cost is not a realizable speedup until an
equivalent or accuracy-validated implementation replaces the ablation.

Logical bytes copied do not establish measured DRAM traffic or bandwidth
saturation. Check counters when accessible, or triangulate with controlled
scaling/working-set probes and report what remains unknown. Profiling overhead
and disabled-output runs must be distinguished from ordinary timings.

For interacting changes, compare the baseline, useful component variants and
the combined route as the budget permits. Do not require every component to
win alone before testing a coherent architecture. Likewise, do not infer a
combined gain by multiplying microbenchmark speedups.

For temporal caches, advance representative geometry/state: repeatedly replaying
one frozen input can artificially inflate cache hits. Measure invalidation and
rebuild cost, including cold restart, without calling a local frozen replay a
trajectory benchmark. Local replay remains useful for kernel correctness/cost.

## Keep uncertainty useful, not ceremonial

- Define workload, station, timing boundary, warm-up and runtime identity.
  Startup matters to short jobs; long-run steady-state behavior matters to long
  jobs. Report both when relevant rather than universally discarding startup.
- Alternate or randomize comparable baseline/candidate blocks to reduce drift;
  record order and individual observations. Use the same physical stations for
  evolving problems. Repeated increments in one trajectory are not independent
  repetitions of the same workload.
- When repeated timing changes the decision, repeat at the level causing the
  variation (execution, scheduling, build, etc.). There is no universal repeat
  count, warm-up length, minimum gain percentage or confidence level.
- A single conspicuously poor result may justify withholding this candidate;
  modest/noisy differences remain inconclusive. Never report that result as a
  statistically established slowdown or proof that the method cannot work.
- Separate screening from confirmation: selecting the best of many candidates
  biases its observed gain. Confirm the selected candidate with fresh data;
  do not repeatedly sample until a desired percentage appears.
- If uncertainty is comparable with a small effect and resolving it is costly,
  defer the candidate. Report the uncertainty; do not require a formal confidence
  interval from too few or dependent observations, or invent "90% confidence".
- Do not alter system-wide security, power or scheduling settings merely to
  imitate a published benchmark recipe. Prefer process-local controls. A quiet
  benchmark is not automatically representative of ordinary user operation.

For full-run estimates use representative physical regimes and their expected
work, including restart/cache warm-up and scheduled output. Label assumptions
and uncertainty; do not linearly extrapolate a quiet early or unusually fast
segment into an achieved complete-run time. A numerical-method change may also
change which regimes or events occur, requiring a new model of cost.

## Numerical-method exploration is a separate valid route

Allow adaptive stepping, alternative integration, mixed precision, spatial
adaptivity, different reductions and new backends when in scope. Define error
targets on meaningful quantities and histories, not only final force. For
contact/damage include onset, unloading/recontact, dissipative work and stability
where relevant. Compare common physical stations rather than equal step counts.
Preserve work/energy semantics; internal work is not automatically stored energy.

Replay is a cheap local equivalence tool, not a substitute for trajectory checks
when state evolution changes. A new representation may start from common initial
data instead of inheriting an incompatible checkpoint. Use convergence or an
independent reference appropriate to the claim; do not demand byte identity of
legitimately reordered arithmetic. Diagnose tolerance failures before deciding
whether code or the criterion is wrong, and retain both decisions transparently.

## Primary methodological sources

- [Hoefler and Belli, scientific benchmarking](https://htor.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf):
  interpretable experiments require context and uncertainty, not just one timing.
- [Kalibera and Jones, rigorous benchmarking in reasonable time](https://kar.kent.ac.uk/33611/):
  repetitions should address the sources of variation and their measurement cost.
- [Google Benchmark interleaving](https://google.github.io/benchmark/random_interleaving.html):
  interleaving is one tool against run-order variation, not a guarantee for this solver.
- [Causal profiling](https://www.usenix.org/publications/login/summer2016/curtsinger):
  time spent in a function and potential application speedup are different questions.
- [Berkeley Lab Roofline](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/):
  arithmetic intensity, bandwidth and throughput inform hardware limits; source
  array sizes alone do not measure which limit a real execution reaches.
- [PETSc profiling](https://petsc.org/release/manual/profiling/):
  stages, imbalance and instrumentation overhead matter to interpretation.

These sources inform experiment design; none demonstrates an AgentFEM speedup.
