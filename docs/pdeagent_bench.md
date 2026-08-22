# PDEAgent-Bench Integration

AgentFEM maintains a fixed-solver integration for
[PDEAgent-Bench](https://github.com/YusanX/pde-agent-bench). The integration
answers a narrow question before any model-generation experiment is claimed:

> Given only the case information exposed to an AI agent, how much of the
> benchmark can the AgentFEM scientific workflow solve accurately and within
> the official time gate?

The current adapter is pinned to benchmark commit
`0ba9853f82a78196796fa4eeaf0951eb4c000a00`. This matters because the public
README, paper, dataset, and runner have changed at different rates. Results
without a benchmark commit and AgentFEM source identity are not comparable.

## Boundaries of the Adapter

The adapter consumes the official agent view only:

- PDE type, coefficients, source, and time data;
- domain geometry and boundary conditions;
- required output grid;
- public evaluation configuration and agent knobs.

It does **not** consume the oracle mesh, oracle element order, oracle solver,
manufactured solution, or case-specific hidden reference. The numerical policy
uses exposed dimension, geometry scale, and expression bandwidth; it does not
dispatch on case identifiers.

The supported families are Poisson, heat, linear elasticity, Helmholtz,
convection--diffusion, reaction--diffusion, scalar wave, Burgers, Stokes,
Navier--Stokes, and biharmonic equations. General geometry specifications are
lowered through `mesh.from_spec(...)`; structured unit domains remain
independent of Gmsh, while general planar domains use the optional Gmsh
integration. Stabilized transport, named reaction laws, mixed velocity--
pressure fields, incompressible-flow operators, and fourth-order split
operators are public AgentFEM capabilities rather than benchmark-local code.

```python
from agentfem.integrations.pdeagent_bench import solve_case

answer = solve_case(public_case_spec)
u_grid = answer["u"]
evidence = answer["solver_info"]
```

`u_grid` follows the official `(ny, nx)` or `(nz, ny, nx)` array order.
Irregular-domain points outside the mesh are `NaN`; the adapter also reports
finite coverage instead of exploiting missing samples to reduce the comparison
domain.

The added scalar contracts retain their physical equations explicitly:

\[
\partial_t u-\nabla\!\cdot(\varepsilon\nabla u)
  +\boldsymbol{\beta}\!\cdot\nabla u=f,
\qquad
\partial_t u-\nabla\!\cdot(\varepsilon\nabla u)+r(u)=f,
\]

and

\[
\partial_{tt}u-c^2\Delta u=f.
\]

The nonlinear transport and incompressible-flow contracts add

\[
\partial_t u+u\,\boldsymbol{1}\!\cdot\!\nabla u-\nu\Delta u=f,
\]

\[
-\nu\Delta\boldsymbol{u}+\nabla p=\boldsymbol{f},
\qquad \nabla\!\cdot\!\boldsymbol{u}=0,
\]

and the steady Navier--Stokes momentum convection
\((\boldsymbol{u}\!\cdot\!\nabla)\boldsymbol{u}\). Stokes and Navier--Stokes
use public Taylor--Hood velocity/pressure spaces. Three-dimensional structured
Stokes cases use a Q3/Q2 block formulation, an explicit constant-pressure
nullspace, and a viscosity-scaled velocity-Laplacian/pressure-mass
preconditioner; this avoids treating a saddle-point system as an ordinary
scalar elliptic solve. Navier--Stokes starts from a Stokes predictor and
advances with a consistent Newton tangent. Scalar
periodic Burgers cases use matching-face MPC constraints when the optional
`dolfinx_mpc` integration is installed.

For \(\Delta^2u=f\), AgentFEM exposes the mixed split
\(w=-\Delta u\), \(-\Delta w=f\). When the public case supplies only
\(u=g\) on the boundary, the adapter closes the otherwise under-specified
fourth-order problem with \(w=-\Delta g\). This modeling choice is recorded in
`solver_info`; it is derived from the public boundary expression and never
from a withheld manufactured field.

Steady advection--diffusion can request SUPG stabilization through the public
transport operators. Reaction--diffusion uses backward Euler, with
Crank--Nicolson available for linear reactions; the wave path uses the
average-acceleration Newmark method. Time-step count, scheme, nonlinear
iterations, solver convergence, sampled coverage, and wall time remain in
`solver_info` rather than being inferred from a successful process exit.

## Official Runner

Clone and check out the pinned benchmark revision, install AgentFEM in a
compatible FEniCSx environment, and pass the fixed entry point to the official
runner:

```bash
# `gpt-5.1` is a runner-compatibility label in solver-path mode.
python scripts/run_benchmark.py \
  --agent gpt-5.1 \
  --solver-path /path/to/agentfem/tools/pdeagent_bench_solver.py \
  --equation-types poisson heat linear_elasticity helmholtz \
    convection_diffusion reaction_diffusion wave burgers stokes \
    navier_stokes biharmonic \
  --version v2 \
  --output /path/to/evidence
```

The literal `gpt-5.1` above is only a legacy runner-compatible directory label.
No GPT-5.1 call occurs when `--solver-path` is supplied, and the label must not
be reported as the evaluated model. The adapter was developed collaboratively
with Codex (GPT-5.6-sol); the numerical result itself is produced by loading
the already-fixed AgentFEM source. Public reports should therefore identify
this track as **AgentFEM fixed adapter**, not as a GPT-5.1 score.

Normalize an official summary into AgentFEM's stable failure taxonomy with:

```bash
python tools/pdeagent_bench_report.py \
  /path/to/evidence/gpt-5.1/dolfinx/summary.json \
  --case-catalog /path/to/evidence/gpt-5.1/dolfinx \
  --json report.json \
  --markdown report.md
```

Disjoint family runs may be passed as multiple positional summaries. The
reporter rejects duplicate case IDs rather than silently counting a rerun
twice, and records every source path in the JSON evidence.

Failures are classified as schema, geometry, solver, output, accuracy, time,
or execution failures. A successful execution is not silently promoted to a
scientific pass.

## Three Separate Claims

Keep these evaluation lines separate:

1. **Fixed AgentFEM numerical capability** — this adapter and the official
   evaluator, without model generation.
2. **AgentFEM assistance effect** — the same AI model and budget, comparing
   direct backend code with AgentFEM-authored workflows.
3. **Official library track** — a maintained environment, prompt, harness,
   output adapter, and representative tests agreed with benchmark maintainers.

A fixed-adapter pass rate demonstrates numerical and workflow coverage. It
does not by itself measure an AI model or prove that AgentFEM improves code
generation. The second claim requires a controlled A/B experiment.

## Maintainer Engagement and Submission

The public leaderboard currently describes model and code-agent runs, not a
pre-existing scientific platform executed through `--solver-path`. AgentFEM
therefore must not upload the fixed-adapter snapshot under the legacy
`gpt-5.1` runner label. The responsible sequence is:

1. publish the pinned, hash-verifiable fixed-adapter evidence;
2. contact the benchmark maintainers with the exact evaluation distinction;
3. propose either a separate **scientific platform/fixed adapter** track or an
   official AgentFEM library integration;
4. submit a leaderboard artifact only after its category and timing policy are
   agreed;
5. run the separate same-model, same-budget agent A/B study for any claim about
   AI assistance.

Two open upstream protocol questions materially affect interpretation:
[runtime scope](https://github.com/YusanX/pde-agent-bench/issues/10) and
[the v2 accuracy multiplier](https://github.com/YusanX/pde-agent-bench/issues/11).
Near-zero reference fields also need a scale-aware absolute-error fallback;
relative L2 error is undefined at exactly zero and numerically unstable for a
reference consisting only of discretization residual.

AgentFEM reported the fixed-platform result and the proposed broader evaluation
scope to the maintainers in
[PDEAgent-Bench issue #12](https://github.com/YusanX/pde-agent-bench/issues/12).
The opportunity is larger than one extra leaderboard row: PDEAgent-Bench can
also become an international reference for collaboration between AI agents and
scientific-computing platforms.

## Verified Development Snapshot

On 21--22 August 2026, the fixed adapter was run with the official v2 runner at
the pinned benchmark commit above. Across complete-family runs and the
recorded periodic/constant-boundary repair reruns, all 645 public cases in all
eleven families produced executable, schema-valid output. The official final
gate passed 558/645 cases: an 86.5% micro-average and an 87.3% unweighted
family macro-average. Every family exceeded a 60% final pass rate.

| Complete public family | Passed | Total | Pass rate |
|---|---:|---:|---:|
| Poisson | 77 | 91 | 84.6% |
| heat | 40 | 50 | 80.0% |
| linear elasticity | 53 | 63 | 84.1% |
| Helmholtz | 52 | 62 | 83.9% |
| convection--diffusion | 68 | 84 | 81.0% |
| reaction--diffusion | 64 | 64 | 100.0% |
| scalar wave | 42 | 42 | 100.0% |
| Burgers | 41 | 43 | 95.3% |
| Stokes | 40 | 61 | 65.6% |
| Navier--Stokes | 24 | 28 | 85.7% |
| biharmonic | 57 | 57 | 100.0% |
| **micro total** | **558** | **645** | **86.5%** |

The unweighted family macro-average is **87.3%**, and the minimum complete-
family pass rate is **65.6%**. The reported execution host was a 14-inch Apple
M5 MacBook Pro with 32 GB unified memory and macOS arm64, using one MPI rank in
CPU mode. Thread-count environment variables were not pinned, so this result
must not be described as a controlled single-thread measurement.

Within the nine declared 3D Stokes cases, the new block route passes 4/9,
compared with 1/9 for the previous monolithic route, while reducing the
representative solve time by roughly one half. It passes four manufactured
velocity fields and substantially reduces the error in two further cases.
Two remaining cases have analytically zero velocity because a constant body
force is absorbed by pressure under complete no-slip boundaries; their
near-zero numerical references make a purely relative L2 comparison unstable.
The other strict failures use the legacy 0.05 oracle-error multiplier tracked
in the upstream protocol issue above.

Stratification by the benchmark's declared dimension remains essential:

| Public subset | Passed | Total | Pass rate |
|---|---:|---:|---:|
| all declared 2D cases | 553 | 586 | 94.4% |
| all declared 3D cases | 5 | 59 | 8.5% |
| 3D Stokes | 4 | 9 | 44.4% |
| 3D Helmholtz | 1 | 10 | 10.0% |
| other declared 3D families | 0 | 40 | 0.0% |

Three-dimensional coverage is therefore the clearest remaining numerical gap,
not something hidden by the aggregate score. The public agent view of several
scalar 3D manufactured cases is also internally inconsistent:
for example, `poisson_3d_smooth_trig` asks the solver to use
`2*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)`, whereas the negative three-dimensional
Laplacian of the stated sinusoidal field is `3*pi**2` times that field. The
heat cases repeat the same dimensional mismatch. AgentFEM solves the exposed
equation and deliberately does not read the withheld manufactured solution.
The remaining 3D elasticity and transport routes use uniform public policies
that have not yet met the official accuracy/time gate. Separate public-
equation tests remain in AgentFEM so benchmark-reference inconsistencies do
not replace ordinary numerical verification.

These numbers are a local fixed-adapter development result, not an official
PDEAgent-Bench leaderboard submission and not a measurement of `gpt-5.1`.
The agent name only satisfies the current runner interface when
`--solver-path` is active.

## Expansion Gates

The staged targets are:

- Poisson, heat, and linear elasticity: at least 90% combined and 85% in each
  family before calling them the trusted set;
- six to eight families: at least 75% macro-average, with stabilization and
  mixed-field semantics implemented in the core rather than in case scripts
  (**achieved for seven complete families in the snapshot above**);
- all eleven families: at least 60% macro- and micro-average
  and at least 60% in every individual family
  (**achieved at 87.3%, 86.5%, and a 65.6% minimum family rate**);
- a same-model, same-budget A/B study before claiming a systematic AI-native
  advantage over regenerating backend code for every problem.

The eleven-family milestone makes the next work qualitative rather than a
search for more family names. The first three-dimensional mixed-flow upgrade
now provides a block Q3/Q2 route with pressure-nullspace and viscosity-scaled
preconditioning. Remaining work is to converge that public route under a
clarified timing gate, generalize periodic-vector constraints, expose richer
fourth-order boundary pairs, and preserve immutable official-run evidence.

After a fixed-adapter run, freeze the disjoint official summaries, normalized
report, source identity, benchmark revision, and SHA-256 hashes with:

```bash
python tools/freeze_pdeagent_bench_evidence.py \
  /path/to/family-1/summary.json /path/to/family-2/summary.json ... \
  --catalog /path/to/pde-agent-bench/data/benchmark_v2.jsonl \
  --output evidence/pdeagent_bench/YYYY-MM-DD-fixed-adapter
```

The resulting manifest states explicitly that solver-path evaluation calls no
model; the upstream `gpt-5.1` directory label is retained only for byte-level
traceability to the runner output.

Verify a committed bundle without rerunning the numerical cases:

```bash
python tools/freeze_pdeagent_bench_evidence.py \
  --verify evidence/pdeagent_bench/2026-08-22-fixed-adapter-3d-flow
```

The verified 558/645 development snapshot is retained in the repository at
[`evidence/pdeagent_bench/2026-08-22-fixed-adapter-3d-flow`](https://github.com/haoming-luo/agentfem/tree/main/evidence/pdeagent_bench/2026-08-22-fixed-adapter-3d-flow).
Those changes must improve public numerical contracts and external cases; they
must not become case-ID dispatch or reference-field fitting.
