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
use a public Taylor--Hood velocity/pressure space; Navier--Stokes starts from a
Stokes predictor and advances with a consistent Newton tangent. Scalar
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

## Verified Development Snapshot

On 21--22 August 2026, the fixed adapter was run with the official v2 runner at
the pinned benchmark commit above. Across complete-family runs and the
recorded periodic/constant-boundary repair reruns, all 645 public cases in all
eleven families produced executable, schema-valid output. The official final
gate passed 552/645 cases: an 85.6% micro-average and an 86.4% unweighted
family macro-average.

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
| Stokes | 34 | 61 | 55.7% |
| Navier--Stokes | 24 | 28 | 85.7% |
| biharmonic | 57 | 57 | 100.0% |
| **micro total** | **552** | **645** | **85.6%** |

The original three-family set passed 170/204 cases (83.3%). Stratification by
the benchmark's declared dimension remains essential:

| Public subset | Passed | Total | Pass rate |
|---|---:|---:|---:|
| all declared 2D cases | 170 | 174 | 97.7% |
| 2D Poisson | 77 | 81 | 95.1% |
| 2D heat | 40 | 40 | 100.0% |
| 2D linear elasticity | 53 | 53 | 100.0% |
| all declared 3D cases | 0 | 30 | 0.0% |

The 3D result is not presented as an AgentFEM capability score. The public
agent view of the scalar 3D manufactured cases is internally inconsistent:
for example, `poisson_3d_smooth_trig` asks the solver to use
`2*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)`, whereas the negative three-dimensional
Laplacian of the stated sinusoidal field is `3*pi**2` times that field. The
heat cases repeat the same dimensional mismatch. AgentFEM solves the exposed
equation and deliberately does not read the withheld manufactured solution.
The remaining 3D elasticity cases use a uniform conservative policy that has
not yet met the official accuracy/time gate. A separate public-equation 3D
test is retained in AgentFEM to verify the actual 3D Poisson path.

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
  (**achieved at 86.4% and 85.6%, respectively**);
- a same-model, same-budget A/B study before claiming a systematic AI-native
  advantage over regenerating backend code for every problem.

The eleven-family milestone makes the next work qualitative rather than a
search for more family names: improve three-dimensional mixed-flow accuracy,
generalize pressure-nullspace and periodic-vector constraints, expose richer
fourth-order boundary pairs, and preserve immutable official-run evidence.
Those changes must improve public numerical contracts and external cases; they
must not become case-ID dispatch or reference-field fitting.
