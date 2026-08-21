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

The initial supported families are Poisson, heat, linear elasticity, and
Helmholtz. General geometry specifications are lowered through
`mesh.from_spec(...)`; structured unit domains remain independent of Gmsh,
while general planar domains use the optional Gmsh integration.

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

## Official Runner

Clone and check out the pinned benchmark revision, install AgentFEM in a
compatible FEniCSx environment, and pass the fixed entry point to the official
runner:

```bash
python scripts/run_benchmark.py \
  --agent gpt-5.1 \
  --solver-path /path/to/agentfem/tools/pdeagent_bench_solver.py \
  --equation-types poisson heat linear_elasticity \
  --version v2 \
  --output /path/to/evidence
```

The `--agent` value satisfies the current runner's CLI validation; no model is
called when `--solver-path` is supplied. Record that distinction in any report.

Normalize an official summary into AgentFEM's stable failure taxonomy with:

```bash
python tools/pdeagent_bench_report.py \
  /path/to/evidence/gpt-5.1/dolfinx/summary.json \
  --case-catalog /path/to/evidence/gpt-5.1/dolfinx \
  --json report.json \
  --markdown report.md
```

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

On 21 August 2026, the fixed adapter was run with the official v2 runner at
the pinned benchmark commit above. It executed all 204 Poisson, heat, and
linear-elasticity cases without schema, geometry, solver, output, or process
failure. The official final gate passed 170/204 cases (83.3%). Stratification
by the benchmark's declared dimension is essential:

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
  mixed-field semantics implemented in the core rather than in case scripts;
- all eleven families: at least 60% macro- and micro-average;
- a same-model, same-budget A/B study before claiming a systematic AI-native
  advantage over regenerating backend code for every problem.

The next family sequence is Helmholtz and wave, followed by
convection--diffusion and reaction--diffusion, then Stokes. Burgers,
Navier--Stokes, and biharmonic problems remain later because they require
nonlinear transport, mixed pressure spaces, stabilization, or fourth-order
formulations that deserve first-class AgentFEM contracts.
