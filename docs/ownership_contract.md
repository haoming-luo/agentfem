# Architecture and ownership

AgentFEM is not a second finite-element kernel. It owns the scientific
boundary between an engineering model and the numerical runtime that executes
it. This page defines that boundary for contributors, extension packages,
GUIs, and AI agents.

## Stable ownership boundaries

| Boundary | Question it answers | Owns | Does not own |
| --- | --- | --- | --- |
| Model | What engineering problem is being solved? | Study, geometry, regions, fields, material assignments, loads, constraints | Newton iteration, time integration, result acceptance |
| Constitutive | How does a material point respond? | Update law, stress, consistent tangent, internal-variable schema | Global equilibrium, mesh traversal, history lifetime |
| State | What accepted and trial history must survive? | Commit, rollback, snapshots, restart, time levels | Material equations, solver choice, output format |
| Operator | What mathematical contribution is assembled? | Residual, tangent, mass, damping, source, composition | Step sequencing, state acceptance, verification |
| Procedure | How is the problem advanced and solved? | Algorithm, increments, provider dispatch, option contract | Engineering intent, backend algebra, scientific acceptance |
| Backend | Which runtime performs finite-element execution? | Compilation, assembly, DOFs, linear and nonlinear algebra | Engineering semantics and verification policy |
| Result / Verification | What was computed and why may it be used? | Fields, histories, provenance, failure, evidence, acceptance | Solver mutation and constitutive evolution |

The same contract is available through:

```bash
agentfem capabilities --json
```

The records under `ownership_contract` are generated from
`agentfem._architecture_contract`; they are not duplicated documentation.
Architecture tests also reject selected cross-layer imports, including a
`Model` that constructs a discrete `problem` directly.

## The public execution path

The recommended route remains:

```text
Study -> Model -> scientific assets -> model.step(...)
      -> StepRequest -> provider -> builder -> problem/backend
      -> SimulationResult -> Verification
```

`Model` is a readable engineering facade, not the owner of every object on
that path. Historical material-specific `*_step()` methods remain thin 0.2.x
compatibility delegates; new workflows use `model.step(...)`.

Provider selection is intentionally narrower than lowering. The private
registry stores, orders, and resolves declared providers without importing
builders or executing them. A selected provider then performs scientific
lowering, after which the dispatch boundary binds the common execution
context. Public provider contracts and dispatch remain in
`step_providers.py`; built-in predicates, lowerers, and declarations live in
`_builtin_step_providers.py`. This keeps extension discovery independent of
built-in physics without inventing a second plugin API.

Built-in builders are divided by scientific family when their dependencies
and validation rules form a genuine independent unit. Linear and thermal
lowering live in `_step_builders_thermal.py`; finite-kinematics hyperelastic,
mixed, and fabric-membrane lowering live in
`_step_builders_finite_strain.py`; stateful inelastic and hereditary lowering
live in `_step_builders_inelastic.py`; frequency- and time-domain dynamics
have their corresponding family modules. `_step_builders.py` is now only the
stable private facade consumed by providers and 0.2.x compatibility methods.
Further splitting inside a family should occur only when ownership evidence
requires it, not to satisfy a line-count target.

The same ownership rule applies after a procedure finishes. Discrete problem
objects may advance state and expose the solution they computed, but private
factories in `agentfem.results` assemble fields, histories, checkpoints,
artifacts, processing metadata, and verification evidence. Static,
incremental nonlinear, affine nonlinear, modal, and transient procedures all
use this boundary. This prevents `problems.py` from becoming a second result
system as new solver families are added.

The same rule applies before a Step is built. Model-first conveniences such as
`model.stiffness(...)`, `model.mass(...)`, `model.conduction(...)`, and
`model.heat_capacity(...)` remain the stable public language, but they pass
the model-owned material assignments to a private operator lowering boundary.
That boundary resolves regional measures, validates physical coefficients,
builds each contribution, and composes partitioned operators. It consumes an
immutable assignment sequence rather than importing `Model`, so the readable
facade does not become the owner of finite-element forms.

## State is a boundary, not one universal algorithm

`agentfem.state` provides two minimal structural protocols:

- `RestartableState`: `snapshot()` and `restore()`;
- `ReplaceableState`: restart plus `commit()` and `rollback()`.

Beginning a trial intentionally remains procedure specific. A material-point
update, a Newton load increment, an ordered fatigue cycle, and a transient
time step need different physical inputs. AgentFEM exposes those differences
rather than hiding them behind one misleading `begin()` signature.

First- and second-order transient states are owned by `agentfem.state`.
`agentfem.problems` retains compatibility aliases, so existing projects do not
need a mechanical migration. The lumped mass operator is similarly owned by
`agentfem.operators` and re-exported from `problems` for compatibility.

Transient procedure implementations are owned by
`_transient_problems.py`: they advance time, enforce accepted-step cadence,
report progress, and coordinate checkpoint/restart. `problems.py` retains the
stable factory functions and compatibility exports, while transient result
assembly remains in `results/_transient_step.py`. This separates discrete
problem descriptions from time-evolution policy without changing user code.

Incremental and affine nonlinear procedures follow the same rule.
`_nonlinear_problems.py` owns load-path advancement, cutback, trial/accepted
state transactions, nonlinear checkpointing, and accepted-increment
snapshots. Shared residual-to-reaction recovery lives in
`_problem_fields.py`. The `problems.py` facade retains direct linear/nonlinear
problem descriptions and stable factory functions.

## FEniCSx-first kernel boundary

DOLFINx owns finite-element spaces, form assembly, degree-of-freedom handling,
and distributed data movement. PETSc owns scalable linear and nonlinear
algebra. AgentFEM should use these capabilities rather than reimplementing
them merely to appear more independent:

- [DOLFINx finite-element API](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html)
- [DOLFINx PETSc assembly and nonlinear problems](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html)
- [PETSc SNES nonlinear solvers](https://petsc.org/main/manual/snes/)

AgentFEM's independence lies in the engineering language, scientific
lowering, state lifecycle, execution evidence, and verification boundary.

Akantu provides a useful contrast: its solid-mechanics model creates and owns
an `FEEngine` for interpolation, integration, and assembly, and its local
materials extend the kernel's `Material` abstraction. AgentFEM adopts the
clarity of that ownership model, not the same ownership boundary, because
DOLFINx/PETSc already provide its production finite-element engine:

- [Akantu model and FEEngine ownership](https://akantu.readthedocs.io/en/latest/manual/models.html)
- [Akantu user-defined material route](https://akantu.readthedocs.io/en/stable/manual/solidmechanicsmodel.html#adding-a-new-constitutive-law)

## Refactoring rule

File size is a maintenance signal, not an architecture test. Split a module
when responsibility or dependency evidence demands it. Do not create a second
abstraction merely to reduce line count, and do not add a second backend until
the current FEniCSx-first boundary can be implemented without special cases in
the core model language.
