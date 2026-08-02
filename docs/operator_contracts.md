# Scientific Operator Contracts

## Why the second layer exists

AgentFEM's second layer is the durable finite-element vocabulary between the
readable model and the FEniCSx/UFL/PETSc backend. It should not hide weak forms;
it should name, compose, inspect, and validate them.

The design follows five long-lived qualities emphasized in Cast3M's own
development principles: simplicity, orthogonality of elementary processes,
visibility/locality, regularity, and documentation. AgentFEM expresses those
qualities with typed Python objects and keeps a direct escape hatch to UFL.

## Four system forms

The first release distinguishes:

\[
Kx=F,
\qquad
C\dot{x}+Kx=F,
\qquad
M\ddot{u}+C\dot{u}+Ku=F,
\qquad
R(u)=0,\quad K_t=\frac{\partial R}{\partial u}.
\]

They are represented by `LinearSystem`, `FirstOrderSystem`,
`SecondOrderSystem`, and a residual/tangent pair. This is scientific structure,
not a claim that every system is solved by the same algorithm.

```python
C = operators.capacity_operator(temperature, rho_c)
K = operators.conduction_operator(temperature, conductivity)
Q = operators.heat_source_vector(source, temperature)
system = operators.first_order_system(C, K, Q)
system.check()
```

For a nonlinear problem:

```python
R = operators.residual_operator(residual_form, family="hyperelasticity")
Kt = operators.linearize(R, displacement)
R.check()
Kt.check()
```

UFL performs symbolic differentiation; AgentFEM records that the tangent is
the derivative of a named residual and checks that the residual is a linear
form while its tangent is bilinear.

The release also exposes common orthogonal building blocks that should not be
rewritten case by case: inertial virtual work, prescribed boundary flux,
Robin/convection boundary matrix and environment vector, and Rayleigh damping
`C = alpha M + beta K`. Each remains independently inspectable and composable.

## Operator roles and checks

An `OperatorForm` declares one of five roles:

- `matrix`: bilinear weak form with two arguments;
- `vector`: linear load/history form with one argument;
- `residual`: nonlinear weak equilibrium with one test argument;
- `scalar`: functional with no arguments;
- `operator`: an intentionally opaque or backend-neutral operator.

`operator.validate()` returns stable, addressable `ValidationIssue` objects;
`operator.check()` raises only when those errors exist. Derived operators keep
their operands and operation (`sum`, `scale`, `linearize`) in summaries, so a
human or agent can inspect how `K`, `C`, or `F` was assembled.

## Extension rule

A new public operator is not admitted because a UFL expression can be written.
It needs:

1. a scientific name, role, family, equation, assumptions, and units;
2. a small public constructor and an inspectable summary;
3. role/arity and parameter validation;
4. a unit or finite-element test;
5. a Scientific Function Card and, for numerical claims, benchmark evidence;
6. at least one real consumer in a Step, Problem, Result, or documented case.

This keeps the second layer extensible without turning it into hundreds of
unrelated wrappers.

Reference: [Cast3M presentation and development principles](https://www-cast3m.cea.fr/html/ManuelCastemEnsta/ManuelCastemEnsta.html).
