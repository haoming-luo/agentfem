# Lower exact MPCs through the formal linear Step

## Decision

The ordinary linear-static, steady-heat, and constant-property transient-heat
`model.step(...)` providers may
consume exactly one reviewed exact-MPC constraint. `LinearSystemProblem`
separates strong Dirichlet data from the MPC provider before assembly, then
dispatches to the shared solver lifecycle. The Model remains an engineering
registry and provider dispatcher; it does not own MPC assembly.

An explicit `constraints=` argument is included in preflight validation. One
unknown provider or more than one exact-MPC provider fails before assembly.

## Reason

AgentFEM already exposed rectangular exact constraint construction and a
reusable numerical lifecycle, but the formal engineering Step still treated
every constraint as a Dirichlet wrapper. This made the advertised public
grammar incomplete and allowed validation and construction to disagree.

There are genuinely two algebraic contracts. Strong data use ordinary
Dirichlet elimination; periodic relations change the admissible space through
exact multi-point elimination. Naming that distinction in the problem layer
keeps ownership stable and prevents adapters from rebuilding solver glue.

## Evidence boundary

Exact elimination proves neither a physical reaction distribution nor
macroscopic constraint work. Until the active provider publishes real dual
evidence and its conjugate coordinate, `SimulationResult` records force and
work balance as unavailable. It never relabels a reduced residual as a
complete engineering balance.

## Verification

- vector linear-static, scalar steady-heat, and two-increment transient-heat
  Steps reproduce constant fields;
- both routes execute through ordinary `model.step(...)` in serial and MPI;
- transient runs retain one prepared operator lifecycle across accepted steps;
- the Result identifies the exact provider and retains solver convergence;
- force/work evidence remains fail-closed without a provider dual;
- multiple exact providers fail before matrix assembly;
- late Dirichlet/MPC overlap remains rejected by the shared solver lifecycle.
