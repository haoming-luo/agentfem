# AgentFEM 0.3.3

AgentFEM 0.3.3 closes the first provider-neutral structural direct-harmonic
workflow. Structural modal analysis remains at engineering maturity, and the
new harmonic workflow now follows the same readable `Study -> Model ->
scientific assets -> model.step(...) -> SimulationResult` lifecycle.

## Structural direct harmonic response

- A direct-harmonic step consumes inspectable stiffness, mass, viscous damping,
  loss-stiffness and force operators (`K`, `M`, `C`, `K_loss`, `F`) without
  hiding the assembled frequency-domain problem inside a material-specific
  procedure.
- One prepared problem supports a single frequency or a bounded canonical
  sweep, named complex responses, forward or reverse execution, and shared
  progress, history, checkpoint and result evidence. The sweep is
  operator-invariant: it reuses one real spatial `K/M/C/K_loss/F` system, with
  frequency entering through scalar coefficients and one global phase
  multiplying the real spatial force.
- Nested field-split and monolithic real-block layouts are available. The
  selected layout, matrix type and residual evidence are recorded, while
  solver policies that rely on an unjustified SPD assumption fail before the
  solve.
- The prepared backend freezes the operators, live coefficients, constraints,
  load phase, solution fields and solver policy that it actually lowered.
  Configuration drift requires a new Step, and non-finite solution, residual
  or energy evidence is rejected before publication.
- Repeated L2 recovery prepares its projection space, mass matrix and linear
  solver once, then updates only the right-hand side. This keeps scientific
  recovery explicit while avoiding repeated setup across a sweep.
- Reusable harmonic and exact-MPC numerical allocations have an explicit,
  idempotent `close()` lifecycle. Their owner releases distributed PETSc
  resources collectively, preventing rank-local cyclic garbage collection
  from entering an unrelated MPI collective; accepted results, summaries,
  public fields and provider-owned constraint graphs remain available.
- `U_AMPLITUDE` and `U_PHASE` are component-wise polar values of individual
  displacement coefficients. The reported maximum vector amplitude is the
  largest physical-cycle maximum of a discrete node/DOF-coefficient vector,
  computed analytically even when component phases differ; it is not a
  continuous-domain supremum.

## NAFEMS R0016 Test 5H evidence

The automated three-dimensional beam comparison uses the published 50-point
40--45 Hz frequency axis, geometry, linear-elastic properties, density,
Rayleigh damping and pressure loading. On the declared complete-Q2 mesh,
AgentFEM obtains:

| Observable | AgentFEM | Public reference | Relative difference |
| --- | ---: | ---: | ---: |
| Discrete peak frequency | 42.5510 Hz | 42.65 Hz | 0.232% |
| Midspan displacement amplitude | 13.2326 mm | 13.45 mm | 1.616% |
| Declared recovered longitudinal-stress amplitude | 236.398 MPa | 241.9 MPa | 2.275% |

The comparison independently checks the assembled residual, cyclic energy
balance, loaded area and applied resultant. Its 1%, 2% and 3% acceptance limits
are **AgentFEM release gates**, not tolerances published by NAFEMS. The exact
source, extraction convention and discretization boundary live in
`nafems_r0016_test5h_forced_vibration.json`.

A separate complete-Q2 sequence with 5 x 2 x 1, 9 x 4 x 3 and 13 x 6 x 5 cells
checks spatial stability. The final pair changes by 0.242% in peak frequency,
0.466% in displacement amplitude and 0.499% in recovered stress amplitude.
Because the refinements are nonuniform and the public-reference errors are not
monotonic, this is deliberately described as **three-observable stability**:
no observed order, Richardson extrapolation, GCI or continuum-error bound is
claimed.

The complete 50-point response and its scalar quantities are also checked in
serial and with two MPI ranks. Algebraic residuals and energy-balance errors
remain acceptance evidence rather than quantities that must be bitwise equal
under a different reduction order.

## Portable sweep restart

Direct-harmonic sweeps now write an atomic, partition-independent scalar
ledger. The versioned identity binds the scientific inputs, executable
operators, constrained degrees of freedom, response definitions, solver
contract, frequency axis and AgentFEM version. Acceptance exercises both
one-rank to two-rank and two-rank to one-rank restart, including a change in
execution order.

The checkpoint stores accepted scalar frequency evidence only. It does not
store or reconstruct a finite-element field at every frequency; this is stated
in both the checkpoint and `SimulationResult` instead of being implied by the
word “restart”. Bounded progress and checkpoint retention use the common
execution lifecycle, so long sweeps do not retain an unbounded field history.

## Maturity boundary

- **Engineering:** structural modal analysis and the small-strain linear
  structural direct-harmonic workflow for the declared operator system,
  supported strong constraints and bounded frequency sweeps.
- **Experimental:** generalized-Maxwell harmonic response. Test 5H is an
  elastic Rayleigh-damped benchmark and therefore does not validate the
  generalized-Maxwell frequency-dependent constitutive response.
- **Not claimed:** complex modes, prestressed small-on-large response,
  nonlinear harmonic balance, experimental validation, arbitrary constraint
  families, arbitrary complex spatial loads, frequency-dependent operator
  families, or full-field harmonic checkpoint recovery.

The release wheel is accepted only after the versioned contract, benchmark
cards, harmonic operator/result/checkpoint implementation and external
comparison assets are present in the installed distribution and the serial,
MPI, documentation and package-integrity gates pass against that exact wheel.
