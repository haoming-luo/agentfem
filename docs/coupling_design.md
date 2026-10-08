# Bounded coupling design: after sequential thermal mechanics

Status: bounded 3D verification prototype implemented; a production two-way
Step provider is not yet available.

## Implemented experiment

`operators.thermoelastic_heat_source(...)` contributes explicit reversible
volumetric heat feedback for constant coefficients on a shared 3D mesh. It
does not enable coupling automatically. A separate private benchmark driver
compares staggered implicit Euler with independently written mixed-space UFL
forms, both executed by the existing prepared PETSc solver. No second solver
backend, global Model state or nested transient-step clock was introduced.

```bash
PYTHONPATH=src python tools/benchmark_thermoelastic_coupling.py --nonuniform --output outputs/coupling-prototype.json
PYTHONPATH=src agentfem mpi-run -n 2 --timeout 120 -- python tools/benchmark_thermoelastic_coupling.py --alpha 0.01 --relaxation 0.25
```

These are repository development commands, not installed-user templates. The
private class under `benchmarks/thermoelastic_coupling.py` is not a stable
public Procedure. The result remains `computed`; successful execution does not
claim general coupled validation.

The experiment uses an insulated unit cube with symmetry rollers, homogeneous
isotropic elasticity, constant-strain volumetric capacity and no external
mechanical work. The uniform-source oracle is independent:

```text
theta(t) = Q*t / (rho*c_epsilon + 9*K*alpha^2*T0)
u(x,t) = alpha*theta(t)*x
```

A nonuniform source additionally activates conduction. The comparison tests
two mesh/time resolutions, fixed relaxation, zero coupling, strong-coupling
nonconvergence, single-rank acceptance rejection and retry. They test agreement
with the discrete reference, not a full mesh/time convergence certificate.
Residuals are unrelaxed volume-normalized FE RMS norms, with each field
accepted against `atol + rtol*norm(field)`. Both criteria must pass. Exactly
zero coupling needs only one solve of each independent block.

Constant thermal and elastic matrices are each assembled once and reused
through outer iterations and accepted windows. Solver execution counters retain
failed attempts; accepted time/history and fields do not. No output is published
from rejected windows. Non-finite global diagnostics and rank-inconsistent
iteration controls are rejected explicitly.

### Energy evidence: deliberately separate identities

For insulated boundaries, the volume integral of the linearized heat equation
checks `rho*c_epsilon*delta_theta + beta*T0*div(delta_u) - dt*Q`.
This is not a nonlinear total-energy closure claim.

With homogeneous mechanical boundary work, the prototype also verifies the
discrete quadratic identity. Write `a(u,v)` for elastic virtual work and let

```text
E_n = a(u_n,u_n)/2 + integral(rho*c_epsilon*theta_n^2/(2*T0))
D_dt = a(delta_u,delta_u)/2 + integral(rho*c_epsilon*delta_theta^2/(2*T0))
E_n - E_old + D_dt + dt*integral(k*|grad(theta_n)|^2/T0)
    = dt*integral(Q*theta_n/T0)
```

The time-discretization term is not material dissipation. This positive
quadratic balance follows by testing the mechanical equation with delta_u and
the thermal equation with dt*theta_n/T0; opposite coupling terms cancel.
Keep it separate from physical total internal energy and boundary heating.

One local two-cell-per-axis nonuniform run (three dt=0.1 windows, alpha=0.002)
used 9 outer iterations per window and one assembly per participant; maximum
temperature/displacement reference errors were approximately 2.2e-12 K and
4.4e-15 m. These are RMS differences between two discrete solves, not physical
prediction errors or a platform speedup. Reproduce with the command above.

### Remaining promotion gates

1. Nonzero prescribed-motion work and thermal boundary input with an explicit
   thermodynamic ledger, not merely the restricted quadratic identity.
2. Joint portable checkpoint/restart for both accepted fields, time and
   iteration-independent physical history; current rollback is in memory only.
3. Physical space/time convergence and an external problem beyond the cube.
4. A bounded ordinary Step provider and its option contract, only after the
   participant State/restart semantics are stable. No new Model god object.

## Product decision

Finish the shared-mesh sequential route first. Keep Model as the problem
definition, Operator as the physical residual contribution, Procedure as the
iteration owner, State as the accepted/trial owner, and Result as evidence.
Do not introduce a new multiphysics Model or a thermal-wall-specific solver.

The first two-way verification slice is small-strain, homogeneous linear
thermoelasticity on one mesh with a reversible volumetric-strain-rate heat
term. It is selected for an independent linear-system reference and explicit
energy identity, not because it covers industrial forming. No plastic-work
heat fraction, contact heating, arbitrary mesh transfer, or nonlinear material
history belongs in this first slice.

For temperature departure theta = T - T0 and constant isotropic coefficients:

```text
sigma = C:epsilon(u) - beta*theta*I
rho*c*theta_dot - div(k*grad(theta)) = r - beta*T0*div(u_dot)
beta = 3*K*alpha, K = bulk modulus
```

Here c is the heat capacity consistent with the chosen strain-based linear
thermoelastic free energy. Derive units, signs, boundary work and entropy/
energy identities before coding the thermal feedback Operator. Do not insert
this term into existing one-way heat models automatically. The 2D assumption
must be explicitly derived; start with a 3D manufactured reference to avoid
silently using a plane-stress reduction for a plane-strain model.

## Numerical route and ownership

1. Save the accepted time-window state of **both** participants.
2. Solve thermal then mechanical subproblems at the same proposed physical
   time, with a bounded outer fixed-point iteration and optional fixed
   under-relaxation. Keep the accepted old-time state fixed during every retry.
3. Measure the **unrelaxed** update/residual of each exchanged field, using
   MPI-global owned coefficients or physical FE norms. Report dimensional
   absolute and scaled relative residuals separately. A tiny relaxation factor
   must not create false convergence.
4. Require every declared field criterion and the inner solver criteria.
   An inner Newton convergence flag does not imply outer coupling convergence.
   Near-zero reference fields require an absolute criterion, not division by
   an arbitrary tiny denominator. Tolerances and field scales are user-visible.
5. Commit both states/time only after outer acceptance; on failure restore
   both, discard trial output, and report an explicit exhausted-iteration
   status. Time cutback, if enabled, restarts from the same accepted boundary.

Sequential `EngineeringStep` configuration remains declarative. An eventual
coupled Procedure owns this loop; `state.field_transaction` alone cannot roll
back a participant's completed time index or constitutive history. Before
promoting the loop, audit participant snapshot/restore for all time,
history, callbacks and operator invalidation, not just displacement/temperature.

With constant coefficients and dt, coupling updates change the RHS, allowing
prepared subproblem reuse. A changed conductivity, elastic tangent, dt or
constraint membership follows the existing operator invalidation contract.
Timing and assembly counters must demonstrate reuse under unchanged physics.

## Acceptance and stop conditions

- Zero coupling reduces to the existing independently verified heat/solid
  solutions; no needless outer iteration is required.
- Compare to an independently assembled monolithic block system for a tiny
  linear manufactured problem, including sign and unit checks.
- Verify first-law accounting with imposed motion and thermal boundary input;
  do not equate mechanical strain energy alone to a thermodynamic balance.
- Compare dt and mesh refinement, two rank counts, iteration tolerances and
  acceleration disabled/enabled. Under-relaxation must change convergence
  speed, not the converged solution.
- Force one-rank failure and outer nonconvergence: neither participant may
  advance accepted time; restart/retry must match an uninterrupted solution.
- Record outer residual history, inner solve counts, accepted time windows,
  rollback/cutback counts and immutable input references in existing results.

Timebox the first prototype to these gates. If participant rollback requires
a broad rewrite, stop and publish the gap rather than adding a callback loop
that merely appears coupled. Aitken/quasi-Newton acceleration follows a correct
fixed-point baseline; a second backend and nonmatching meshes remain deferred.

## References and design rationale

- [Bleyer: full thermoelastic evolution with FEniCSx](https://bleyerj.github.io/comet-fenicsx/tours/linear_problems/thermoelasticity_full/thermoelasticity_full.html):
  constant-strain heat capacity, entropy coupling and implicit-Euler weak
  forms. Our 3D implementation and separate reference forms follow the stated
  equations; the benchmark geometry and verification obligations are ours.

- [Abaqus: fully coupled thermal-stress analysis](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-couptempdisp.htm):
  distinguishes coupled unknowns from predefined-temperature sequential use.
- [preCICE: implicit coupling](https://precice.org/couple-your-code-implicit-coupling):
  participant checkpoints support rejected iterations; time advancement is
  conditional on coupling acceptance.
- [preCICE: coupling configuration](https://precice.org/configuration-coupling):
  bounded iterations and explicit absolute/relative convergence measures.
- [preCICE: acceleration](https://precice.org/configuration-acceleration):
  under-relaxation, Aitken and quasi-Newton alternatives, and the importance of
  sufficiently converged participant solves.

These references inform the contract. This plan neither adds a preCICE
dependency nor claims that borrowing its iteration pattern verifies physics.

The zero-coupling regression also exposed two general edge cases: UFL erasing
the argument of an exactly zero eigenstrain/projection form, and stale PETSc
solution norms after direct NumPy edits. Both now have targeted regressions.
The prepared-solve boundary increments vector state before KSP, following
[PETSc object-state semantics](https://petsc.org/main/manualpages/Sys/PetscObjectStateIncrease/);
it does not invalidate the matrix or its factorization.
