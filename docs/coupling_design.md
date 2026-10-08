# Bounded coupling design: after sequential thermal mechanics

Status: design, not an implemented two-way coupling capability.

## Product decision

Finish the shared-mesh sequential route first. Keep Model as the problem
definition, Operator as the physical residual contribution, Procedure as the
iteration owner, State as the accepted/trial owner, and Result as evidence.
Do not introduce a new multiphysics Model or a thermal-wall-specific solver.

The first possible two-way vertical slice is small-strain, homogeneous linear
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
implementing the loop, audit participant snapshot/restore for all time,
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
