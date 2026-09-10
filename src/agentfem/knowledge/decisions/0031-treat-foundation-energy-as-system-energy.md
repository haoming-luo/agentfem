# Treat elastic-foundation energy as system energy

## Decision

The linear elastic foundation is the first native weak provider in the shared
constraint-balance contract. It publishes the force exerted on the solid as
the negative action of its assembled boundary stiffness and retains that
distribution as a live result field.

Its conservative spring energy remains inside `0.5 * u.T @ K @ u`. The work
ledger does not add a second `0.5 * R_foundation.T @ u` term. The capability
therefore declares `work_evidence="internal_energy_operator"`; its provider
dual closes force balance without pretending to be prescribed kinematics.

## Reason

Treating a foundation as strong Dirichlet data loses support compliance.
Treating its reaction as a natural external load double-counts energy because
the same spring is already in the left-hand-side operator. A distinct weak
provider preserves both physical meanings while using the shared result and
verification lifecycle.

## Consequences

- Model continues to register the engineering boundary model;
- the Step builder includes its operator in stiffness and its asset in the
  evidence contract without passing it to strong-BC lowering;
- the boundary model owns reaction recovery from the exact operator it created;
- Result owns global balance, visualization routing, and compact evidence;
- nonlinear, dissipative, moving-normal, and contact laws require separate
  work/energy semantics and do not inherit this completion flag.

## Verification

- a traction-loaded body supported only by an elastic foundation is nonsingular;
- the foundation resultant opposes and matches the applied resultant;
- force and proportional-work balances close in serial and with two MPI ranks;
- stored spring energy is positive and labelled as part of system strain energy;
- the nodal foundation reaction is present in `SimulationResult`.
