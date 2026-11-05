# Bounded physical-time small-strain material consumer

## Decision

Reuse `SmallStrainMaterialStep`, its Newton loop, quadrature transaction,
incrementation and result lifecycle. Add an optional positive `duration` and
`time_unit` to the ordinary `model.step` option contract. Without duration, the
existing rate-independent route and portable v1 checkpoint remain unchanged.

The new route is experimental: serial, homogeneous 3D, small strain, strong
displacement constraints and reference dead loads with one shared normalized
amplitude. The provider declares `stress_free_initial_state=True`: zero strain
and zero named history represent a stress-free reference. Initial displacement
and named history must be zero. Nonzero initial states, prescribed temperature,
contact, regional composition and MPI are not admitted by this slice.

## Time and transaction

For path coordinate `a`, material endpoint time is `duration*a`; a trial from
`a0` to `a1` receives `dt=duration*(a1-a0)`. The amplitude remains a function of
normalized coordinate, not seconds. Tabular hold/reversal knots are visited
exactly, including when a nominal fixed increment would cross them.

Construction does not call a rate-dependent provider with a fictitious unit
time increment. Each Newton/line-search trial starts from committed history.
Rejected increments restore accepted stress, tangent, energy fields and
diagnostics directly; they do not invoke another positive-time material update.
Only the new physical-time route takes the extra response snapshot per attempt.

The first checkpoint is available after the first accepted positive increment.
V2 manifests reuse portable displacement and quadrature-state payloads, adding
an exact accepted-response archive. Response restore is same-layout serial;
`portable=True` is rejected. No material call reconstructs stress or tangent
on restart. The outer restore transaction from PR #149 protects the live state.
Duration, units, material identity, load, strong boundary targets and path policy
belong to scientific identity. Changing them cannot silently continue a path.

## Independent references and acceptance

- A standard linear solid with backward-Euler deviatoric internal strain checks
  prescribed strain relaxation, force-controlled creep, and time refinement.
- A Kelvin–Voigt local law uses the same structural consumer and verifies the
  strain-rate contribution under a linear ramp.
- A passive age variable verifies that initialization, repeated Newton trials,
  rejected increments and restart never add fictitious physical time.
- Interrupted/continuous results compare displacement, internal state, stress
  and tangent; corrupt response arrays and changed identities reject atomically.
- Existing rate-independent solves and portable restart remain regression gates.

These are numerical verification laws, not calibrated engineering materials or
P92 multiaxial validation. Their local coefficients are intentionally synthetic.

Public implementation references:
[MOOSE GeneralizedMaxwellModel](https://mooseframework.inl.gov/releases/moose/2024-11-11/source/materials/GeneralizedMaxwellModel.html)
documents internal-strain time integration with backward Euler;
[NEML creep/plasticity interface](https://neml.readthedocs.io/en/dev/interfaces/creep_plasticity.html)
separates local stress/history updates from the global structural solver.
We reuse that separation of responsibilities, not their validation claims.

## Executed candidate checks

The targeted material, small-strain consumer, restore, array ownership,
CI-routing and documentation group passed 237 tests; the one serial skip is
the dedicated two-rank rejection test, which passed separately on both ranks.
The existing rate-independent restore group passed all ten cases on two ranks.
All MPI checks used `agentfem mpi-run --timeout 120` and one numerical thread
per rank. Ruff, generated documentation and 58 scientific function cards pass.

A temporary pure-Python candidate wheel was installed into a fresh target
directory against the existing FEniCSx runtime. The four-file example ran through
`agentfem check` and `agentfem run` without the source tree on its import path.
Its midpoint restart and continuous run have exactly equal numeric displacement,
quadrature-state and accepted-response archives. Both final XDMF views contain
time 2 seconds and one grid with U, UMAG, S_CELL, EV_CELL and RF.

The first installed scalar-provider run exposed a read-only diagnostic mapping
that was not directly JSON serializable. Serialization now normalizes those
mappings, and scalar-vs-batch plus scalar restart regressions cover the defect.
Additional injections cover interruption during assembly and late response-file
write failure: the accepted live state and previously published checkpoint remain
usable. Declared stored-energy components and dissipation fields survive restart;
this transport test is not a claim of general first-law closure.

## Next gate

Keep the published P92 dwell reference independent. Audit its multiaxial extension,
energy conventions, static recovery and local algorithmic tangent before admitting
a 3D provider; do not promote a uniaxial parameter fit by embedding it in a cube.
Prescribed temperature and co-located history recording can then be composed
with this consumer as separately tested increments.
