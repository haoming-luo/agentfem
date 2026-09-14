# AgentFEM 0.3.5

AgentFEM 0.3.5 stabilizes the scientific-computing boundary introduced in the
0.3 series and adds the first coherent composite-material and textile-membrane
foundation. The public workflow remains unchanged:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

## Stable ownership without a larger public API

The implementation now enforces seven responsibilities: Model, Constitutive,
State, Operator, Procedure, Backend, and Result/Verification. Model inspection,
operator lowering, provider selection, built-in scientific construction,
nonlinear/transient evolution, and result assembly have separate internal
owners. Architecture tests prevent selected reverse dependencies from growing
back.

This is not a second finite-element kernel. DOLFINx/PETSc continue to own
finite-element assembly, distributed degrees of freedom, and numerical
algebra. AgentFEM owns the engineering definition, scientific lowering,
procedure and state lifecycle, and evidence returned to people and agents.

Configured engineering Steps no longer temporarily replace the source Model's
load and constraint registries. Lowering uses a shallow configured view, while
the executable retains the exact active configuration used for provenance and
output.

## Composite and textile foundation

- Add two- and three-dimensional orthotropic elasticity with explicit
  engineering shear conventions and independently assigned material frames.
- Add ply and laminate-section assets with ordered placement, `A/B/D`
  stiffness, generalized resultants, and section-point recovery.
- Add a `FiberFrame` for non-orthogonal reinforcement directions and
  independent tabulated warp, weft, trellising-shear, and bending channels.
- Add an experimental finite-kinematics woven-membrane provider through
  `studies.static_membrane()` and the standard Step/result lifecycle.
- Add objective finite-rotation director-surface measures as the kinematic
  foundation for a future fibrous-shell provider.

The current global provider is an in-plane membrane. Nonzero fabric bending is
rejected rather than ignored. A forming-capable shell, locking treatment, tool
contact, friction, and inter-ply slip remain separate validation gates.

## Operations and evidence

- Long nonlinear runs have a concise bounded heartbeat rather than silent
  multi-hour execution or unbounded log growth.
- CI depth follows the evidence affected by a change; complete numerical, MPI,
  installed-wheel, platform, and scientific gates remain mandatory for a
  release.
- Source promotion audits now bind themselves to the current checkout and fail
  if an older installed AgentFEM package would otherwise be mixed with the
  current Git commit.
- Compatibility Problem and Step types remain visible in generated API
  documentation after their implementations move to their correct owners.

## Maturity boundary

- The ownership refactor changes internal responsibility, not the recommended
  user language.
- Orthotropic materials, frames, and laminate sections are reusable scientific
  assets with local and FEM evidence appropriate to their capability records.
- The woven membrane and director-surface route remain experimental.
- This release does not claim a validated composite forming shell, general
  contact, arbitrary-path fracture, or universal finite-strain plasticity.

The release is published only after its exact wheel passes metadata and
payload checks, complete serial and MPI verification, installed-template
acceptance, strict documentation, optional PyTorch tests, and the versioned
release contract.
