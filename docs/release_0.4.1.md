# AgentFEM 0.4.1

This small release makes bounded two-way thermoelasticity available through
the ordinary `model.step()` workflow and strengthens sequential thermal-field
handoff. It does not rebuild complete offline installers or promote unrelated
experimental physics.

## Usable, explicit coupling

`studies.coupled_thermoelastic()` selects shared-mesh, three-dimensional,
small-strain thermoelasticity with constant isotropic material in SI units.
Displacement, temperature departure, participant loads and strong boundaries
are explicit. Procedure owns staggered implicit Euler; Model remains the
engineering definition. See the [complete design and example](coupling_design.md).

Both fields advance or roll back together. Progress uses the existing reporter;
checkpoint retention is configurable. Joint portable restart retains accepted
fields, force history, time and input identity. Final-state XDMF and a standard
SimulationResult carry physical time, diagnostics and performance evidence.
Manual `advance()` and automatic `run()` share configured iteration controls.

Boundary heat, prescribed-motion path work and the backward-Euler quadratic
identity are distinct records. The quadratic identity is **not** a certificate
of general thermodynamic energy conservation. Regional materials, nonlinear
coupling, nonmatching meshes and transient field-series output are not added.

## Verification and reliability

- Analytical heating and an independently assembled block reference.
- Independent Fourier-mode refinement: spatial order approximately two and
  temporal order approximately one for both fields, within its smooth-mode scope.
- Failed-window rollback and input-bound restart rejection.
- Public single-to-two-to-single-process continuation, including moving
  prescribed boundaries and force/work histories.
- Installed-wheel serial and MPI examples; publication remains gated by the
  immutable release wheel's Linux/macOS checks.

Sequential multi-material thermal-mechanical workflows also retain field
transfer evidence across rejection. Zero eigenstrain/projection and prepared
PETSc solution-state handling are corrected without giving up matrix reuse.

## Contact and next steps

The current source already contains moving rigid tools, generalized work,
multiple bounded pairs, triangle search and MPI routing in the explicit path.
Do not mistake the older fixed-plane implicit provider for the whole contact
capability. A targeted two-rank audit passed 34 tests per rank across the
ordinary explicit provider, distributed search and accepted work state.
This does not establish arbitrary industrial contact or forming.

Next work is selected from measured search/assembly/output bottlenecks and
unsupported mesh/formulation combinations. Shell forming, self-contact and
implicit friction remain separate scientific developments, not prerequisites
for using this bounded coupling release.
