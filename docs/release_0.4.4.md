# AgentFEM 0.4.4

0.4.4 consolidates bounded implicit contact, accepted-state evidence and
long-running project reliability. It preserves the ordinary
`model.step(...) → SimulationResult` workflow. A stable package version does
not turn experimental contact combinations into validated industrial forming.

## Contact composition

Compared with 0.4.3's single frictionless rigid tool, the serial three-dimensional
finite-strain J2 route now admits named multiple pairs, supported prescribed
motion, selected Coulomb branches and selected affine-periodic constraints.
Contact and bulk residuals/tangents enter the same Newton solve, including
line-search observations and reduction by the periodic constraint map.

The supported tool branches include analytical planes, reviewed planar facets
and separately reviewed closed convex/nonconvex surfaces. They do **not** share
one unrestricted friction capability. The
[geometry/friction/MPC matrix](guide/model_setup.md) is authoritative. Ambiguous
projections, unsupported active sharp features and distributed implicit contact
are rejected, not smoothed or omitted silently.

Experimental normal augmentation covers a narrower serial P1 plane route,
including prescribed translation and a disjoint-normal-row periodic case.
Optional safeguarded Aitken acceleration does not alter contact acceptance.
Rotating/curved tools and friction are not admitted by that augmentation route.

## Evidence and numerical safety

- Accepted histories distinguish contact force, moment, tool work, penalty
  potential, friction loss, elastic/hardening storage and plastic loss.
  Trial failure restores the supported joint state and ledger.
- Raw contact-point output retains reference and current positions, projection
  and integration-measure metadata. Recovered fields do not replace raw state.
- Physical quadrature-to-DG0 averages include the mesh Jacobian on non-affine
  and curved cells. Unsupported small-strain material/boundary compositions
  now fail before solving instead of dropping registered contributions.
- Invalid increment counts and nonfinite controls are rejected before execution.
  Nonlinear failure reports retain the underlying cause and accepted state.

The nonuniform periodic plastic-contact study remains experimental. A bounded
graded-cycle refinement reduces the final relative work/energy residual from
0.2046% to 0.00625%, with a 0.1137% final-force change. This is path sensitivity,
not spatial independence or an independent commercial-solver match. Peak
pressure and contact-edge resolution remain open scientific gates. Passing an
algebraic or same-discretization test is not a continuum-accuracy certificate.

## Output, restart and ordinary use

The live frame-file output mode publishes immutable accepted frames and
atomically updates their XDMF index. Completed frames can be viewed during a
run. Supported restarts preserve the original files and write a new segment
starting from the restored time; they do not append to or automatically merge
the previous XDMF file. A validated
partial index can also recover supported interrupted legacy output. No output
scheme can recreate an unsaved past field from scalar history alone.

Checkpoint cadence, scalar history cadence and visual frame cadence remain
separate controls. Only accepted state may be published. Existing output
regressions cover interrupted publication, distinct time indexing and
restart-segment continuity; users need not supply AI-generated file-writing glue.

Optional [binary history storage](guide/history_storage.md) replaces oversized
inline JSON arrays with lossless, checked NPZ companions. Reading or comparing
these histories does not need to start the numerical runtime. The option
changes storage, not sampling, time integration or physical acceptance.

Windows onboarding now offers a short staged route: reuse or prepare WSL,
obtain a checked Ubuntu distribution, prepare the ordinary user/environment,
then run the first solve. Explicit mirror choice is retained. A blocked stage
gives one manual action rather than a repeated automatic retry. Fresh
Windows/WorkBuddy field acceptance is distinct from PowerShell and platform CI.

## Execution efficiency

Changes reuse material spectra and call-local responses, coalesce repeated
contact stencils, avoid unused observation tangents/field copies, and offer
opt-in managed GMRES preconditioner reuse. Defaults and scientific state
ownership remain explicit. Help/version discovery no longer starts MPI.

Performance reports include fixed-workload comparisons and correctness checks.
One 48-increment implicit fixture with combined opt-in execution policies
measured 24.32% less time; a separate 64-increment observation revision measured
3.68% less time against its merged-main baseline while adding the energy ledger.
These percentages cannot be added, applied to every computer, or transferred
to the explicit composite simulation. No new full-composite speedup is claimed.

## Release boundary

This release does not claim general self-contact, deformable-master implicit
contact, arbitrary sharp-feature friction, shell forming or distributed
implicit contact. Existing feature-specific maturity remains in force.

The exact candidate and remaining evidence are tracked in the
[release decision record](internal/release_0_4_4_candidate_gate_20261027.md).
Availability requires successful publication of the verified artifacts;
a release note, merged PR or Git tag alone is not publication evidence.
