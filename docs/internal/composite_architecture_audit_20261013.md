# Composite workflow architecture audit — 2026-10-13

## Decision

Keep the ordinary Model → Step → SimulationResult workflow. Do not introduce
a composite-case solver or another time loop. The original confidential input,
material law, geometry preparation and installed run remain outside the public
repository. This audit is not an Abaqus-equivalence claim.

## Ownership and actual admission before native promotion

| Concern | Reusable core surface | Remaining boundary |
| --- | --- | --- |
| Engineering model | dynamic_solid, Model, fields, named boundary models | Bounded response-only route is serial, one full-domain material |
| User material | explicit_user_material, named state schema, array batch response | Confidential law and its native implementation stay external; no fabricated tangent |
| Body execution | ExplicitHexBodyKernel, finite uniform-strain Hex8, owned_arrays | Fast fused implementation remains a trusted external extension; core validates transport, not mathematical equivalence |
| Contact | rigid planes/cylinders, deformable_contact_pair, Q1ContactExecution | One-sided penalty and restricted geometry are not ALL EXTERIOR, hard contact or self-contact |
| Initial interface | initial_nodal_bond, elastic_cohesive, nonmatching_cohesive_force | Eligibility and adjustment are explicit case inputs; no inferred rebonding |
| Stabilization | bulk viscosity, hourglass policy, numerical-inertia declaration | Parameters and stability ceilings need sensitivity evidence, not universal defaults |
| State | trial/commit/rollback, ordinary checkpoint save/load | Response-only material state does not yet admit MPI migration |
| Reporting | accepted history, SimulationResult, append-only journal | Source-specific damage names remain material-owned; a completed result is not vendor validation |

The frozen private project uses these public interfaces, but imports private
material/contact runtimes and geometry/export helpers. Therefore public API
integration is real, while universal out-of-the-box speed and entirely generic
mesh/export preparation are not yet established. Do not rename these remaining
dependencies as completed core migration.

## Implemented reinforcement

1. Reuse checkpointing.every rather than reimplementing modulo logic in cases.
   Regression covers restart between scheduled stations, final saving and both
   accepted-state storage modes.
2. Bridge AcceptedHistoryJournal to the existing transient SolveEvent stream.
   One solve owns one execution lifecycle; reporting reads accepted frames only.
   It neither owns state nor changes scientific history integration cadence.
3. Fail closed on stale frames, incompatible cadence, unsupported procedures,
   implicit MPI collection and poisoned output. I/O failure after acceptance
   leaves accepted numerical state available for explicit checkpoint recovery.
4. Demonstrate automatic checkpoint + journal in the nonconfidential moving-tool
   user-material example. No new numerical route or material special case.
5. Correct the documentation: bounded in-memory execution traces are not a
   promise that all accepted increments remain in the result manifest.

## Efficiency implications

The previous case called run every ten increments. For a 100,000-step fresh run
that means 10,000 run entries, repeated preflight/reporter setup, and—in overlap
mode—repeated worker-scope creation and shutdown. An accepted-event observer
permits one run entry with identical requested history/checkpoint cadence.
This removes structural overhead; it is **not a measured speedup claim**.
Journaling still flushes/fsyncs at each requested reporting station.

Core native arithmetic is optional, stateless and guarded by reference paths.
It does not change the material, mass, contact or time increment. The private
completed continuation reached step 100,000 and passed same-station comparison
against the older implementation; this neither times a fresh full run on the
new runtime nor establishes equivalence to the external solver.

## Next gates, in order

1. Freeze a new installed candidate without altering the completed project.
   Migrate operational looping to the public journal/checkpoint contracts; retain
   case physics and the confidential material independently. A supervisor owns
   wall-clock interruption; the Procedure owns accepted physical increments.
2. Compare candidate and frozen reference from identical early, damaged and late
   checkpoints. Include hidden state, damage, work, contact and restart. Then
   measure a fresh full run including preparation and final output.
3. Promote external fast arithmetic only after proving it independent of one
   material schema and one geometry. Keep confidential constitutive details out
   of the core. Require a reference fallback and at least two unrelated laws or
   geometries before declaring general acceleration.
4. Unify remaining export/preparation helpers with existing mesh identity and
   result contracts. Never lose Hex8 ordering, reference geometry, stress
   measure, material state schema or source-node mapping for readability.
5. Separate scientific gates: inertia/step/contact sensitivity and comparison
   to Abaqus. Do not expand friction, shells or MPI simply to widen this sprint.

## Verification of this reinforcement

- 190 targeted checks passed using a separately installed native wheel, with
  pytest source-path injection disabled. Selection: transient journal, journal
  integrity, response material Step, transient restart, native arithmetic,
  general results and CI scope classification.
- The public example ran from outside the repository against that installation:
  ten accepted increments, scheduled checkpoints at 5 and 10, and five journal
  rows at increments 2, 4, 6, 8 and 10.
- Native wheel SHA-256:
  `4f63d4e0952e1641c51d71644c5f6538e817c78d3261fee2a99047f1e1346e5f`.
- These are API, arithmetic and lifecycle regressions, not a new full-case
  timing or MPI journal acceptance. The completed private project was untouched.

## External design references

- [MOOSE output lifecycle](https://mooseframework.inl.gov/syntax/Outputs/):
  output schedules attach to execution stages, including timestep end. The
  AgentFEM implementation reuses its own accepted event contract rather than
  importing another framework.
- [DOLFINx assembly](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/_modules/dolfinx/fem/assemble.html):
  reusable packed constants/coefficients avoid repeated preparation when inputs
  are unchanged. This supports explicit preparation ownership; it does not
  justify caching evolving damage or displacement across accepted increments.

## Native promotion — implementation, not final acceptance

The generic fast paths now have core implementations and ordinary factories:

- `elements.explicit_hex_kernel`: material-independent Piola conversion,
  uniform-strain internal force and hourglass force, with a NumPy reference.
- `elements.native_explicit_hex_kernel`: a trusted, reentrant material callback
  ABI with arbitrary state/property widths, joined workers and exact repair of
  rejected points before trial publication. The confidential law stays outside.
- `boundary_models.q1_contact_runtime`: packaged Q1 bounds, candidate search,
  swept separation and exact-predicate acceleration. Ambiguity and unresolved
  certificates still use the established reference behavior.
- The candidate private case uses those factories, one Procedure run, scheduled
  checkpoints and an accepted-event journal. The completed old project is intact.

Native compilation is attempted by default and remains optional. GMP-dependent
Q1 acceleration and the threaded material ABI are currently POSIX build paths;
the locally tested macOS wheel is not a claim of Windows or Linux wheel coverage.
See [execution contracts](../native_explicit_execution.md).

Evidence so far: 112 targeted tests passed against a separately installed wheel.
State widths 1, 7 and 31, reference repair, guarded geometry and both accepted
storage modes are exercised. On identical original-mesh accepted inputs, old
and new material/body/kinematics/viscosity responses were bitwise equal. This is
a local arithmetic comparison, not trajectory validation or measured acceleration.
The same quantities were bitwise equal for a small-increment offline probe on
the old full-run damaged final state. A pure-Python wheel separately passed
five applicable checks (14 native-only checks skipped), including the classic
wave example. Native and pure builds used separate build/install directories.

The first 200-increment candidate failed the unchanged pointwise acceleration
gate (relative L2 error approximately 6.13e-9). Cross-product runs isolated the
cause to Q1 projection: packaging omitted the scale-conditioned margin build
option, selecting different projection branches near edges. Restoring that
existing policy made all 2,000 archived difficult projections bitwise equal;
the complete 200-increment replay then passed the original gate. Both failed
and successful comparisons are retained. The packaged module now requires
that policy at compile time and exposes its identity for installation tests.
This is not yet full-trajectory or vendor validation.

A nonconfidential classic example, `examples/elastic_bar_wave.py`, runs through
the ordinary installed interface. Joint spatial/time refinement at 20/40/80
axial cells gave displacement errors approximately 0.2714/0.0710/0.0209 percent
against the small-amplitude analytical standing wave. This demonstrates a
bounded elastic workflow, not contact, large-strain or industrial validation.

Remaining acceptance: diagnose whole-step differences; sample damaged and late
states; freeze the new project/runtime; measure a complete fresh original case
including preparation/export; compare all accepted scientific outputs. Generic
geometry/export preparation and independent vendor validation remain separate.

## Single-grid export correction

The delivered legacy descriptor repeated the mesh for each function. The
displacement attribute therefore existed on only part of the multiblock output;
Warp left other blocks undeformed. This was an output-layout bug, not evidence
of two physical bodies or a need to change the solver's connectivity.

`ResultWriter` now routes new serial output through the existing unified writer;
the classic example uses it. The presentation-only `consolidate_xdmf_fields`
adapter repairs old single-mesh synchronized archives without modifying their
HDF5 arrays. Independent VTK reading/warp of two full-sized previous archives
confirmed one grid, 255,041 volume cells, 296,074 points, and coordinates x+u.
The originals remain intact. This does not alter the already frozen full run.
Installed ParaView 6.1.1 additionally reproduced seven blocks from the legacy
file and read exactly one block from the corrected descriptor using
`Xdmf3ReaderS`. Its actual Warp filter produced x+u with zero maximum coordinate
error. The reusable `tools/check_paraview_warp.py` records this presentation gate.

Cell-array export is now reusable through `cell_field_from_values`, with
explicit cell identities and no cross-interface smoothing. DG0 export reads
the actual cell-to-dof map instead of assuming identity ordering. Cross-mesh
field mixing is rejected even when array sizes match. These changes reduce
case-owned export plumbing; no whole-run acceleration percentage is claimed.

## Localized native repair

The native material/body dispatcher now computes successful point forces in
the original worker pass and revisits only rejected points after exact material
repair. Unaffected worker slices are not launched for repair, and one-worker
execution avoids temporary thread creation. Output remains unpublished until
every phase joins; failed repair still aborts the trial. Execution identity is
revision 2, so a revision-1 checkpoint/run is not silently retargeted.

The installed candidate passed 196 focused tests, including no/single/alternating/
all-point rejection for multiple state widths and worker counts, result output,
restart and the public bar example. A fixed damaged final-state probe on the
original mesh produced bitwise-identical stress, state, work, Piola stress,
force, hourglass, geometry admission and viscosity outputs versus the previous
private arithmetic. The probe overlapped the frozen full run, used one repeat,
and is correctness evidence only—not a speed comparison or full trajectory.

After integration with the current main branch, the installed wheel passed
223 focused tests, including geometric-elasticity regression. An independently
pinned public elastic-bar project completed 640 increments with 40 cells:
relative displacement error against the linear-limit reference was
0.000709678842, and relative energy-balance error was 0.00000193265549.
This is a small public workflow check, not the original composite full-run
acceptance or a large-strain validation. The original full run remains on its
frozen revision-1 dispatcher; it does not inherit revision-2 repair changes.

## Result contract hardening

Serial frame admission now precedes HDF5 mutation. Nonfinite values, invalid
time coordinates, duplicate/reserved names, schema changes and moved reference
geometry cannot leave half a frame. Storage exceptions poison the writer.
Continuous auxiliary fields are interpolated using the primary's actual nodal
element variant, including equal-coefficient-count/different-node cases.
Unchanged reference geometry uses one shared HDF5 dataset across time.

The installed candidate passed 195 targeted output, native admission, material
transaction and restart tests. Its Cauchy-stress symmetry check preserves the
same relative predicate while removing whole-mesh absolute-tensor temporaries;
no integration-throughput claim is made while the frozen full run is active.
The standard installed templates and discoverable visualization knowledge card
direct agents to the same output APIs rather than hand-built connectivity.
Arbitrary external Python exporters cannot be controlled by this contract.

Independent ParaView 6.1.1 reading and Warp passed for a two-frame series whose
first saved time is 1 (one volume grid, 27 points, 8 cells, zero coordinate
error on that frame). The checker now selects an actual saved time rather than
silently requesting the reader's default time zero. This is presentation QA,
not a scientific verification of the field.

## Native distribution acceptance

Linux numerical, MPI and release-example acceptance passed on candidate
`c3ab5a7f`. The following macOS stage failed because it attempted to install the
Linux native wheel, exposing an obsolete platform-independent-wheel assumption.
The corrected workflow builds a macOS wheel from the exact Linux-tested source
archive, requires the installed native body and arithmetic modules, and reruns
platform acceptance. Promotion still requires one version and source commit;
platform-specific binaries correctly have distinct hashes. Twelve local gate
and installed-identity regressions passed; remote acceptance of this correction
is still required. No native platform gate was removed or marked passed by hand.

## Accepted-body ownership and installed diagnostics

A three-second read-only macOS process sample of the frozen original run showed
material/polar updates, body arithmetic, memory movement/freeing and worker waits.
Thread-wait samples include idle workers; they are not additive wall-time phase
percentages and do not establish a projected speedup. Sampling and local build/
test activity overlapped the run and must be disclosed with its eventual timing.

The response-only body's accepted gradient, force and displacement are now
immutable payloads. Rollback capture creates fresh array descriptors over those
bytes, instead of three additional whole-array copies. This reuses the existing
freeze helper; it does not add a competing state owner or a persistent cache.
For a model with 255,041 cells and 296,074 three-component nodes, those arrays
contain 32,574,504 bytes per snapshot (excluding material and contact state).
This is an eliminated allocation/copy count, not a measured elapsed-time saving.
Tests cover independent snapshot metadata, write rejection, old-record retention,
failure rollback and restart in both material storage modes.

Installed `doctor` now reports native modules as available, not built or failed
to load, without running kernels or creating worker pools. All three packaged
extensions loaded from the separately installed macOS candidate. Availability
does not imply that a Step selected that backend or passed scientific validation.

The stricter output guard initially rejected two real mixed-element regressions
because separately collapsed but equivalent spaces have different object
identities. It now compares the UFL/Basix element, block sizes and cell-to-dof
maps when identities differ. Both previously failing mixed-element tests and
the additional reconstructed-space fixture passed; incompatible layouts remain
rejected. The source-only focused group passed 50 checks, and the separately
installed ownership/native/contact group passed 114 checks before this last
output correction. These do not replace remote release acceptance.

Final combined installed-wheel check: 243 targeted tests passed, including both
previous remote output failures, before submission for renewed CI acceptance.
The original frozen run's step-40,000 station matched the completed previous
implementation bitwise for 3 nodal arrays, 57 auxiliary arrays, 20 numeric
history channels, the damage subset and 76 contact evidence entries. This is
offline implementation replay; it neither admits a restart migration nor
validates the scientific model against Abaqus.
