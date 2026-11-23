# Development portfolio checkpoint — 2026-11-23

## Actual work in progress, not the size of the backlog

At this audit there are **two open repository PRs**, not dozens of simultaneous
development tasks. This thread owns one active scientific workstream (#197).
The candidate and its required same-driver control were two bounded numerical
jobs within that one workstream; both identification and frozen replay have
now completed. No full composite solve or MPI test was started. Open branches/worktrees and
historical evidence do not prove that work is actively executing elsewhere.

| Delivery | Checked state | Boundary |
| --- | --- | --- |
| #197 P92 recovery discriminator | Head 1bbd95b3: all applicable CI passed; the new mobility-population batch completed its scientific comparison and is being locally closed | Do not merge/release; experimental response acceptance still fails |
| #198 installed workflows / model inspection | Head eb11f573: all applicable CI passed, including Windows shells and macOS installed acceptance | Separately owned work; no edits or merge in this thread |

#198 reports faster warmed metadata inspection, not faster material integration
or a new full-composite solve time. Its CI and description were checked
read-only; this portfolio note is not a new independent review of its code.

## Product routes and their next evidence

| Route | Existing foundation | Next bounded acceptance |
| --- | --- | --- |
| P92 cyclic response | Recorded RF/SLR/SLC paths, continuous internal history, numerical oracles, common early identification, frozen development screens | One shared card satisfies the original resolved hold, entry, reversal and peak gates; no protocol-specific correction |
| Creep–fatigue life and uncertainty | Co-located histories, source/cohort identities, rupture and strain-life consumers, censoring and bounded reliability statistics | Compatible specimen-level calibration and held-out outcomes with uncertainty and censoring retained; no cross-heat pooling or design-life claim from response fitting |
| General user materials / thermal mechanics | Physical-time local updates, accepted/trial lifecycle, histories, restart and admitted temperature inputs | Transfer an accepted law through material-point, tensor and structure comparisons; unsupported combinations remain separate tests |
| Contact / composite damage | Bounded contact, motion work, transactions, restart and explicit execution evidence | Resolve the outstanding nonuniform spatial/independent-reference gates before broader parity claims; recheck representative composite states only for affected owners |
| Mesh and element adaptation | Imported identities, regions, topology checks, supported element contracts | One added topology/formulation at a time: import → patch → solve → field output → restart; do not label connectivity recognition as formulation support |
| Lightweight result viewing | Standard published fields, histories, live accepted series and continuation views | Separate optional read-only viewer: open local result, field/time/component selection, warp and picking, missing-frame handling; no copied solver or required web dependency |
| AI-native / installed UX / performance | Structured public workflow, capability/error inspection, atomic records and installed regression | Finish separately owned #198 review/integration when authorized; measure real task success and solver-level timings separately from metadata improvements |

The last four rows are maintained foundations or queued routes in this thread,
not four newly started sprints. Front/postprocessing stays separate from P92.

## Sequence and milestone decisions

1. **Mechanism discrimination:** keep recoverable strain; distinguish kinetic
   mobility from flow strength. Prespecify replacement, deletion control,
   unchanged parameter count and budget. Reject conflicting fixed screens
   rather than tune every mechanism indefinitely.
2. **Joint identification:** the original 22-coordinate/10-direction budget,
   early three cycles only, derivative and local observability checks. A
   successful optimizer is not material acceptance.
3. **Frozen response:** all protocols, subsequent inspected development cycles,
   independent integration and waveform refinement, unchanged physical gates.
   Publish every failure, not only a lower aggregate objective.
4. **Independent use and life:** only after the response is adequate for its
   declared use, test a genuinely independent compatible condition and life
   outcomes. Long-term rupture and uncertainty are separate scientific gates.
5. **Productization:** promote reusable interfaces/tests/documentation from
   demonstrated needs. Do not ship the latest failed research mechanism as a
   general P92 material. Continue the lightweight viewer in its own branch.

If a candidate helps holds but worsens reverse peaks after bounded joint
identification, stop that candidate. Revisit what observations distinguish
model structure and initial state rather than expanding coefficient ranges,
inventing extra observations, or treating CI as experimental validation.

The practical bottleneck is now **cross-protocol constitutive adequacy**, not
proof that a material-point ODE or FE job can execute. Software foundations can
be accepted independently; the P92 industrial material/life claim cannot.

This batch completed the planned literature-led replacement, equal-budget
identification and frozen comparison, but did **not** achieve the requested
physical target: SLC improves in several observables while SLR cycle-10 peak
error increases from 7.6481% to 10.0675%. Reject the candidate. A completed
experiment and a completed material qualification are different milestones.

## Acceptance language

The public scope of [ASTM E2714-26](https://store.astm.org/standards/e2714)
supports separating cyclic deformation, creep/relaxation and crack-formation
observations, with complementary fatigue and creep data for the same material
and temperature. [ASTM E139-24](https://store.astm.org/standards/e139) concerns
creep/rupture tests; [ISO 12106:2017](https://www.iso.org/standard/64687.html)
describes axial strain-controlled fatigue. These are test-method scopes, not
universal constitutive-error tolerances or a software certification.

Our unchanged development screen uses stress limits max(5 MPa, 5% of the
declared observed magnitude), strain limits max(2e-5, 10% of the declared
excursion), and 20% relative secant error only where at least five recording
quanta are resolved. Unresolved observations stay unresolved. These are
project gates, not tolerances attributed to ASTM or ISO; engineering use still
requires an explicit application domain, appropriate data and uncertainty.
