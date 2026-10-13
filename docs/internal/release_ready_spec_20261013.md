# Cahier des charges — reusable explicit workflows

## Outcome

Deliver a readable private case, reusable installed AgentFEM execution, and a
release candidate whose behavior and performance are evidenced. Do not create
a second solver, a composite-specific Model, or a new modeling language.
The private model and source material stay outside this repository.

## Acceptance

| Requirement | Acceptance evidence |
| --- | --- |
| Readable case | Mesh/regions, material, interaction/motion, Step and outputs are identifiable without reading the launcher. No time-integration loop, native kernel, manual connectivity, or raw checkpoint serialization in the modeling file. |
| Core ownership | Installed factories own body arithmetic, Q1 search, worker dispatch, trial/commit/rollback and field writing. Private code owns the constitutive equation and explicit source-model preparation only. |
| Performance | Same mesh, law, loading, numerical inertia, time grid and reporting scope as the retained baseline; measure fresh process through export/shutdown. Target approximately four hours on this machine; report actual time, not an extrapolated pass. |
| Numerical parity | Compare accepted nodal/material state, damage, work and contact evidence at relevant stations and completion. Algorithm changes require stated tolerances/sensitivity rather than cosmetic checksum parity. |
| Output | An independent reader sees volume cells and complete displacement on one mesh; Warp reproduces reference coordinates plus displacement. Preserve discontinuous material fields and independent interface nodes. |
| Recovery | Failure cannot publish partial numerical state; accepted checkpoints resume only under the supported identity contract. Reporting is not another state owner. |
| Distribution | Source archive, installed wheel, native availability/fallback, platform identity and public examples pass the applicable gates. A successful PR workflow does not prove the separate publication workflow is correct. |

## Case design

Abaqus separates model data and history data. Adopt that readability, not its
keyword syntax: ordinary Python remains AgentFEM's modeling truth. Keep named
physical/numerical parameters visible, operational reporting in a separate
runner, and provider implementation behind its existing public contract. A
short file obtained by hiding all physics in a helper does not pass this gate.

Separate source-deck fidelity from software parity. The current reviewed
experiment retains explicit initial adjustment, numerical inertia, elastic
bonding and one-sided penalty contact. A matching AgentFEM implementation is
not by itself equivalent to the source deck's hard contact / ALL EXTERIOR /
variable-mass-scaling semantics. Do not remove this distinction for presentation.

## Bounded execution plan

1. Keep the current full run and installed runtime untouched. Reuse its frozen
   evidence instead of starting another full run after every cosmetic edit.
2. Refactor a separate delivery candidate into modeling and execution files;
   compare constructed scientific settings and short accepted behavior before
   attributing existing full-run evidence to it.
3. Audit core migration and publication routes. Correct demonstrated ownership,
   packaging or output gaps; avoid an unrelated whole-repository rewrite.
4. Review complete timing, same-station results and independent reader output.
   Prioritize only a measured remaining bottleneck if the release target still
   requires it. Keep follow-on research outside this release gate.
5. Record ready / failed / pending per gate. Preserve experimental capability
   labels and unsupported combinations; do not equate release readiness with
   general industrial-forming validation.

## Methods are revisable

The local performance skill is an engineering aid, not an authority. Explore
cheaply, test changed boundaries proportionately, and reserve installed/full-run
acceptance for selected candidates. Repeated finite scans or copies may be
removed when explicit ownership/admission proves the same invariant; skipping
coverage or caching arbitrary mutable arrays is not that proof.

## References

- [SIMULIA: model and history input](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMODRefMap/simamod-c-ov.htm)
- [PyPA: binary extension packaging](https://packaging.python.org/en/latest/guides/packaging-binary-extensions/)

## Acceptance checkpoint

Readable case, core ownership for the bounded execution route, fresh full-run
timing, implementation parity, one-volume-grid output and final-state recovery
have concrete evidence in [the candidate record](release_0.4.2_candidate.md).
The complete run measured 4 h 17 min 45 s. This meets the approximate four-hour
delivery target, not a strict under-four-hour threshold or the faster prior
private-backend time. Installed candidate checks and the unrelated public
classic example passed. Final PR checks and any later release publication are
separate gates; no stable tag is implied by this record.
