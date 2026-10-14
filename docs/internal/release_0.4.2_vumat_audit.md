# 0.4.2 candidate: user-material and delivery audit

Date: 2026-10-14. Base: `ef293cde`. Scope: release preparation, not promotion
of experimental constitutive/contact capabilities.

## Changes and ownership

- Core now owns the reviewed legacy 3D VUMAT build/ABI/batch transport profile.
  A law's constitutive formula stays in user source. No confidential material,
  case, mesh, or reference results were copied into the repository.
- Parameters, state count and field count are not case-specific constants.
  Compiler caching is keyed by source graph, wrapper, flags and platform.
  One block uses one native call; existing optimized material/body execution
  was not replaced by this compatibility route.
- Procedure/State remain the sole acceptance owners. The bridge protects old
  arrays, returns uncommitted responses, and neither invents a tangent nor
  mistakes specific internal energy for elastic storage.
- Public API, CLI inspection, generated API reference and migration guidance
  describe the same bounded profile. VUMAT does not become another Step type.

## Defects found and fixed

1. Empty immutable array slices could have a different sentinel pointer and
   fail shared storage bounds validation. Zero-width channels now retain the
   immutable contract without meaningless pointer subtraction.
2. The user-subroutine inspector missed no-parenthesis `CALL XIT` and similar
   calls. No-argument calls and the explicit-runtime exit utility are detected.
3. Build cache corruption and collisions with generated artifact names now
   produce explicit failures rather than obscure loader/build errors.
4. Input array metadata is revalidated immediately before crossing the native
   boundary, even when the request dataclass is frozen.

## Verification performed

- Repository-wide configured Ruff checks: passed.
- Generated documentation check and strict MkDocs build: passed; output was
  directed outside the existing `site/` tree.
- Source archive -> macOS arm64 wheel -> separate target installation:
  **394 targeted tests passed, no skips**. Checkout import injection disabled.
  Coverage includes compiler execution, tensor shear, arbitrary/empty state,
  optional energy, two independent laws, repeated history, native body/Hex,
  CLI arguments, checkpoints, response substeps, mass policy, contact work,
  XDMF topology/layout and packaging. This is not the complete repository suite.
- Standard installed examples and static/heat/dynamic project templates passed
  their execution/provenance checks. The first platform certification correctly
  failed because the working tree was dirty; a clean-source certification is
  required separately. Preserve this failure instead of disabling the gate.
- Publish workflow now requires a real GNU compiler and installed-wheel VUMAT
  tests; a compiler-dependent skip cannot silently substitute for that step.

Local detailed logs are in `/private/tmp/agentfem-vumat-release-audit/`; this
temporary location is not a public release artifact or a reproducibility URL.

## Remaining release gates and exclusions

1. Review/integrate this candidate through PR and applicable remote checks.
2. Certify an exact clean source revision and its installed wheel; update the
   release/version/citation metadata together before tagging 0.4.2.
3. Keep Linux/WSL2 and macOS evidence separate. Do not label the collaborator's
   Windows machine tested from this Mac result.
4. The adapter has not established arbitrary finite-rotation VUMAT kinematics,
   deletion or global MPI semantics. Those remain explicit adapter work, not
   hidden defaults or promises of full Abaqus compatibility.
5. Historical full private-case timings remain bound to their original package
   and numerical inputs. This audit does not measure a new full-run speedup,
   validate numerical-mass sensitivity, or provide independent Abaqus parity.

The release can ship a bounded experimental material bridge without waiting
for universal VUMAT support. It must not claim the latter to close the former.
