# AgentFEM 0.4.2

This release brings reusable explicit-simulation execution into the installed
core, improves ordinary project execution and introduces a bounded legacy
VUMAT batch bridge. A project retains its physical definition and user material;
the software owns numerical execution, accepted state and result export.

## Installed execution and results

- Material-independent native Hex8 body execution and batched Q1 contact
  geometry are available through core factories. User materials remain separate.
  Worker counts are configurable execution choices, not hardware-specific
  performance promises.
- Explicit numerical-mass policies record physical and numerical mass, budgets
  and initial increment controllers. Initial elastic information is not a
  certificate of nonlinear-path stability or accuracy.
- Accepted material fields, contact resultants, moments, work and coverage
  follow shared State and Result contracts. Checkpoint recovery retains history;
  failed increments are not published as accepted observations.
- Standard `agentfem run -- CASE_ARGUMENTS` forwarding and sibling imports work
  like ordinary Python project execution. Failure/interruption remains visible.
- Volume-cell connectivity and single-grid field export have targeted reader
  regressions. Coincident interface nodes are not welded by coordinate guessing.

## Reviewed user-material sources

The experimental `constitutive.compile_vumat` API compiles trusted legacy
three-dimensional VUMAT source with GNU Fortran, caches by build identity and
updates a whole block in one native call. State and field counts are arbitrary;
all physical inputs and tensor conventions are explicit. Old-state arrays are
protected and unavailable energy remains unavailable.

This first profile supports the legacy scalar-`dt` interface on macOS,
Linux and WSL2. It is not a complete Abaqus runtime: objective kinematics,
stable stepping and global material adapters must still be verified. Modern
`dtArray`/`jInfoArray`, shells, annealing and vendor utility calls are not added.
See [the material compatibility boundary](abaqus_user_material_bridge.md).

## Verification and scope

The development candidate passed 394 targeted installed-package tests without
skips, including native execution, material compilation, state handling, CLI,
contact work, output topology and packaging. Its clean-source macOS installation
also passed standard static, thermal and dynamic project workflows. Formal
publication additionally requires the exact release artifacts to pass the
Linux/macOS and scientific checks in the publish workflow.

Experimental physics remains experimental. This release does not establish
arbitrary VUMAT compatibility, full industrial forming capability, independent
Abaqus equivalence, or a hardware-independent speedup. Historical private-case
timings belong to their recorded packages and numerical inputs. Complete offline
installers are not rebuilt as part of this package release.
