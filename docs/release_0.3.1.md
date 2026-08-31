# AgentFEM 0.3.1

AgentFEM 0.3.1 turns reliability and installation into explicit product
contracts without moving either concern into the finite-element solver.

## Reliability and support

Basic anonymous reliability reporting is on by default and permanently
user-controllable. Its schema contains only coarse runtime, command, outcome,
duration and failure-class information. Models, meshes, material parameters,
source code, paths, exception messages, tracebacks and results are excluded by
construction and regression-tested.

Repeated failures can be diagnosed locally or packaged as a private task for
Codex. A richer report remains on the machine unless the user explicitly asks
AgentFEM to create an authenticated GitHub issue.

The reference collector stores daily aggregates rather than raw events. The
0.3.1 package does not claim online delivery until a reviewed, project-owned
HTTPS endpoint is deployed.

## Desktop runtimes

The Complete Runtime includes AgentFEM, FEniCSx, PETSc, MPI, a form compiler,
HDF5 and Gmsh with its license and corresponding-source evidence.

- Windows receives one WSL2 offline Preview ZIP with an image, verified
  installer, integrity records and legal materials.
- Apple Silicon macOS has two deliberately distinct routes: an explicitly
  unsigned Preview, and a formal package that may be published only after
  Developer ID signing, Apple notarization, ticket stapling, Gatekeeper checks,
  and clean-runner installation all pass.

Every runtime is assembled from the exact published wheel and uploaded to the
matching GitHub Release only after its platform-specific acceptance step.
