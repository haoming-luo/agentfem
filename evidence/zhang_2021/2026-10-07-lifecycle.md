# Zhang Q9/DPC1 lifecycle diagnostics — 2026-10-07

These are local development diagnostics, not a promoted Table 5 Golden.
Full mesh/formulation convergence and agreement with the published stress,
primal energy and effective tangent remain open.

| Check | Observed result | Scope |
| --- | --- | --- |
| Independently remeshed 1x1, 1x2, 2x1, 2x2 geometric supercells | Maximum first-Piola change 0.7404%; primal energy change 4.1493% | Common mesh target 0.28, degree-4 quadrature; diagnostic thresholds 1% and 5%, selected by AgentFEM, not the paper |
| Serial versus two ranks | Maximum absolute first-Piola difference 2.39e-15 | Same executed discretization and accepted load path |
| One-rank checkpoint resumed on two ranks | Passed | Compared with uninterrupted reader-communicator solution; exact pressure modes, quadrature state and accepted history |
| Two-rank checkpoint resumed on one rank | Passed | Reverse direction of the same portable lifecycle |
| Independent Q9/DPC1 interpolation oracle | Curved physical affine-patch gradient error 1.08e-15 | Hand-written Q2 polynomials; DPC1 span `{1, xi, eta}`; no constitutive or external-reference promotion |

The supercell family is a bounded geometric size diagnostic. Its independently
generated meshes do not establish exact topology-replication invariance. The
1%/5% thresholds must not replace the stricter mesh-convergence contract.

## Reproduction

Use the FEniCSx environment and its own MPI launcher. Set `PYTHONPATH=src:tests`
for these repository verification drivers. The ordinary public user workflow
does not require this setting.

- `tests/zhang_2021_plane_strain_driver.py`: build each supercell with
  `--cell-repetitions NX NY --mesh-size 0.28`, a fixed maximum load increment
  of 0.05, degree-4 quadrature, and macro-tangent recovery disabled.
- `tests/zhang_2021_supercell_audit.py`: consume the four assessment JSON files.
- `tests/zhang_2021_parallel_equivalence.py`: consume one serial and one MPI
  assessment from identical mesh and loading controls.
- `tests/zhang_2021_plane_strain_restart_driver.py write CHECKPOINT`: save at
  load factor 0.5 with default mesh target 0.30 and 20 increments.
- Run the same driver's `read CHECKPOINT --output REPORT.json` with a changed
  communicator size. Repeat in the reverse direction.
- `tests/zhang_2021_q9_dpc1_oracle.py`: run the independent element oracle.
- `tests/zhang_2021_plane_strain_promotion.py`: supply `--supercell-audit`,
  `--parallel-audit`, and both `--restart-audit` reports. The loader recomputes
  supercell/MPI decisions from hashed candidates and verifies both checkpoint
  manifests and their nodal/quadrature payloads. Overall promotion remains
  incomplete until the other scientific gates pass.

## General software fixes found by these checks

Model IR construction now enters distributed material validation on every
rank. Portable checkpoint restoration separates collective identity/key
construction from local mapping failures before reporting errors collectively.
Coordinate keys tolerate one rounding quantum only for a unique, one-to-one
nodal mapping; source-node identifiers and cell-local modal identities remain
exact. Duplicate, ambiguous and many-to-one mappings fail before state writes.

Targeted regression: 87 passed, one intentionally unpromoted external
benchmark skipped. No full-suite run was required for this development slice.
