# Figure-based lifecycle verification — 2026-10-08

**Overall decision: incomplete, not approved for Table 5 promotion.** This
follow-up closes bounded lifecycle checks, not the outstanding spatial and
formulation convergence requirements.

The restart and finite-difference drivers now accept an explicit
`--geometry-source`; every rebuilt or perturbed problem receives it. Restart
comparisons retain that identity as an exact check. Existing prose geometry
remains the default for compatibility; all new evidence below explicitly uses
`figure-10a`.

## Executed problem and source

- Clean commit: `ab73adaa283509cfa63fbf9b8d1b7745a6a449af`.
- Scientific runtime SHA-256:
  `59d72e8165bd9dc64d3216b1a8d802f82462065955d38d2d3bb411d51b112833`.
- Q9/DPC1, curved quadratic geometry, 285 cells, degree-4 quadrature,
  20 fixed load increments. This coarse discretization is a lifecycle test,
  **not an accurate Table 5 energy reference**: its energy error is about 45%.
- No material, Newton tolerance, or scientific acceptance threshold changed.
- Source remained clean and unchanged during all numerical executions.

## Passed checks and exact scope

| Check | Outcome |
| --- | --- |
| Restart 1 rank → 2 ranks | Passed; largest recorded numerical difference `8.83e-15` |
| Restart 2 ranks → 1 rank | Passed; largest recorded numerical difference `6.45e-15` |
| Direct serial/MPI response | Passed; first-Piola maximum absolute difference `2.25e-15`, primal-energy difference `1.52e-17` |
| Independent Q9/DPC1 interpolation oracle | All eight checks passed; curved physical patch error `1.08e-15` |
| Local fixed-old-state macro tangent | Passed existing audit on `1e-4 … 1e-7`; scope and failed larger perturbation below |
| Artifact integrity | All 45 tangent-candidate manifests and the MPI response manifest verified |

Both restart directions compare the resumed result against an uninterrupted
solve on the reader communicator. Increment histories, mesh and constraint
identities agree. Direct MPI comparison also checks the same executable mesh,
scientific source and prescribed path, not just matching scalar outputs.

MPI macro-tangent condensation remains **unsupported**. The first MPI attempt
correctly rejected it; the response comparison was rerun with `--skip-tangent`.
Do not interpret that response equivalence as an MPI tangent verification.

## Full finite-difference record, including failure

Each scale uses a base solve and eight signed perturbations with identical
pre-final history. Errors are dimensionless relative Frobenius norms. The
unchanged point-check tolerance is `1e-3`.

| Perturbation | Relative error | Point check |
| --- | ---: | --- |
| `1e-3` | `2.427611338e-3` | **Failed** |
| `1e-4` | `2.790953464e-4` | Passed |
| `1e-5` | `5.207383579e-8` | Passed |
| `1e-6` | `2.656899110e-7` | Passed |
| `1e-7` | `2.654432847e-7` | Passed |

The four accepted local scales share identical analytical tangents and problem
identities. Their first observed error-reduction order is 3.729 (passing the
existing ≥1.5 entry screen), and their finest successive FD-matrix change is
`1.73523e-9` (below `1e-6`). This is not a claim of exactly second-order
asymptotics over every interval: the finest errors plateau and are not
monotonically decreasing throughout. The `1e-3` result is retained in the
archive and table, excluded explicitly from the **local** accepted interval,
and is not relabelled as passed. No global five-scale pass is claimed.

## Remaining decision

The content-bound lifecycle audit still reports these missing gates:

- load-increment path convergence;
- spatial convergence;
- quadrature convergence;
- plane-strain formulation convergence;
- periodic supercell invariance.

The [spatial follow-up](2026-10-08-spatial-followup.md) remains separate:
its finest global energy change is `0.72236%`, above the unchanged `0.5%`
contract. Local stress and global tangent also fail their decreasing-change
screens. Historical runs from another geometry or source identity are not
silently combined into a new complete certificate. Fine-discretization and
formulation evidence must be completed before promotion; this note does not
authorize a release maturity change.

## Archive, tests and execution notes

Maintainer archive: `outputs/rve-final-checks-20261008/evidence.tar.gz`.
SHA-256:
`8de4ce1a8854d7c5b581bc8e72cf0aa2ed95cae51f64e81c65edbb4b9e08de3f`.
It contains all five tangent studies, both checkpoint graphs, both restart
reports, the direct MPI response, the independent element oracle and audits.
Original numerical artifacts were not rewritten.

Targeted regression: **68 passed, one explicit unpromoted-benchmark skip**;
Ruff passed. The preceding commit `9534a0dd` additionally passed remote
FEniCSx and macOS installed-wheel acceptance in run `37715987269`; that CI run
is not evidence for later, not-yet-pushed commits.

Initial command failures were preserved as execution findings: use the actual
returned `*.checkpoint.json` filename, and run the tangent orchestrator
directly rather than inside an MPI rank (its subprocesses otherwise inherit
invalid PMI handles). MPI runs used the environment-matched launcher and a
600-second process-group timeout; tangent studies had 1200-second limits.
No numerical jobs remain running at this handoff.
