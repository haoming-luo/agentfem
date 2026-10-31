# History-material integration acceptance — 2026-10-31

## Bounded engineering closure

Keep the five ownership boundaries and the existing public `model.step` route.
No new material family, solver loop, output framework or broader MPI/contact
admission is introduced. Integrate the already reviewed thermal/history/plane-
stress/checkpoint sequence, then the output-control slice, only on accepted CI.
The published 0.4.4 tag stays unchanged; moving source now identifies itself as
0.4.5.dev0 with a development contract. New workflows remain experimental.

The audit found three concrete reliability gaps:

1. `solve(until=...)` could silently convert booleans, complex NumPy scalars or
   one-element arrays. Both solve and result publication now require a finite
   real scalar, excluding boolean coordinates, before mutating state or output.
2. Translating the reference mesh preserves an affine gradient but invalidates
   construction/checkpoint identity. Hash reference coordinates and geometry
   dofmaps at construction and check at existing accepted-state boundaries.
   This is a fixed-reference total-Lagrangian contract, not ALE or remeshing.
3. The version check was skipped by shallow CI checkouts, allowing new source
   to identify as an already published release. The numerical gate now fetches
   tag history, and the development contract explicitly retains earlier limits.

## Installed project, not source-only acceptance

`tools/check_installed_history_project.py` is a reproducible engineering gate,
not part of the user case. It rejects source/editable imports, copies just the
four example files, removes inherited `PYTHONPATH`, and invokes the installed
CLI with the same interpreter. The Linux numerical CI runs it after installing
the candidate and the optional independent VTK reader.

For both educational laws, the gate runs planned pause, restart, continuous and
sparse-output paths. It compares every checkpoint component and common field,
checks the two-file scheduled retention policy, joins accepted segments and
warps all 16 times through VTK. Sparse output retains four frames without
changing the final state. These exact comparisons concern the same host and
runtime, not bitwise portability across hardware.

The isolated installed candidate passed all eight paths locally. Its package,
source and checker hashes are retained in
[the confirmation record](../../evidence/history_output_cadence/installed_project_confirmation.json).
The disposable environment shares the existing FEniCSx dependencies; this does
not claim a clean dependency installation, Windows acceptance, MPI support or
calibration of the synthetic materials. No full installer was built or released.

## Test boundary audit

The first focused run passed 219 material/checkpoint/output tests. A subsequent
expanded run passed 540 tests and skipped three explicitly two-rank tests before
an interruption-output test failed. Its child interpreter loaded an unrelated
installed 0.4.2.dev0, whereas the parent pytest used the checkout. That installed
writer lacked the current per-frame flush. The test now selects and asserts the
same package path as its parent, for both source and installed-wheel CI. No
HDF5 integrity or partial-index assertions were relaxed. The failed run is not
counted as a passing expanded suite; final rerun/CI must close that gate.

Packaging and CI-scope checks passed 77 tests; repository correctness lint,
generated documentation and 58 scientific function cards validate. Local
sandbox MPI initialization failures and intermittent execution-approval timeouts
are environment failures, not passing tests or evidence of a solver defect.

The previous short-path output-preparation measurements retain their original
source identity. This integration adds state guards and makes no new runtime
percentage or full-composite acceleration claim.
