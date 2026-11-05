# Accepted small-strain user-material histories

## Scope and ownership

Reuse the existing material-point recorder and physical-time small-strain Step.
The ordinary `model.step(material_points=..., duration=...)` records raw `S`,
total infinitesimal `E`, and named local state at selected original cell/point
identities. The legacy no-duration and MPI checkpoint paths are unchanged.
No extra material calls, projected fields, temperature inference or life model
are introduced. Units follow the model or state schema; unknown units stay unknown.

## Acceptance boundary

Prepare the detached point frame before committing state. Commit failure,
including failure after partial commit, restores displacement, response and
named history. Only then append a frame. Selection and units enter scientific
identity; geometry and quadrature are checked before advance, output and save.
V2 checkpoint manifests retain the existing hashed history payload and validate
its complete accepted-time ledger before restoring any live fields.

## Evidence

The targeted suite passed 133 tests with one dedicated two-rank admission test
skipped in serial. Tests cover raw tensor association, result NPZ roundtrip,
continuous versus restarted history, rejected increments, corrupt/missing
payloads, selection changes, and failures before/after constitutive commit.
The ordinary relaxation example has an opt-in `--material-point` flag.

This is a data/lifecycle capability, not P92 calibration. Further physical
validation must retain the recorded strain measure and original material point.
