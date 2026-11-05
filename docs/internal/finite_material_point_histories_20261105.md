# Finite-strain material-point history consumer

## Design before implementation

Reuse `results.material_point` and its existing recorder in the ordinary
finite-strain UserMaterial Step. No project sampling loop, new file format,
global solver or constitutive model is needed. The initial scope follows the
existing serial homogeneous 3D/plane-strain/declared-plane-stress Step.

Record Cauchy `S`, first Piola `P`, effective three-dimensional `F`, total
Green–Lagrange `E_GL`, named state and supplied kelvin `TEMP`. `E_GL` includes
thermal/inelastic deformation and is not mechanical strain or a ready-made
strain-life input. Plane-stress F33 is the converged thickness stretch.
Units are inherited from the Model and state schema, never guessed.

Select original input-cell IDs and reference quadrature indices. Prepare a
detached frame before committing material state; append only after successful
commit. Invalid output must not leave a partial commit. Checkpoints bind the
selection, units and complete history to accepted physical times. Validate a
restored history before changing live fields; legacy checkpoints without this
option retain their identity. Do not copy accumulated history on every Newton
trial or output frame.

## Acceptance

- Two locations and all supported dimensions preserve raw tensor association.
- Temperature, state and physical time share one accepted ledger.
- Failed attempts/commits add no samples; retry matches a continuous path.
- Pause/restart retains the complete history; corrupt or changed selections
  reject atomically, including an already advanced receiving Step.
- Sparse field output does not decimate history; ordinary NPZ result history
  round-trip works without new serialization.

The design retains integration-point state as in the
[FEniCSx plasticity tour](https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html).
It does not use visualization averaging: the
[PyVista cell-to-point filter](https://docs.pyvista.org/api/core/_autosummary/pyvista.dataobjectfilters.cell_data_to_point_data)
explicitly averages neighboring cell values and is not a material history.
This slice provides verification data, not a validated P92 law or life model.

## Executed evidence

The affected numerical/lifecycle/CI/documentation run completed 415 tests with
one stale generated-knowledge-document failure. Regenerating that reference
and rerunning the documentation/CI group passed all 68 tests. The numerical
group includes 348 checks; counts overlap and must not be added as independent
evidence. Ruff, generated documentation and knowledge import checks pass.

An intermediate selected-array optimization accidentally shadowed its boolean
selection flag. Nonuniform/cross-cell and existing creep regressions caught it;
the corrected recorder passes both consumers. Observation-on/off tests retain
identical displacement, deformation gradient, constitutive state and accepted
times for 3D, plane strain and plane stress. Nonuniform fields are checked at
four original cell IDs rather than only at equivalent points of a uniform patch.

A separate virtual environment, sharing the existing FEniCSx dependencies but
installing a newly built wheel with no source-path override, ran the ordinary
four-file project through `agentfem check/run`. Damage + plane stress + uniform
thermal cycle + selected point recording completed 18 accepted increments with
five visualization frames. A nine-increment pause and restart retained all 18
point samples; final U/F were identical and all selected tensor/state/temperature
histories matched the uninterrupted run within 1e-12. Existing Newton diagnostics
remain execution-segment histories, not falsely reconstructed pre-restart logs.
Native accelerators were disabled for this installation check; no new native
performance or clean-host installation claim follows.
