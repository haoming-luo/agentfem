# P92/history composition acceptance, 2026-11-06

This is local integration evidence, separate from each PR's latest CI and merge
status. It is not a release, clean-operating-system installation or native
accelerator performance test.

## Frozen combination

Local integration commit `39d19913efb0e83b8759561b608a8d7183ae2d9b` combines:

- `ef1a0f82`: rate-material checkpoint clocks (and its temperature/history base);
- `38120ab3`: P92 isothermal/structural cyclic checks and 435-cycle reports;
- `e29262bc`: reviewed main rupture/life work and the evidence-map documentation.

Merge conflicts were only overlapping CI-map entries and roadmap sections;
both sets were retained. No solver conflict or altered numerical tolerance.
The combined candidate is not an alternative path around branch protection.

## Checks

- 273 tests passed, one serial skip, in 81.39 seconds: physical-time material
  Steps, selected histories, thermal inputs and clocks, checkpoints, P92
  structure/cycles/life/rupture, generic rupture curves, documentation and CI map.
- The skipped two-rank admission test then passed on both ranks via
  `agentfem mpi-run -n 2 --timeout 180`. It verifies rejection of the unadmitted
  MPI user-material combination, not support for it.
- Built a pure-Python wheel from `git archive` of that exact commit with
  `AGENTFEM_BUILD_NATIVE=0`, installed into an isolated temporary target using
  existing `fenicsx-env` dependencies, and checked the imported package path.
  Wheel SHA256: `d2650ad5902aff8f70ec4162823d0cfc7c8482f3a9004124d05b95c72acf860d`.
- Ran `examples/small_strain_relaxation` through installed `agentfem run`
  with `--material-point`, then a separate run with `--restart` pointing to its
  midpoint checkpoint. Both completed 16 increments and published results.
- Compared all selected `sample.E`, `sample.EV`, `sample.S` history records:
  exact equality. All seven datasets in final `results.h5` were exactly equal.
  `agentfem verify` verified the resumed manifest's provenance seal; this seal
  is not experimental verification of the material law.
- Read both final XDMF files with VTK's real XDMF reader, applied displacement
  `U` through `vtkWarpVector`, and verified one unstructured grid, 16 cells,
  correct warped coordinates and retained `S_CELL`/`EV_CELL` arrays. The first
  probe used the display label `Displacement`; inspection identified the actual
  exported array name `U`, and the corrected probe passed without editing output.

Numerical checks used `FI_PROVIDER=tcp`, `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`. No user installation or original project was changed.
The first test command had a misspelled test filename and collected no tests;
the corrected complete command produced the count above. An initial comparison
script incorrectly indexed the manifest's history-name list; the corrected
comparison used the actual `history_records` objects, without changing results.

## Scientific boundary

The installed example is a synthetic relaxation law, not P92 qualification.
The separately archived 435-cycle P92 computation checks structure/time
integration against a continuous reference. Engineering life still requires
compatible independent observations and the selected application/failure
criterion; integration success does not supply those missing data.
