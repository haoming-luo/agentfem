# Unified result output — implementation and verification

Date: 2026-10-14. Branch: `feat/unified-result-output`.
Base: `36f4c1b0`. This is a development branch, not a new public release.

## Delivered

- `results.output_series` coordinates existing serial XDMF/HDF5 and optional
  ParaView backends, writes accepted frames immediately, and registers artifacts.
- Standard static field output, serial history export, and transient XDMF
  requests consume the common session. Explicit legacy PVD requests remain
  supported. Existing expert writer APIs are retained.
- MPI transient XDMF requests now keep XDMF/HDF5 and a PVD companion rather than
  silently substituting PVD alone. MPI interpolation to geometry degree is
  recorded; it is not advertised as lossless high-order state storage.
- Field semantics distinguish displacement, mode shapes and general vectors.
  Display and scientific formats use the same declared field alias.
- Segment coordinates, field schemas, file dependencies and HDF5 dimensions are
  checked. Incomplete sessions retain a marker. Restart uses a new segment;
  multiple segment artifacts are preserved.
- Publication checks declared output, preserves output-failure status, and
  exposes strict missing-output errors as `AFM-OUTPUT-INCOMPLETE` at the
  `result_output` stage. Scalar-only results need no XDMF. Optional presentation
  failure preserves successfully finalized field data.
- `custom_energy_output` demonstrates a custom nonlinear elastic potential,
  Newton solve, P2 displacement, DG0 projected stress/energy, five load levels,
  dual output and standard publication without private writer code.

## Numerical readback and rendering

ParaView 6.1.1 `pvpython` actually loaded every one of the five frames from both
formats, read the point vector and cell energy, applied Warp By Vector and
rendered the final state. Both formats contain 175 points per frame. After
matching nodes by coordinates (rounded to 12 decimals with uniqueness checked),
maximum displacement difference between the two formats was
**8.673617379884035e-19**. This compares serialized fields, not engineering model
accuracy. Screenshots use actual displacement scale 1.

Reproduction: run the installed `examples/custom_energy_output` project, then:

```sh
pvpython validation/result_output/paraview_readback.py RUN_DIRECTORY validation/result_output
```

The ParaView simple helper's zero-time handling was avoided by calling
`reader.UpdatePipeline(time)` explicitly; the zero-load frame was checked too.
The application emitted an optional OpenVKL library warning but completed
surface rendering. Evidence: `paraview_readback.json`, `warp_xdmf.png`,
`warp_pvd.png`.

## Serial and MPI checks

Tests cover snapshot preservation with a reused live Function, duplicate times,
empty output, changed geometry, non-displacement semantics, missing heavy data,
corrupted dimensions, directory relocation, segment registration, presentation
failure, publication status, generator export, and exclusion of failed trials.

The separate MPI driver tests 2D and 3D with two ranks, including rank-local
nonfinite input rejected collectively before I/O. The existing parallel
ParaView test passed, as did all **9 parallel transient tests on each rank**.
Final full-suite run: **2474 passed, 157 skipped, 2 failed**. The two failures
are existing baseline issues: `test_multi_void_fixed_stack_matches_versioned_golden`
(solver identity versus the versioned fixture) and
`test_unreleased_checkout_does_not_reuse_a_published_tag_version`
(the checkout still uses the published 0.4.1 version). Neither fixture nor
package version was changed to hide these failures.

After the final publication/CLI refinements, the related targeted suite passed
**159 tests**, and the latest output/publication/CLI contract suite passed
**44 tests**. These counts overlap and must not be summed. The final CLI test
checks that a partially published result with output errors returns nonzero.
An isolated installed wheel also successfully runs the custom output project.

## Allocation measurement

`stream_memory.py` measured Python-tracked allocation peaks in separate runs
on a 30x30 P2 mesh:

| Frames | Python allocation peak | Bytes if all primary arrays were retained |
|---:|---:|---:|
| 10 | 4,559,664 | 595,360 |
| 200 | 6,703,537 | 11,907,200 |

The increase is about 2.14 MB, versus 11.31 MB for retaining the additional
primary arrays alone. This supports the intended streaming behavior. It is not
a total-RSS measurement or a speedup claim. XML/index metadata grows with frame
count; existing solvers may independently retain accepted-state histories.

## Boundaries

Fixed mesh and supported field spaces are required within each segment.
Explicit raw material/checkpoint output remains separate. Scientific writer
errors raise; `strict=False` permits optional presentation failure, not silent
continuation of a damaged scientific writer. Neither XML checks nor a successful
ParaView rendering establish physical correctness. Collective preflight is
covered, not recovery from a killed MPI process or failure inside third-party
collective I/O.

No changes to model equations, material parameters, solver tolerances or
benchmark reference values were needed for this output work.
