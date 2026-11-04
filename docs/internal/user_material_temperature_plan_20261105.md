# Prescribed temperature for history materials: bounded implementation plan

## Product decision

Extend the existing finite-strain UserMaterial Step, not the Model vocabulary
or global solver. Accept `temperature=` as a positive scalar in kelvin or an
existing scalar `histories.temperature(...)` with linear interpolation and
physical-time coverage. Freeze the sampled input at Step construction. Do not
mutate the history's live source or infer a thermal equilibrium solution.

## Ownership and numerical contract

- Material receives endpoint T and T(end)-T(start), at the same physical times
  as F and committed history. Its algorithmic dP/dF holds prescribed T fixed.
- Procedure visits the union of mechanical knots and temperature knots; cutback
  subdivides both time and temperature consistently. Every Newton candidate
  sees the same old state and endpoint temperature.
- The existing response driver forwards thermal channels to scalar, batch and
  plane-stress updates. It retains collective failure and transactional state.
- State publishes accepted TEMP with stress/history; failed trials restore it.
  Checkpoints bind the frozen thermal input and verify the saved temperature
  against the accepted time, without a fictitious material update.
- Result identifies one-way prescribed thermal input, not heat flux, heat
  balance or thermomechanical energy closure. Units remain kelvin; time values
  use the Step's declared time unit, without implicit conversion.

Spatial histories, external fields, thermal feedback, MPI promotion and new
temperature-dependent material laws are separate work. Reuse the existing
serial 3D/plane-strain/explicit plane-stress boundaries.

## Verification before admission

1. Scalar and frozen history validation; reject nonpositive temperatures,
   missing coverage, non-time coordinates, wrong units and spatial histories.
2. Independent homogeneous thermoelastic expansion and constrained reaction;
   fixed-old-state tangent and temperature-dependent history accumulation.
3. Mechanical/thermal knot union, failed-trial rollback, cutback partition of
   delta-T, continuous versus restarted accepted response.
4. Changed thermal history rejects restart atomically; no-temperature existing
   checkpoint identities remain unchanged.
5. TEMP accompanies the same accepted mesh/time in result output, including
   partial continuation. No automatic additional heat solve or energy claim.

The existing local material input already distinguishes temperature and its
increment. This plan closes the structural consumer gap; it does not infer
material parameters or validate P92 recovery physics.

## Local evidence

The combined history-temperature, finite-strain structural, plane-stress,
batch-evidence and stress-conversion regression completed with **230 passed**
in 63.37 s in `fenicsx-env` (serial, thread counts limited to one). This includes
both ordinary CLI example variants. No full composite solve or performance
claim follows from this test duration.

After adding an accepted-temperature mutation guard test and regenerating the
knowledge reference, the temperature, CI-scope and documentation checks passed
**87 tests** in 6.19 s. These overlap the earlier 230-test run and are not
reported as 317 distinct tests. Ruff and `git diff --check` also passed.
Two further checks verify both thermal example providers' tangents after a
history-producing preload and their finite rigid-rotation objectivity. The
final temperature-only suite passed **21 tests** in 3.70 s.

## Isolated installed-use check

Commit `cbddfc29` was frozen into a temporary pure-Python wheel and installed
without changing the existing installation, in a temporary venv sharing the
already installed FEniCSx dependencies. An isolated interpreter confirmed the
import came from that venv's `site-packages`, not the source checkout. This is
not a clean-OS dependency installation or native-accelerator benchmark.

Ordinary `agentfem run` completed the viscoelastic thermal example and the
plane-stress damage example. The latter was run both continuously and as
pause-at-0.5 plus restart. All ten numeric checkpoint arrays (displacement,
effective gradients, state, stresses, tangent, stored-energy channel and TEMP)
matched exactly in this run. Independent VTK reads and Warp By Vector checked
all 17 output frames across these runs; temperature and stress stayed on the
same grid and cell counts were unchanged. Partial output remains readable;
the resumed output starts at the restored physical time, without fabricating
unsaved earlier fields.

The generated API reference was subsequently synchronized after CI identified
it as stale. `build_docs.py --check` and the site build passed locally; generated
site files were not committed. No release package was distributed.

The independent patch uses an isotropic Green-strain thermal shift. Free 3D
expansion gives `Fii = sqrt(1 + 2 alpha delta_T)`; constrained axial nominal
stress is compared with the analytical transverse-relaxation solution. This
tests structural plumbing and fixed-temperature tangents, not a calibrated
large-temperature engineering constitutive law.

Design reference: [MGIS state management](https://thelfer.github.io/mgis/web/bindings-cxx.html)
separates beginning/end material states, external variables and update/revert.
AgentFEM uses its existing corresponding material/State contracts rather than
adding an MGIS runtime dependency.

## Shared history input and reader regression

The review reproduced silent complex-to-real truncation in five new history
tests before the fix. Real-valued histories now reject complex sources,
frames, sample coordinates and archive arrays before conversion. A corrupted
archive cannot reuse the real-valued content hash after losing its imaginary
part. Rejection preserves the live field and accepted history.

The final combined history, temperature, CI-scope and documentation run passed
**109 tests in 7.55 s**, including a real VTK read-and-warp regression for the
thermal patch. These counts overlap earlier runs. The existing distributed
temperature-history portable-archive test also passed on two ranks using
`agentfem mpi-run` with a 180-second timeout. That test covers the shared history
archive, not MPI support for the new temperature-dependent structural Step.
