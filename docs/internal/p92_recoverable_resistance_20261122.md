# Recoverable-memory resistance: prospective single discriminator

The completed mobility fit lowers joint cost but increases first SLC early
relaxation and frozen SLR peak errors. At the approximately 210 MPa band,
the candidate's B contributions remain nearly equal in SLR and SLC, despite
their differing measured nonelastic rates. Do not refit that mobility law.

[Li et al. (2021), section 4, equations 7 and 14--17](https://www.dora.lib4ri.ch/empa/islandora/object/empa%3A26352/datastream/PDF/Li-2021-In-situ_and_ex-situ_microstructure_studies-%28published_version%29.pdf)
links bowing to strain and friction resistance in ex-service 316H at 650 C.
The full paper is now accessible through Empa's repository. Its coefficients,
FCC Taylor factor and density interpretation are not transferable to P92.
This motivates testing resistance feedback, not claiming a reproduction.

Our explicit scalar hypothesis is k_B_effective=hypot(k_B, H_slow*a_slow).
Keep both recoverable strains, original directional recovery, moduli, times,
energies and every existing coefficient. No fitted scale, exponent or new
state is introduced. The unit coupling is a fixed model-form choice, not a
deduced microscopic constant. The flow threshold changes; this is not the
previous multiplier on reference flow or static-recovery rate. Nonnegative
strength retains nonnegative flow dissipation and zero recoverable memory
recovers the original law. Both signs of recoverable strain give equal
resistance, which must be tested rather than assumed to describe real density.

Before seeing results: compare only original and this single strength arm,
from the same archived seed and measured controls, early cycles 1--3. Require
all paths complete, reduced joint cost, improved first SLC early relaxation
rate and no loss of its creep-endpoint gate before considering identification.
Also retain first-hold entry, end, late secant and SLR peaks explicitly. If
this conjunction fails, stop: no adjusted scale, extra arm or fit. If it
passes, use the existing 22/10 directions, radius 0.3 and 40-evaluation ceiling,
then freeze parameters for the unchanged ten-cycle screen. No promotion is
possible solely from lower training cost or thermodynamic consistency.

Verify rate/power balance, zero-memory identity, signed symmetry, recovery
after unloading, split continuation, independent integration and failure
retention before data use. The experiment remains research-only on PR #197.

## Fixed-card result: reject before fitting

Both paths complete, but joint cost increases from 0.00257608 to 0.00834338
(3.24 times). The first SLC post-creep early secant improves from -4.815 to
-3.751 MPa/s against -3.794 observed, while the late secant worsens from
-0.0918 to -0.0467 against -0.1819 MPa/s. Hold-end stress moves from
159.02 to 187.32 MPa against 174.88 MPa observed. Creep-end strain remains
inside its existing gate, but the inherited creep stress worsens. The joint
admission conjunction fails: no identification, scale adjustment or frozen
validation is started. This is negative mechanism evidence, not a material
upgrade. See `evidence/creep/2026-11-22-recoverable-resistance-probe.json`.
