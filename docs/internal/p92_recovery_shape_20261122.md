# Existing recovery-shape coordinates in place of weak static coordinates

The [Seifert, Schlayer and Maier 2025 conference paper](https://opus.hs-offenburg.de/files/11628/SeifertSchlayerMaier_PlastCrpFat_FVWHT_2025.pdf),
equation (4) and pages 39–41, treats the normalized-backstress recovery
exponent as a material quantity and compares mixed-control responses with
and without its effect. This is the already reviewed BED/519n, 600 C source,
not a new BAM material card. Its qualitative comparison supplies motivation,
not a numerical acceptance threshold.

The present reference has four such exponents fixed at 3. The earlier
directional and critical-surface replacements change the recovery rule;
neither identifies the smooth rule's existing shape. Test only the two B
exponents while preserving both Kelvin strains and the original direction
rule. Replace the two weak A static-recovery coordinates (indices 7,8),
holding those static coefficients at their existing values. No new material
coefficient or state is introduced; two previously fixed existing coefficients
become variable and two previously variable coefficients become fixed.

Start exactly at the frozen baseline, chi_B1=chi_B2=3. Use log10 multipliers,
the same 22 candidate coordinates, retained dimension 10, norm bound 0.3,
two finite-difference scales, maximum 40 optimizer evaluations and unchanged
early cycles 1–3 objective. Require the existing derivative/subspace tests and
two-direction observability test before optimization. Do not enlarge bounds
or change the data partition after seeing the result.

First verify zero-coordinate identity, isolated substitution, invalid-combination
rejection and round-trip card identity. Positive exponents preserve the
existing dissipation and backstress domain. If identification converges,
freeze the card and inspect all original response checks on cycles 1–10,
including independent integration and refined measured waveforms. Previously
inspected later cycles remain development checks, not blind validation.

Report creep accumulation, first sampled hold stress, full hold error and
both secants separately. A lower average error alone is not admission.

The run starts from the already matched original card. Consequently the
archived original fit alone is not an equal additional-budget control: if
the new-coordinate audit admits optimization, rerun the original coordinate
set from this same starting card with the same additional budget before
attributing improvement to the replacement.

## Matched result: no demonstrated mechanism gain

| Arm, same starting card and additional budget | Joint cost | Optimizer / residual calls | SLR cycle-10 peak error |
| --- | ---: | ---: | ---: |
| Original coordinates | 0.002275663761 | 9 / 259 | +8.1897% |
| Existing B recovery-shape coordinates replacing A static coordinates | 0.002275795828 | 8 / 238 | +8.2101% |

Both converge. The new-coordinate cost is 0.0058% **higher** than the control.
The approximately 3.6% reduction from the starting card occurs in the original
law too; it is not evidence for the mechanism replacement. All six frozen
protocol paths pass independent integration and waveform refinement; all six
experimental response screens fail. No further restart or enlarged box is
admitted by this experiment.

The first SLC post-creep hold endpoints are 162.430 and 162.410 MPa,
respectively, versus 174.880 MPa observed. The replacement's first-sample
residual is only -0.493 MPa: the continuing problem is hold shape, not merely
a common starting offset. Its early/late secants are -5.183/-0.08234 MPa/s
versus -3.794/-0.18193 observed. Both reversible strains remain present.

The new identity-checked paired report recomputes every retained screen from
its observations rather than trusting a status label. It verifies common
source, seed, budget and driver identities and the exact fitted/frozen card
binding, and retains numerical and experimental conclusions separately.
Archives are the `2026-11-22-recovery-shape-*` records in `evidence/creep/`.
A later complex-tail-input rejection changes no real-valued numerical path;
archived execution hashes remain those actually used, not rewritten hashes.

The paired report can also be rebuilt directly from the four repository
archives. For losslessly archived fits, it reconstructs the original serialized
payload and checks its hash before accepting the provenance link. It checks
the actual retained-coordinate box, not only the reported budget label.

```sh
python -m tools.p92_matched_response_report \
  evidence/creep/2026-11-22-recovery-shape-control-matched.json \
  evidence/creep/2026-11-22-recovery-shape-matched.json \
  evidence/creep/2026-11-22-recovery-shape-control-frozen.json \
  evidence/creep/2026-11-22-recovery-shape-frozen.json \
  /new/path/recovery-shape-paired.json
```

No source download or material solve is needed for this report reconstruction.
Input-file hashes differ from the original temporary-file invocation, while
costs, numerical conclusions and all retained response checks reproduce.

The report also separates observation resolution from model error. All ten
SLC late-creep secants remain below the predeclared signal requirement; no
parameter change can make those particular recorded windows resolved. They
remain unresolved, never passed or discarded. This does **not** explain the
large, resolved post-creep hold mismatch. Both obligations remain: improve
the model on resolved observations, and obtain adequate observation support
before claiming an all-checks pass. A numerical fit cannot manufacture data.
