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
