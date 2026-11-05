# Rate-dependent history clocks

Extend the declared FieldHistory clock check to the existing pure-creep,
coupled creep/plasticity and generalized-Maxwell structural consumers. These
Steps take their physical time unit from the Model unit system, not a new
Step option. Reuse the user-material temperature guard; do not introduce
conversion, another history type, or another material solver.

Check at construction before constitutive evaluation, and before each live
temperature application so later metadata changes cannot silently alter the
meaning of a running path. Unlabelled legacy histories retain the established
model-unit convention. Numeric interpolation, material equations, accepted
state ownership and output remain unchanged.

The acceptance tests compare labelled and unlabelled paths for identical
displacement and material state, reject mismatching or missing target units,
and check that a rejected live-history mutation does not change the state.
They cover all three consumers, including a spatial temperature field for
creep. This is unit admission, not temperature calibration or a new P92 life law.

The real `fenicsx-env` targeted group passed 161 tests in 106.77 s, including
coupled creep/plasticity, generalized Maxwell, the four history-clock consumers
and CI routing. Ruff and whitespace checks passed. The first test attempt
incorrectly supplied a Step time-unit option and then compared a nested Maxwell
snapshot as an array; both test-harness issues were corrected, with no change
to physical tolerances. This run is serial; existing MPI ownership tests remain
part of the CI selection, not a claim of a new local MPI acceptance.
