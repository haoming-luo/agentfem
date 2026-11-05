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
coupled creep/plasticity, generalized Maxwell, the history-clock consumers
and CI routing. Ruff and whitespace checks passed. The first test attempt
incorrectly supplied a Step time-unit option and then compared a nested Maxwell
snapshot as an array; both test-harness issues were corrected, with no change
to physical tolerances. This run is serial; existing MPI ownership tests remain
part of the CI selection, not a claim of a new local MPI acceptance.

The producer now inherits the clock unit from the existing StepExecutionContext
when `capture_history` is used on a Model-owned transient heat Step. No extra
Step constructor state is introduced. Direct low-level capture can declare
`coordinate_unit`; a conflicting Model declaration is rejected before adding
a recorder. This closes the ordinary heat-to-mechanics transfer without asking
the caller to duplicate the same unit label.

The same check also rejects a FieldHistory labelled in Celsius, with no
temperature unit, or against load factor rather than physical time. The old
rate-dependent consumers unwrapped the numerical values and reported kelvin;
explicitly incompatible metadata must not silently enter those equations.
Raw scalar temperatures retain their documented kelvin convention.
The final thermal-label/unit/history regression group passed 88 tests in
23.35 s; it includes all three live rate-dependent consumers and both frozen
user-material temperature consumers. No material parameters or physical
tolerances were altered.

After this producer addition, 98 history/unit/CI-routing tests passed in 17.39 s
and 54 transient heat/operator/restart/journal tests passed in 3.12 s. Separately,
the unchanged distributed creep equilibrium and Maxwell relaxation/checkpoint
tests passed on two ranks (2 passed per rank, 0.74 s), with one bounded MPI job.
