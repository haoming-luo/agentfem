# P92 fixed-temperature history composition

The research tensor reference is isothermal. Its published parameter choice
already fixes 600 or 650 C; accepting a matching explicit kelvin value with
zero temperature increment does not interpolate parameters or add a thermal
strain model. This closes a useful composition gap: the generic structural
consumer can record co-located S, total E, TEMP and internal state on one
accepted physical clock, including restart.

Admit either the existing absent thermal channels or both matching fixed
temperature and zero increment. Reject a lone channel, another temperature,
or any nonzero increment. The initial configuration remains stress-free at
the selected test temperature; no ambient-to-hot heating simulation is implied.

Verification compares explicit and implicit isothermal declarations with exact
state/stress/tangent equality, then compares continuous and interrupted
structural histories at both parameter temperatures. This is numerical
composition of a research reference, not P92 nonisothermal, multiaxial or
engineering service-life validation.

## Evidence

On 2026-11-06, the real `fenicsx-env` targeted group
(`test_p92_tensor_structure`, `test_small_strain_temperature`,
`test_p92_tensor_normalization`, `test_ci_validation_scope`) passed 143 tests
in 21.09 s with one-thread numerical libraries. Both 600 and 650 C structural
histories match continuous execution exactly after checkpoint recovery.
The model explicitly declares the N/mm/MPa unit system; the recorded derived
stress unit is checked against that system rather than relabeling SI results.
This does not add a measured rupture curve or validate engineering life.
