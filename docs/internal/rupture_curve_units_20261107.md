# Explicit rupture-table unit conversion

The structural result records consistent derived stress units, while public
rupture tables commonly use MPa and hours. Rebuilding tables or relabelling
accepted histories in each project is unnecessary and error prone.

Add `TabulatedRuptureCurve.in_units(stress_unit=..., time_unit=...)`: an
explicit, immutable table transformation. Support a bounded list of stress
and time units, including the existing SI and N-mm-MPa derived labels.
Keep temperature in kelvin and retain stress measure, material condition,
failure definition and source. Record conversion factors and original unit
labels in the returned curve's detached evidence. Existing assessment unit
checks remain exact; no implicit conversion enters a solver or checkpoint.

Acceptance: node/interpolated lives and assessed time fractions are invariant
after consistent conversion; inputs are unchanged; conversion history is
detached; unsupported units and unrepresentable numbers reject. This is not a
general unit-expression parser or a new material calibration.

Local acceptance: 121 targeted curve, assessment and real FEniCSx
material-point-history tests passed. The structural composition uses an
isothermal creep solve and a synthetic constant five-hour rupture curve,
checks the exact time fraction in seconds, retains conversion evidence and
confirms that accepted stress arrays are unchanged. It verifies software
composition, not a real-material rupture law.
