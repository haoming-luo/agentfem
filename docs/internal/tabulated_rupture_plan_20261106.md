# Bounded tabulated rupture relation

## Purpose and ownership

The existing creep time-fraction consumer accepts a user callable. Provide one
reusable callable for reviewed isothermal tensile rupture tables so projects
do not repeat interpolation, units and domain checks. It belongs to assessments,
not constitutive evolution, and cannot change structural material state.

`TabulatedRuptureCurve` freezes strictly increasing positive stress nodes and
positive nonincreasing rupture lives. Log-stress/log-life interpolation is an
explicit numerical policy. Raw repeated/scattered specimens must first be
reviewed into a curve; no automatic sorting, fitting, averaging or extrapolation.
One positive kelvin temperature, material condition, failure definition, source,
stress/time units and stress measure are required. A caller must not supply
equivalent stress under a label calibrated for uniaxial tensile stress.

The existing assessment checks those declarations when this concrete relation
is supplied and records the complete curve in its result. Arbitrary reviewed
user callables remain supported. No built-in alloy data, temperature interpolation,
allowable-life margin, statistical confidence or regulatory approval is implied.

## References and tests

[NEML Larson-Miller documentation](https://neml.readthedocs.io/en/stable/larsonmiller.html)
separates rupture correlations from constitutive response and emphasizes the
underlying test database. This slice does not implement Larson-Miller or fit
its constant. [ANSYS logarithmic interpolation](https://ansyshelp.ansys.com/public/views/secured/corp/v252/en/ans_mat/mat_loginterpscal.html)
documents logarithmic table interpolation as an explicit material-data policy;
we independently verify our much narrower isothermal life interpolation.

Acceptance: exact nodes, power-law midpoint, unit-invariant damage, immutable
inputs, malformed table rejection, out-of-domain stress/temperature rejection,
same-location sampled-history composition, source retention and unchanged
callable-based assessments. Synthetic coefficients only; real data admission
remains a separate scientific decision.

## Candidate verification

184 targeted tests passed across the new table, sampled creep history,
interaction, strain-life, documentation and CI-routing suites. Exact table
nodes and analytical power-law interpolation agree; Pa/h and MPa/s forms of
the same history give the same damage. Inputs are detached from caller arrays,
and results retain the complete source-identified relation. Boolean callback
returns now reject instead of being interpreted as a one-unit lifetime.
The first documentation run caught a stale generated knowledge index; it was
regenerated and the complete group rerun. No P92 parameters are embedded.
