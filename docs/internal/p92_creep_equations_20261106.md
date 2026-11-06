# P92 creep equation reproduction: numerical closure, source discrepancy retained

The bounded reference is `tools/p92_creep_equation_reference.py`. It executes
the as-received equations, unlike the earlier published-table arithmetic audit.
It does not change a runtime material or calibrate parameters.

## Source and explicit choices

The [accepted manuscript](https://strathprints.strath.ac.uk/68319/1/Zhang_etal_TAFM_2019_Microstructural_damage_mechanics_based_model_for_creep_fracture.pdf)
for DOI 10.1016/j.tafmec.2019.102269 was inspected as rendered pages 7–15 and 18.
Its SHA256 is `fe858d8ed1c4173fcd89220010c1a977a8785ecac446b282ae8f29c7edcdb01b`.
Equations 14–21 and Table 2 supply the numerical parameters. The implemented
scope is constant 130 MPa, 650 C, as-received material, initial damage zero,
initial resistance 3 MPa, ductility 0.2, and initial lath width 0.375 micrometre.
Zero initial damage is an explicit implementation assumption.

The source has unresolved transcription/definition issues:

- Equation 9 includes applied stress in the backstress; equation 15 omits it.
  With dimensionless H-star in Table 2, only equation 9 has stress dimensions.
  The literal equation 15 remains a separately named diagnostic, not a candidate
  material recommended for use.
- The K_L and A unit labels do not match equations 18 and 21. The reference uses
  their numerical micrometre convention, not a claimed SI material card.
- The prose gives K_N=1/3, while the final table gives 0.3. The implementation
  uses the table, explicitly. Initial fitting prose also mentions H-star=0.45;
  the final table supplies 0.5.
- The stated failure construction uses the last curve point with divergent
  creep rate, without specifying a finite numerical rate threshold. We report
  a declared D_N endpoint and study its limit, not an integration failure.

## Integration without a false rupture event

For positive fixed stress, creep strain is monotonic. Using strain `e` as the
independent variable gives these exact reductions (zero initial damage):

```text
sigma0(e) = 6.2 - 3.2 exp(-2400 e / 6.2)
D_L(e) = e / (0.375 + e)
D_P(t) = 1 - (1 + 1.5e-5 t)^(-1/3)
D_N(e) = 1.5 e
dt/de = 1 / creep_rate(e, t)
```

The inverse hyperbolic sine rate is evaluated in logarithmic form, avoiding
overflow as D_N tends to one. Benign underflow of the inverse rate means the
remaining time contribution is negligible; it does not signal solver failure.
This reduction is specific to the declared fixed-stress reference, not an
arbitrary-history structural constitutive integration algorithm.

## Measured results

| Reading | Time to D_N=1−1e−8 |
| --- | ---: |
| Stress-scaled equation 9 | 1188.0432345 h |
| Literal equation 15 diagnostic | approximately 0.04733 h |
| Published Table 1 prediction | 1146 h |

The 3.67% difference from the published prediction remains unresolved. No
coefficient was fitted to erase it. Agreement of numerical methods does not
close this source discrepancy or establish experimental validation.

Tests independently integrate all five original state equations in physical
time with Radau to D_N=0.3 and compare against the reduced equation. Exact
state derivatives, initial conditions, DOP853/Radau, tighter tolerances and
endpoint refinement are checked. At endpoints 0.15/0.24/0.3 the times are
1172.7345/1187.3871/1187.9839 h; endpoints 0.5 through near-one approach
1188.04323 h. Stopping the integration earlier is not used to force agreement.

Run from the source root in fenicsx-env:

```bash
python tools/p92_creep_equation_reference.py
python -m pytest -q tests/test_p92_creep_equation_reference.py tests/test_p92_prior_damage_audit.py
```

The extended implementation passed 80 targeted checks including CI selection.
Prior-fatigue prediction
also needs the dislocation-density ratio, which the paper obtains from an
external cyclic microstructure model rather than listing for every specimen.
Do not silently set it to one or infer it from the validation lifetime. Next
work should resolve the source definitions or bound their sensitivity before
any runtime material integration. High-damage surface-crack failures remain
part of the validation record.

## Conditional prior-state screen and source boundary

The reference accepts explicit `lath_um` and `density_ratio` for conditional
experiments. For these inputs the exact expressions become
`sigma0=6.2-3.2 exp(-2400 e/(6.2 density_ratio))`,
`D_L=e/(lath_um+e)` and `D_N=(1.5+6.418(lath_um-0.375))e`.
They do not reconstruct the prior loading history. At lath width 0.766 micrometre,
the endpoint time changes from 746.42 h at density ratio 1 to 896.69 h at 0.5
and 1039.52 h at 0.1. These are sensitivity points, not measured density values
or validation predictions. No value is chosen by fitting the observed life.

The source's reference 32 is openly available:
[Barrett et al., International Journal of Fatigue 100 (2017), 388–406](https://research.universityofgalway.ie/ws/portalfiles/portal/25749119/IJF_Paper_-_Revised_Manuscript_-_R2_-_v5_for_Aran.pdf),
DOI 10.1016/j.ijfatigue.2017.03.018. Its application/validation is P91 at
400–600 C. Copying its coefficients into P92 at 650 C is a transfer hypothesis,
not a same-condition calibrated input. This is a concrete reason to retain the
density input explicitly instead of quietly implementing a second large model.

A diagnostic trial of the manuscript's other printed numbers gives 1118.77 h
using K_N=1/3 with H-star=0.5, and 433.69 h using K_N=0.3 with H-star=0.45.
Neither has been substituted into the Table 2 baseline or selected by fit.

The conditional Radau test exposed negative *trial* time during error
estimation at the initial point. The analytical state function now has its
smooth mathematical extension inside the integrator, with no clipping; all
accepted times must still be nonnegative and nondecreasing. Public physical
state queries retain nonnegative-time checks. The failed trial was numerical,
not evidence of a negative physical duration or material rupture.
