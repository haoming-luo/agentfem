# Rate-sensitive recovery discriminator

## Question and source boundary

The recorded-stress diagnostic separates excessive early SLC irreversible
flow from a low-stress forward-path artifact. A replacement must change the
history arriving at the post-creep hold without discarding the two recoverable
strains. Multiplying all flow rates on a frozen history is not sufficient.

Two primary publisher records justify testing prior-rate-sensitive kinematic
history, rather than another cumulative-strain multiplier:

- Ho (2008), *Effect of the rate dependence of nonlinear kinematic hardening
  rule on relaxation behavior*, IJSS 45, 821–839,
  <https://doi.org/10.1016/j.ijsolstr.2007.09.003>. The accessible abstract
  distinguishes hardening rules by whether backstress depends on loading and
  relaxation rate. It supports the mechanism question, not P92 parameters.
- Chen, Wang, Kitamura and Feng (2017), *A modified unified viscoplasticity
  model considering time-dependent kinematic hardening for stress relaxation
  with effect of loading history*, IJMS 133, 883–892,
  <https://doi.org/10.1016/j.ijmecsci.2017.09.048>. The accessible publisher
  abstract discusses prior loading-rate effects. Its full equations have not
  been verified here; this is not a reproduction of that model or a P92
  validation dataset.

## Explicit hypothesis, not an attributed literature equation

For each existing irreversible branch, write its accumulated flow rate as
`v = v0 (max(|sigma - sum X| - Y, 0)/120 MPa)^n`. Replace only the dynamic
recovery exponent by

`chi_eff = chi [1 + v0/(v0 + v)]`.

The rate law remains `Xdot = H flow - [(H/a)(|X|/a)^chi_eff v + r] X`.
The fixed interpolation is a predeclared research assumption. Its factor of
two and use of the existing reference flow are **not** fitted or established
P92 physics. No new free coefficient or state is introduced. This changes
kinetic shape, not the nominal invariant saturation bound; it does not claim
to implement the rate-dependent saturation law of either cited paper.
The rate here is the **branch's normalized inelastic rate**, not the imposed
total strain rate. With the existing power law, `v/v0` is a function of
overstress: scaling `v0` alone at the same state does not change `chi_eff`.
This restricted discriminator therefore cannot establish arbitrary loading-rate
sensitivity or independently identify a new physical timescale.

At low flow it retains more sub-saturation backstress than the control;
at high flow it approaches the original rule. At `chi=0` the control is exact.
The competing effect is that stronger inherited backstress can also distort
SLR peaks; all three protocols therefore remain jointly constrained.

At `|X|=a`, the dynamic factor is one for either rule, so the vector field
points into the existing invariant interval. The energy remains
`sigma^2/(2E) + sum X^2/(2H) + sum Hk ak^2/2`; the corresponding dynamic and
static recovery dissipation terms remain nonnegative. There is no derivative
of `chi_eff` in this stored energy because it has no exponent dependence.
Both Kelvin modes and the twelve-state accepted history are retained.

## Prospective comparison contract

- Seed: `2026-11-22-recovery-shape-control-frozen.json`.
- Actual measured RF, SLR and SLC waveforms; cycles 1–3 only for identification.
- Control and candidate use the same 22 candidate coordinates, ten retained
  directions, two derivative step sizes, log-coordinate radius 0.3 and
  maximum 40 optimizer evaluations. Derivative/subspace gates are unchanged.
- Freeze each successful fit; independently replay ten continuous cycles
  using Radau, BDF and refined waveform knots. Do not reset history at a hold
  or cycle boundary. Report every original response gate, not just total cost.
- Later cycles were previously inspected and remain development checks, not
  blind validation. Recording-resolution failures remain unresolved, not
  silently accepted. No engineering response or lifetime qualification.
- Keep the mechanism under research tools and PR #197; no merge or release.

Local mechanism and neighboring regressions: 50 tests passed; the final full P92
and CI-scope regression has 1,167 passing tests (65.34 seconds). These establish implementation
checks, not agreement with measured P92 response.

The fixed-card ten-cycle replay is retained as
`evidence/creep/2026-11-23-rate-recovery-fixed.json`. All three independent
integration/refinement checks pass; all three original response screens fail.
For the first SLC post-creep hold, the predicted final sampled stress moves
from 162.4297 to 162.7723 MPa against 174.8800 MPa observed. Its first-sample
residual worsens from -0.4845 to -0.8261 MPa. The early stress secant changes
from -5.1834 to -5.1244 MPa/s against -3.7937 observed; the late secant moves
from -0.08231 to -0.08455 MPa/s against -0.18193 observed. These small shifts
do not close the conflict. SLR cycle-10 peak error remains +8.075%.

## Completed matched identification and freeze

`2026-11-23-rate-recovery-paired.json` checks identical source/seed and driver
identities, early-only training partition, retained dimension, coordinate
radius and evaluation ceiling before comparing both frozen results.

| Quantity | Original control | Rate-shape replacement |
| --- | ---: | ---: |
| Early joint cost | 0.002239828765 | 0.002231890070 |
| Optimizer evaluations (ceiling 40) | 4 | 5 |
| Total residual evaluations, including derivatives | 174 | 195 |
| SLR cycle-10 peak error | +7.9474% | +7.8344% |
| First SLC post-creep hold first-sample residual | +1.1160 MPa | +1.3347 MPa |
| Same hold final sampled stress (observed 174.88 MPa) | 163.0703 MPa | 162.9415 MPa |
| Early stress secant (observed -3.7937 MPa/s) | -5.3703 MPa/s | -5.3864 MPa/s |
| Late stress secant (observed -0.18193 MPa/s) | -0.08068 MPa/s | -0.08217 MPa/s |

Both derivative/subspace checks pass; both optimizers satisfy their original
gradient termination criterion. Same budget means identical allowed effort,
not equal realized evaluations after early convergence. The replacement lowers
joint cost by only 0.3544%. All six frozen numerical comparisons pass, while
all six original response screens fail. SLC entry, endpoint and early secant
worsen despite the slightly improved SLR peak and late SLC secant. All ten
unresolved late SLC creep secants remain unresolved in each arm.

**Decision:** do not promote this bounded replacement, broaden its parameter
range or restart its fit on inspected later cycles. The experiment does not
exclude every rate-dependent hardening model, but does reject claiming this
particular substitution solves the retained history conflict. Recoverability,
state continuity, source identity and original acceptance standards remain
intact. Further work needs a separately justified irreversible-history law
or discriminating observations, not another aggregate-cost improvement claim.
