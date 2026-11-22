# Isolated primary-creep discriminator

Predeclared research hypothesis, not a published P92 card or a production law.
Keep magnitude dynamic recovery and the fast recoverable branch unchanged.
Replace only the slow Kelvin branch by irreversible, strain-hardening primary
flow. Reuse its two positive coefficients H and tau; add no fitted coefficient.
The candidate has its own model/state identity: slot 11 is accumulated primary
strain p, not recoverable signed strain. Never reinterpret an old checkpoint.

For stress sigma in MPa, set s0=120 MPa (the existing flow normalization),
v=sigma/(H*tau)*exp(-H*p/s0), pdot=abs(v). The primary element stores no energy;
its dissipation is sigma*v>=0. At constant stress from p=0,
p=(s0/H)*log(1+abs(sigma)*t/(s0*tau)). Thus initial rate is finite,
decelerates continuously, and gives logarithmic primary creep. Reversal
changes flow sign but does not erase accumulated history; zero stress gives
zero primary flow. These are explicit hypotheses, not measured P92 facts.
H and tau now control irreversible kinetics, not recoverable
compliance. This replacement can therefore fail unloading/hold tests.

Motivation: Khayatzadeh et al., Engineering Fracture Mechanics 175 (2017),
DOI 10.1016/j.engfracmech.2017.02.008, reports that primary creep matters to
P92 relaxation and compares strain/time hardening. Its 650 C data and
monotonic law do not establish this cyclic 620 C extension. The regularized
law above is our bounded discriminator, not a transcription of their equation.
Source: https://strathprints.strath.ac.uk/60381/

More directly, Alam, Yamamoto and Odette (2025), Metals 15, 354,
https://doi.org/10.3390/met15040354, Eq. (1), describes an initial excess rate
decaying exponentially with strain, above a minimum creep rate. Here the
existing backbone retains the other flows, and only the replaced branch uses
this excess-rate form. Our stress-proportional initial rate, s0/H strain scale
and absolute-history extension to reversal remain explicit modeling choices.
Their 375--550 C, 10--80000 h database is NOT calibration evidence for our
620 C, seconds-to-minutes mixed-control cohort.

An exploratory rational-decay version was stopped during its derivative audit
(before optimization) after this more direct source was located. Its seed cost
was 0.005264396768738119. It is not a completed matched comparison, and no
acceptance conclusion follows from it. The exponential version and its
analytic checks below supersede that provisional choice before identification.

Before fitting: constant-stress analytic check, stress reflection, zero-load
memory, nonnegative dissipation, integrated work balance, split continuation,
independent BDF integration, strict state identity. Then use the existing
early cycles 1--3 objective, 22 candidate/10 retained coordinates, norm .3,
40 optimizer-evaluation ceiling and unchanged two-scale derivative gate.
Keep all three protocols, actual ramp shape and inherited SLC force control.
Freeze for cycles 4--10, which are inspected development data, not blind data.
No altered response tolerance, added parameter, directional-law stacking,
merge or release. Report a failed derivative gate rather than fit through it.
