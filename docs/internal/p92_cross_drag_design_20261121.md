# Prospective bounded kinetic interaction discriminator

The equal-stress audit localizes a missing history effect: the frozen model's
SLC positive-hold flow is too fast after creep, even where its SLR flow at
similar stress is close. This supports testing a history-dependent resistance,
not applying a correction selected by protocol name or cycle number.

The [two-mechanism thermodynamic analysis](https://journals.ub.ovgu.de/index.php/techmech/article/download/766/743)
allows interactions and positive evolving drag. The
[martensitic-steel prior-loading study](https://doi.org/10.1016/j.ijmecsci.2020.105546)
describes evolving drag/backstress with hardening and recovery, but its
10%Cr/600 C application does not supply BAM parameters. Only its abstract is
used here. Neither source supplies the particular equation below. It is an
explicitly new scalar research hypothesis, not a published P92 model.

## Minimal kinetic state

Add a dimensionless resistance history h in [0,h_max]. Branch B's reference
flow becomes `q_B*exp(-h)`, equivalently a positive multiplicative drag change.
Use `hdot = c_A*abs(flow_A)*(h_max-h)
             - c_B*abs(flow_B)*h - r*h`.
The same law and coefficients operate during every loading segment. No
controller label, observed stress, cycle count or elapsed-test lookup enters it.
At h=0 its derivative is nonnegative; at h=h_max it is nonpositive. Zero
h_max with zero initial h strictly recovers the original spectrum reference.

h is declared a kinetic resistance variable, not an additional stored-energy
variable. The existing free energy is unchanged and the modified monotone
branch flow still makes effective_stress*flow nonnegative. Recovery and Kelvin
dissipation retain their existing definitions. This proves the stated scalar
power accounting, not a microscopic dislocation interpretation. Adding an
energy conjugate to h would require a different derivation and model identity.

## Verification before any BAM run

Check zero-interaction identity, signed symmetry, instantaneous power balance,
positive bounded h, no mutation on failure, independent Radau/BDF integration,
split continuation and exact decay when both flow branches are inactive.
Use a distinct 13-state identity; never reinterpret a 12-state restart.

Then run only a fixed-card early discriminator: h_max = 0, 0.5, 1, 2;
c_A=c_B=1000 (inverse strain), r=1/1800 s^-1. These are declared screening
scales (0.001 strain and the long-hold duration), not literature parameters
or a fit. Keep the original best spectrum card, actual ramp shapes, all
three early protocols and unchanged response screens. Record every failure.
Do not extend these bounds after seeing the results. Only compatible
improvements in inherited stress, post-creep hold and other protocols justify
a subsequent separately declared identification experiment.

## Fixed-card outcome

All four early continuous-path probes completed. Costs for h_max = 0, 0.5,
1 and 2 were respectively 0.0025760801, 0.0028915877, 0.0036337037 and
0.0064778199. The zero-interaction case reproduces the baseline. The added
history resistance worsens the joint objective throughout this declared screen;
it does not justify a fit or a production capability. No bounds are expanded.
Evidence: `evidence/creep/2026-11-21-cross-drag-probe.json`.
