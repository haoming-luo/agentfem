# Subthreshold flow discriminator, preserving recoverable deformation

## Specific diagnostic

In the archived matched baseline, first SLC creep changes p_A from
0.0004101415569 to 0.001003374088, but p_B only from 0.0003104311871 to
0.0003104622790. The held stress is 123.087 MPa and B's threshold is
125.015 MPa. B is almost inactive during the long hold and its backstresses
relax toward zero. After reloading, B contributes strongly to the excessive
positive-hold flow. This is a model-state diagnosis, not an identification of
the specimen's microscopic deformation mechanisms.

Zhang et al. (2022), DOI 10.1016/j.ijmecsci.2022.107369, explicitly studies
P92 hybrid stress/strain control and motivates a Walker formulation without
a yield surface. The accessible publisher abstract/introduction does not
provide its full parameter card or equations. Do not claim to implement that
published model. Test a narrower hypothesis with the existing verified scalar
power-law branch: replace B's positive threshold by zero, preserving the two
recoverable branches and all backstress equations. Earlier zero-threshold
screening predates the current recoverable spectrum and is not a matched
comparison for this state layout.

## Rate-and-slope matching, not a free coefficient

At a fixed effective stress S>k, the old rate is q*((S-k)/120)^n. Define

    m = n*S/(S-k)
    q_new = q*((S-k)/120)^n / (S/120)^m
    v_new = q_new*abs(effective/120)^m*sign(effective).

Both the rate and its stress derivative agree at S. This avoids attributing
an arbitrary change of the high-stress rate scale to subthreshold activation.
There are fewer independent branch coefficients, not more. S is a declared
screening anchor, not fitted. Predeclare control and S=200,250,300 MPa, within
the early operating range and above the original threshold. Do not extend the
list after inspection. Do not combine this replacement with any new hardening
or nonlinear recoverable law. The original magnitude-recovery spectrum is
the common seed; only cycles 1--3 enter the fixed comparison.

Verify rate/slope equality, positive monotone signed flow, unchanged input
card, energy accounting and independent integration before evaluating data.
If the SLC continuation exceeds its next strain target, retain that failure;
never reverse the next ramp or reset the held force to make it complete.
Any subsequent identification needs a separately declared same-budget plan
and all unchanged response gates. This is not yet a selected candidate.

## Outcome: reject this replacement

The control cost is .002576080096967705; S=200 MPa gives
.03310025547836086. Its SLC first three peak errors are approximately
-114.72, -109.73 and -106.97 MPa. S=250 and 300 fail the original first
SLC post-creep continuation: predicted strains .00357057552 and .00573395214
already exceed the next target .002. Held stresses are 142.11625 and
151.293585 MPa, respectively. Retain these physical-path rejections.
Do not fit this candidate, change the ramp direction, expand the anchor list,
or interpret this local negative test as refutation of the published Walker
model. Matching one high-stress rate and slope does not preserve a coupled
history trajectory.
