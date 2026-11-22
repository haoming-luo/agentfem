# Separate stress-rate curvature from history evolution

The original post-creep hold relaxes too quickly early and too slowly late.
Before adding history, test whether the power-law curvature of the dominant
B flow itself contributes to this conflict. Preserve the yield threshold,
both backstress branches, original directional rule and both recoverable
strains. This differs from the rejected zero-threshold test and the earlier
single-flow sinh model: neither has this matched two-flow recovered state.

The [Hosseini et al. (2020) introduction](https://doi.org/10.1016/j.ijmecsci.2020.105546)
discusses power, hyperbolic-sine and exponential stress dependencies, then
uses a two-term power law for 10%Cr steel. This supports testing stress-rate
curvature separately; the following matched function is our discriminator,
not their published law or a microscopic calibration for P92.

For positive overstress ratio u=max(abs(sigma-X)-k,0)/120, replace only
B's q*u^n by q*sinh(x*u)/sinh(x), where x/tanh(x)=n. At n=1 use the
linear limit. This preserves rate q and derivative n*q/120 at u=1,
without another fitted coefficient or anchor. It retains the threshold and
positive dissipation. At small positive overstress it is linear rather than
u^n. That may correct late curvature, or may worsen creep accumulation;
the test must expose either outcome.

Predeclare only the original and B-replacement arms, from the same seed and
early three protocols. Do not scan matching stresses, combine with mobility
feedback, or move thresholds. Verify rate and derivative matching, threshold
behavior, energy, recoverability, signed symmetry, independent integration
and rollback. Admit a fit only after a completed, jointly improved fixed-card
screen with compatible creep/hold entry. Any fit retains the original
22/10 coordinate budget, radius 0.3, maximum 40 optimizer calls and all
frozen response gates. Previously inspected later cycles are not blind data.

## Outcome: reject the matched curvature replacement

The original reproduces cost 0.00257608010. The replacement completes RF
and SLR but fails the first SLC post-creep continuation: strain 0.0021916681
already exceeds the next prescribed 0.002 target, with inherited stress
136.695354 MPa. No cost is assigned to the incomplete joint trajectory.
Do not reverse the ramp, reset the hold, fit this arm or scan another anchor.
The local rate/tangent match is insufficient to preserve the continuous
mixed-control trajectory. This is a negative result for this specific
mapping, not a refutation of activated flow laws in general.

An analytic review further limits this mapping: the logarithmic derivative
of sinh(x*u) is x*u*coth(x*u), strictly increasing with u. Matching it to n
at u=1 makes the log ratio to u^n attain its minimum zero there. Thus this
replacement never lowers the flow at a fixed positive overstress. Tests
verify that inequality across the relevant exponents and stress ratios.
It can redistribute curvature but cannot independently slow the early rate
at unchanged internal state. This is not a monotonicity theorem for the full
coupled trajectory; it is another reason not to keep tuning this rejected arm.
