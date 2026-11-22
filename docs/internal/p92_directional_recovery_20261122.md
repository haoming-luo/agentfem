# Directional recovery: isolated research comparison

Predeclared before fitting. The existing magnitude-powered recall term acts
during both forward and reverse inelastic flow. Test replacing its rate
`abs(v)` by `max(sign(X)*v, 0)` separately for each backstress component.
Here v is that branch's **inelastic flow**, not imposed total strain rate.
This is a scalar directional-recovery hypothesis motivated by critical-state
kinematic hardening, not a claim to reproduce every Ohno--Wang model variant.
Sources: [Ohno--Wang formulation](https://doi.org/10.1016/0749-6419(93)90042-O),
[isotropic-hardening extension](https://www.sciencedirect.com/science/article/pii/S0308016110000232).
The 1993 publisher full text was not accessible. Subsequently the original
[1992 formulation](https://doi.org/10.1299/kikaia.58.488) was downloaded and
equations 5--12 on printed page 489 visually inspected. Equation 10a explicitly
contains `(alpha_bar/r)^m * <eps_p_dot:k> * alpha/r`, with `k=alpha/alpha_bar`.
For `alpha=diag(2X/3,-X/3,-X/3)` and
`eps_p_dot=diag(v,-v/2,-v/2)`, `alpha_bar=abs(X)` and
`eps_p_dot:k=sign(X)*v`. Multiplying the axial equation by 3/2 yields exactly
the directional dynamic term below, with H=h, a=r, chi=m. Thus that term is
the uniaxial reduction of model II; its use in our two-flow, statically
recovering spectrum remains an independently tested extension, not a claim
that the complete P92 model was published or validated by those authors.

A particularly relevant primary comparison is
[Ratcheting and fatigue behavior of Eurofer97 at high temperature, part 2: modeling](https://pure.mpg.de/rest/items/item_3194270_4/component/file_3194951/content).
Its indexed passage comparing equations 9, 14 and 15 explicitly distinguishes
Macaulay directional activation from absolute-value recovery. It also reports
similar simulated responses in that study, with differences restricted to
short reversed-flow intervals. This is a reason for a controlled comparison,
not evidence of a guaranteed gain or transferable P92 parameters. The indexed
passage was accessible; direct full-PDF retrieval was not.

With `d=(H/a)(abs(X)/a)^chi`, replace `Xdot=H*v-(d*abs(v)+r)*X`
by `Xdot=H*v-(d*max(sign(X)*v,0)+r)*X`. No coefficient or state is added.
At X=0 the removed recall contribution is zero. With positive coefficients,
the recall contribution to dissipation remains nonnegative. For |X|=a,
outward flow is opposed exactly as before, while reverse flow points inward;
the invariant interval is retained. Static recovery and recoverable strain
evolution are unchanged. Initial-creep kinetics are not modified in this arm.

Verify sign reflection, zero flow, forward equivalence, reverse distinction,
power balance, continuation and independent integration. Only then use the
existing 22 candidate / 10 retained coordinate identification, early cycles
1--3, norm bound .3, max_nfev 40 and unchanged two-scale derivative audit.
Freeze all parameters for cycles 4--10 and retain all existing response gates.
Failure of the derivative gate stops fitting; it does not justify relaxing
the gate. No production promotion, PR merge or release is authorized.

Pre-fit checks: 406 affected-path tests passed. Replaying all three early protocols
with the new option off reproduced every archived baseline sampled stress
exactly (maximum differences RF/SLR/SLC = 0/0/0 MPa), with early objective
0.002360780027260875. Thus the previous matched baseline remains a valid
response control; the candidate is evaluated with the unchanged fit budget.

At the baseline first SLC positive-hold entry (segment 5), both backstress
components in both branches and both effective stresses are positive. Hence
the directional gate is active there exactly as in the magnitude law. Any
change to that hold must initially be carried by different incoming internal
states from earlier loading, not by a protocol-dependent relaxation switch.
