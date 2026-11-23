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

Pre-fit checks: 410 affected-path tests passed. Replaying all three early protocols
with the new option off reproduced every archived baseline sampled stress
exactly (maximum differences RF/SLR/SLC = 0/0/0 MPa), with early objective
0.002360780027260875. Thus the previous matched baseline remains a valid
response control; the candidate is evaluated with the unchanged fit budget.

At the baseline first SLC positive-hold entry (segment 5), both backstress
components in both branches and both effective stresses are positive. Hence
the directional gate is active there exactly as in the magnitude law. Any
change to that hold must initially be carried by different incoming internal
states from earlier loading, not by a protocol-dependent relaxation switch.

## Matched identification

The directional fit converged by the unchanged gradient criterion after
12 optimizer evaluations and 322 actual residual evaluations (including
derivative audits and finite differences). The baseline used 8 and 238,
respectively; these are identical budget ceilings, not identical consumption.
The directional cost is 0.002367848048443141, about 0.30% higher than the
baseline 0.0023607800272608744. Three retained coordinates are at bounds;
no bounds were enlarged. Relative two-scale Jacobian difference is
0.0007194364 and retained-subspace distance is 0.002978882, both within the
unchanged gates. This is a local fit, not proof of global optimality.

The material parameters, model identity and source hashes of the temporary
seed exactly match `evidence/creep/2026-11-21-spectrum-frozen.json`; that
archived seed can be supplied to `tools.p92_spectrum_recovery_joint` with
`--directional` and the checksum-verified source directory. The outer seed-file
hash differs because the archived file omits raw trajectories; the fitter uses
only those three verified fields. Final fitted evidence is byte-identical to
the local run output, and can directly drive `tools.p92_recoverable_frozen`.

## Frozen outcome: no promotion

All three ten-cycle paths pass independent BDF and waveform-refinement
checks. Maximum independent stress-state difference is below 8e-7 MPa;
mechanical work/energy/dissipation balance residual magnitudes are below
6.1e-12 MPa. Nevertheless **all three experimental response screens fail**.

| Diagnostic | Magnitude control | Directional candidate |
| --- | ---: | ---: |
| Failed checks RF / SLR / SLC | 21 / 40 / 74 | 20 / 41 / 74 |
| SLR cycle-10 peak relative error | 8.931% | 9.043% |
| First SLC post-creep hold anchor error, MPa | -0.302 | +0.305 |
| Same hold early anchored-shape RMS, MPa | 5.756 | 5.913 |
| Same hold late anchored-shape magnitude, MPa | 13.850 | 14.491 |

Unshifted early SLC hold RMS slightly improves (5.903 to 5.774 MPa), while
its anchored shape worsens. The changed entry offset masks a kinetic error;
it must not be presented as improved relaxation kinetics. Both cards retain
135 failed checks overall. Anchored diagnostics do not replace response gates.

This bounded local experiment does not disprove directional recovery in
general. It shows no advantage sufficient to promote it for this cohort.
Keep magnitude recovery as the control for the next **separate** primary-creep
study; do not stack the directional law and a new creep law and then attribute
any gain to either one. That next study is not implemented or validated here.
No parameter-count increase, tolerance change, merge or release occurred.

Evidence: `2026-11-22-directional-matched.json`,
`2026-11-22-directional-frozen.json`, and
`2026-11-22-directional-residual-conflicts.json` under `evidence/creep/`.
