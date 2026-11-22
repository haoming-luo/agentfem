# Check the actual replacement space, not only the nominal parameter count

The matched experiments replace weak coordinates and retain ten right-singular
directions of the early-response Jacobian. Keeping a named replacement among
22 candidates does not guarantee that its coordinate directions remain in the
retained ten-dimensional subspace. The previous projection-Gram audit detects
some loss, but passing its threshold is not evidence of broad exploration.

For an orthonormal basis B, coordinates z=B*u and box |u_k|<=h, the exact
reachable half-width of coordinate i is h*sum_k(abs(B_ik)). For a unit
combination v of the replacement coordinates, its half-width is
h*sum_k(abs((v.T*B_replacement)_k)). These are deterministic support functions
of the same declared box. They are not uncertainty intervals or new bounds.

The new read-only audit verifies orthonormality, reconstructed fitted
coordinates and the original radius before reporting these quantities. Tests
compare them with every vertex of a synthetic three-dimensional box and
check that a fully excluded coordinate is reported as zero. No new solve,
parameter, residual weighting or acceptance threshold is introduced.

## Existing-fit evidence

| Replacement | Smallest projection-Gram eigenvalue | Weakest combination reachable half-width |
| --- | ---: | ---: |
| Shared static history | 0.000158994 | 0.002807615 |
| Nonlinear recoverable kinetics | 0.021993736 | 0.034074077 |

Both use the original log-coordinate radius 0.3 and ten retained directions.
In particular the shared-history fit explored a narrow combination of its new
coordinates. Its failure remains a genuine failure of that frozen card; it
does **not** prove that every card in the history-mechanism family fails.
Conversely, this geometry does not demonstrate that changing the subspace
would solve SLC or justify increasing the parameter range.

Before a further paired identification, audit the completed transient-strength
fit as well. If a mechanism-directed subspace is proposed, declare its
construction, unchanged dimension/radius/budget and an appropriate control
before solving. Do not silently replace the existing experiment or label
previously inspected later cycles as blind validation.

Evidence: `2026-11-22-static-memory-fit-space.json` and
`2026-11-22-recoverable-kinetics-fit-space.json` under `evidence/creep/`.
Each binds its original fit by SHA-256. This audit is not material admission.

## Prospective replacement-directed space

The diagnostic also constructs a ten-dimensional space containing the two
replacement coordinate axes exactly, plus the eight most sensitive directions
in their orthogonal complement. This is a proposed experimental geometry, not
an automatic fit or a reason to enlarge the original radius. Two finite-
difference scales must still agree, and the restricted sensitivity must pass
the original minimum singular value and condition-number screens.

Using the archived derivatives, the original A static-recovery axes fail this
screen (condition approximately 764,000, smallest singular value 2.01e-6).
Shared-history replacements pass (condition approximately 573), as do nonlinear
recoverable-kinetics replacements (approximately 166). Their two-scale projector
distances are respectively 0.00233 and 0.00192. These results establish local
numerical sensitivity only. They do not establish a better fit, unique material
parameters, or successful later-cycle prediction. In particular, forcing the
weak original axes into a control would not create a trustworthy matched control.

Sensitivity and parameter stability are established concerns in constitutive
identification: [Mahnken and Stein (1996)](https://doi.org/10.1016/S0749-6419(95)00016-X)
study derivative-based identification and Hessian stability;
[Shutov and Kaygorodtseva (2019)](https://arxiv.org/abs/1902.10066)
distinguish fit quality from sensitivity to measurement errors. Neither source
prescribes this particular retained-box construction or validates its use for P92.

If the completed transient-strength result still fails, a limited paired
geometry experiment is justified for the existing shared-history hypothesis:
compare its ordinary top-ten space with the replacement-directed space, using
the same seed, observations, 22 candidate coordinates, ten optimization
coordinates, radius 0.3, box half-width 0.3/sqrt(10), maximum 40 optimizer calls
and tolerances. Retain the stronger original material fit as a separate
scientific control. Do not force its unresolved A static axes into a new
control, or call the two different retained sets identical experiments.
No material equation or fixed coefficient is added by this geometry comparison.
Freeze each converged outcome for the unchanged ten-cycle response screen.
Stop this comparison if the two-scale sensitivity checks fail; do not widen
bounds or keep restarting from the latest optimum.

## Completed geometry comparison

The replacement-directed fit converged in seven optimizer evaluations
(237 residual evaluations including derivative audits), with cost
0.00234455847. This is slightly below the matched original material control
(0.00236078003), but worse than the same shared-history law in its ordinary
top-ten space (0.00231955771). Giving the two replacement axes full retained
projection did not unlock a better joint prediction.

All three ten-cycle paths pass independent integration and waveform refinement;
all three experimental response screens fail. SLR cycle-10 peak error remains
8.70%. First SLC post-creep late stress bias is -14.087 MPa versus the original
-14.152 MPa. Its diagnostic shape error improves slightly while entry bias
worsens from -0.302 to -0.919 MPa. These are not acceptance. Do not expand the
box or restart the fit. The four `2026-11-22-static-memory-replacement-*`
archives preserve the card, frozen screens, exact residual partition and
explored-space audit; full trajectories remain local.
