# Replace a flow mechanism instead of adding one

The bandwidth screen leaves both irreversible flows intact. Its failure does
not test whether the low-stress flow itself should be recoverable on the time
scale of these experiments. Benaarbia, Rouse and Sun (2018),
https://doi.org/10.1016/j.ijplas.2018.03.015, motivate a decomposition into
viscoplasticity and three viscoelastic components for P91 at 600 C. This is a
mechanism precedent, not a BAM P92 parameter source or a reproduction of their
full model (which also treats cyclic softening).

Prospective discriminator: remove irreversible branch A, retain branch B with
its original two backstresses and magnitude-based dynamic recovery, and retain
the original two Kelvin modes. Replace A with one linear Kelvin mode. Its
positive quadratic energy and dashpot dissipation are explicit. There is one
irreversible accumulated strain and three signed recoverable strains; use a
new ten-state identity rather than interpreting an old restart differently.

The initial replacement is fixed before examining its BAM response:

    H_new = H_A1 + H_A2
    sigma_ref = 120 MPa  (the existing flow normalization)
    v_ref = q_A * ((sigma_ref-k_A)/120 MPa)**n_A
    tau_new = sigma_ref/(H_new*v_ref).

This preserves the A branch's initial hardening modulus sum and its zero-state
flow at the reference stress. It does NOT preserve its tangent mobility,
nonlinear unloading or long-time permanent creep. A seed with k_A>=120 MPa is
ineligible for this mapping; do not invent a new reference stress. Remove A's
coefficients from the resulting material record. The new material has nine
candidate branch-B coefficients and six Kelvin coefficients, versus the
original 22 candidates. E and the original ratcheting exponents remain fixed.
There are fewer independent material constants, not merely fewer fit directions.

Verify signed constant-stress creep and unloaded recovery below B's threshold,
the secant-flow mapping, power balance, reflection, invariant domains,
independent integration and split continuation before data evaluation. The
short-time response experiment must not be sold as a low-stress steady-creep
or engineering life model: below B's threshold all strain is recoverable and
eventually saturates. That limitation must remain visible even if the BAM
short-time response happens to pass.

Run the one mapped card on all early protocols with the unchanged measured
controls. No extra centers or hand-picked ranges. If it completes the controls
and numerical verification, a single bounded early-only fit may test the
replacement (not the uncalibrated mapping): at most ten retained directions
from fifteen candidates, original norm .3, max 40 optimizer evaluations,
unchanged residuals/tolerances and two-scale sensitivity checks. Freeze any
converged result for the original ten-cycle response gates. A failed physical
continuation or unresolved sensitivity stops this route without wider bounds.
Compare against the stronger archived original fit, not only the mapped seed.
No production material, merge or release follows from a research fit.

## Mapped-card outcome: continuation fails, no fit

The derived new modulus is 206734.05 MPa and time 137.3911 s. RF and SLR
complete, but SLC's first creep hold predicts strain .00222325197 before a
declared positive ramp to .002, at an inherited stress 148.8583 MPa. This
violates the continuation direction. The driver records the failure and a
null joint cost; it does not reset strain, prescribe the observed creep
endpoint or reverse the next ramp. As predeclared, do not identify this mapped
candidate or adjust the mapping stress after seeing the result.

This rejects the chosen secant mapping, not the published three-component
family or the possibility of a separately justified initialization. The new
reference kernel passes its analytic recovery and energy tests; those tests
do not establish agreement with BAM. Evidence:
`evidence/creep/2026-11-22-recovery-replacement-probe.json`.
