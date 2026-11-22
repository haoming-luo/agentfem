# Separate plastic consistency from persistent overstress flow

## Basis and limits

ANL-ART-167 (2019), pages 4--7 and proposed guidance p.20,
<https://publications.anl.gov/anlpubs/2020/02/158587.pdf>, distinguishes
non-unified rate-independent plasticity plus creep from unified viscoplastic
models. It requires interaction and loading-history checks; it is a Grade 91
report and proposed guidance, not P92 certification. This supports testing
the distinction, not importing its coefficients or claiming a stronger model.

The present B branch contributes excessive post-creep hold flow in the
same-stress diagnostic. Its static-recovery changes redistribute error but
have not resolved early and late relaxation together. A consistency-controlled
B branch is a distinct limit, not another recovery multiplier. Keep branch A,
both recoverable strains, all existing quadratic energies and continuous state.

## Law and parameter accounting

Replace B's power law by f=abs(sigma-X_B1-X_B2)-k_B <=0,
lambda>=0, lambda*f=0. Its signed flow is sign(s_B)*lambda. Retain B's
existing nonlinear dynamic and linear static backstress recovery. A still
uses its original overstress law; the two Kelvin rates remain unchanged.
In strain control the consistency numerator is

    n = sign(s_B) * (E*(edot-v_A-sum(adot)) + sum(r_Bi*X_Bi))

and the denominator is

    d = E + sum(H_Bi - sign(s_B)*c_Bi*X_Bi),
    c_Bi = H_Bi/a_Bi * abs(X_Bi/a_Bi)^chi_Bi.

On the active surface lambda=max(n,0)/d; in prescribed constant-stress
control omit both E contributions. A nonpositive consistency denominator
is a model-domain failure, not permission to regularize with an extra
viscosity. Outside the active surface lambda=0. Locate yield entry and release
as integration events; do not clip accepted stress or reset inherited state.

Flow dissipation is |s_A|*|v_A| + k_B*lambda plus nonnegative backstress
and Kelvin recovery contributions. Check power balance, yield admissibility,
complementarity, reflection, analytic hardening limits, continuation and
independent integration before data.

B reference flow and exponent no longer enter the law. The fixed seed maps
k_B to k_B+120 MPa: this preserves the old virgin flow stress at its declared
reference flow q_B, using the existing 120 MPa normalization. It does not
preserve the full rate-sensitive curve. The resulting material has two fewer
independent constants and no added states. Old evolved checkpoints are
incompatible with the changed yield domain. This mapping is prospective,
not inferred from the observed BAM peaks.

Run only the original and this mapped card on cycles 1--3 of all protocols,
with the same controls and gates. If the mapped path fails continuation or
worsens joint prediction without compatible hold/creep improvement, stop
without fitting or shifting the mapping. Otherwise separately declare one
matched-budget early identification and frozen later-cycle verification.
No runtime promotion, merge, release or life qualification follows from
implementing the reference.

## Mapped-card outcome

All three early paths complete, but cost rises from 0.0023607800 to
0.0193906372. The SLC first post-creep hold early slope improves from
-5.0814 to -3.7343 MPa/s (observed -3.7937), while its late slope worsens
from -0.09027 to -0.04051 MPa/s (observed -0.18193). The first sampled
stress error grows from -0.302 to -15.162 MPa. Its endpoint error alone
improves, from -14.152 to +6.088 MPa, illustrating why endpoint-only
selection would be misleading. RF and SLR hold predictions also worsen.

The prospective stopping rule applies: no identification or altered yield
mapping. Seven focused tests pass, including an analytic Armstrong--Frederick
monotonic limit, time-scaling in the rate-independent limit, reverse loading,
independent integration and split continuation. These verify the reference,
not agreement with BAM. The result rejects this mapped card, not every
non-unified plasticity/creep formulation. All experimental screens remain
failed. Evidence: `2026-11-22-inviscid-branch-probe.json` under `evidence/creep/`.
