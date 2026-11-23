# Audit the inherited elastic modulus before another mechanism fit

The spectrum card inherited E=155176.471102 MPa from an earlier identification
with a different inelastic model. Local reversal slopes are not independent
elastic measurements: delayed recovery and plasticity also contribute.
[Juergens et al. (2019), Table 2](https://www.mdpi.com/2075-4701/9/1/99)
reports 162 GPa at 620 C using sonic resonance (the authors cite ASTM
E1875:2013). This is an independent literature sensitivity point, not a claim
of measured modulus for these particular BAM specimens or of certification.

Compare the inherited value and exactly 162000 MPa. Keep every flow,
backstress, Kelvin coefficient, control waveform and initial-state convention
unchanged. No refit, extra coefficient, revised gate or inferred specimen
correction. Replay RF, SLR and SLC cycles 1--3 from their initial states;
do not substitute new modulus into an old evolved state. Report all protocol
errors, creep accumulation and post-creep hold, not just total cost.

The same uncertainty affects observed strain partitioning. Holding the
recorded path and Kelvin coefficients fixed gives the exact change in the
inferred remainder: delta_sigma*(1/E_old - 1/E_new). Check this identity
without presenting the inferred remainder as an observed plastic strain.
Do not use the literature value as a new fitted range or select it by a
single favorable observable. If the discrepancy survives, retain that
evidence rather than launch a modulus-only fit.

## Outcome

Both cards complete all three early protocols. The 162 GPa point raises the
joint cost from 0.00236078003 to 0.00385757470. The first SLC post-creep
endpoint changes from 160.728 to 161.157 MPa versus 174.880 MPa observed;
its late rate remains -0.091654 versus -0.181928 MPa/s. The first SLR
post-dwell endpoint also moves further above observation (158.407 versus
148.340 MPa). All response screens still fail. This sensitivity does not
justify replacing the card or fitting E alone, nor does it prove E is known.
Archive: `evidence/creep/2026-11-22-modulus-probe.json`.

Further primary-source review: Sonntag et al., DOI
10.1016/j.ijfatigue.2022.107381, page 8, explicitly uses **151 GPa** for
the mechanism-map shear-modulus conversion. Do not present 162 GPa from
the earlier sonic-resonance study as the uniquely correct value for this
dataset. The two literature values describe different reported uses; neither
establishes the uncertainty distribution of the specimen's modulus here.
The completed 162 GPa comparison remains a sensitivity experiment, not a
correction of the experimental source. No post-hoc parameter interval is
inferred from these two numbers.

Reading section 3.1 (page 5) resolves its role more precisely: the authors also
use 151 GPa for their elastic/creep partition throughout life, referencing
their 2021 study [27]. Page 3 states the same material batch as the earlier
studies. This supports a **separate source-modulus control**, not a random
search point: replay the original card and 151000 MPa with all other values
fixed. Keep the completed 162 GPa comparison, source attribution and raw
hashes unchanged. This tests consistency with the paper's analysis convention;
it does not convert that number into an individual-specimen measurement or
retrospectively redefine any gate.

The 151 GPa source-modulus comparison also completes, with cost
0.00300115195 versus 0.00236078003. SLC's first post-creep endpoint becomes
160.089 MPa and its late secant -0.088259 MPa/s. SLR's endpoint is
157.411 MPa. Thus neither documented fixed-modulus substitution resolves
the hold conflict with the other coefficients unchanged. This is not a
joint re-identification or proof of modulus independence. Evidence is kept
separately in `2026-11-22-source-modulus-probe.json`.
