# P92 loading discriminator and qualification gate

## Frozen comparison

`tools/p92_loading_discriminator.py` compares recorded and nominal RF loading
without changing fitted parameters or resetting hidden history. It checks the
source identity against the frozen calibration and rejects a failed fit.

Reproduce with the checksum-verified `Data_RF.lis` from
[BAM's public dataset](https://doi.org/10.5281/zenodo.7198219):

```sh
python -m tools.p92_loading_discriminator Data_RF.lis \
  evidence/creep/2026-11-16-bam-positive-strength.json --cycles 10
```

Source SHA256: `128e6de1854f4633e26eced6c4ea221687cde2f232db275487ef0d1201767e0a`.
Calibration canonical JSON SHA256:
`e70095c61a15f76466ff7365694c1004d67ff3ef3424ae9fe0dc0847925631a9`.

Observed calculation (MPa, rounded for this summary):

| Cycle | Observed peak | Recorded-input Radau | Nominal-input Radau | BE2/Radau maximum stress difference on recorded input |
|---|---:|---:|---:|---:|
| 1 | 257.37 | 261.5771 | 263.4233 | 0.8432 |
| 3 | 290.28 | 287.3083 | 285.9032 | 0.9216 |
| 10 | 276.94 | 269.3891 | 269.0396 | 1.0369 |

At cycle 10, changing the loading description changes the peak by only
0.3495 MPa; the recorded-input response still misses the observed peak by
7.5509 MPa. This is an early-cycle discriminator, **not** an explanation of
the cycle-1000 error. Extrema are sampled, not certified continuous extrema.
The source span contains 2,171 unrecorded cycles. Missing measured paths must
not be invented or supplied with observation-driven hidden-state resets.

Repeating with `--subdivisions 8` reduces the cycle-10 BE/Radau maximum
difference from 1.036899 to 0.265438 MPa (cycle 1: 0.843238 to 0.215845 MPa).
The Radau prediction and its 7.550873 MPa cycle-10 peak error are unchanged.
This supports temporal convergence of the diagnostic, not qualification of the
material law. It does not establish a bound for all later cycles.

The associated 42 targeted tests passed with normal exit using a process-local
`FI_PROVIDER=tcp`. The default local provider passed all assertions but hung in
PETSc/MPI finalization (sampled stack: `MPIDI_OFI_mpi_finalize_hook`); that run
was terminated after diagnosis and is not recorded as a clean acceptance run.

## What can actually be qualified

Separate three claims: verified numerical implementation, validated material
response for a stated population/loading domain, and a qualified life prediction.
The published five-state response law has no damage-to-failure equation. A
successful 2,547-cycle integration therefore cannot validate a failure life.
The 600-degree parameter set applied to the 620-degree BAM cohort is a transfer
experiment, not a validated material card.

The next bounded scientific experiment must:

1. Freeze a candidate response model and its temperature/cohort assumptions;
   assess parameter identifiability using training data before adding parameters.
2. Refine the measured-input time integration separately from fitting. Numerical
   error must be materially smaller than the prediction errors being interpreted.
3. Declare already inspected RF/SLR/SLC records as development data. Test any new
   candidate on an independent specimen/protocol not used to select it; no fresh
   holdout claim from reshuffling inspected observations.
4. For life, acquire failure definitions, censoring, specimen/cohort identifiers,
   and repeated tests. Distinct protocols are not replicate observations of a
   single life distribution. Keep confidence and prediction intervals distinct.
5. Establish intended-use tolerances and reliability targets before evaluating
   the independent results. Report bias, scatter, interval coverage, and limits
   of application; no universal five-percent or factor-two rule is assumed.

[ASME V&V 10](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics/2019)
provides the solid-mechanics credibility framework.
[ASTM E2714](https://store.astm.org/standards/e2714) addresses creep-fatigue
testing. Only public descriptions were consulted; clause-level conformity is
not established. Neither listing supplies a universal software pass threshold.

**Current decision:** numerical diagnostic available; P92 engineering response
and life qualification remain open. Do not promote this research fit to a
default engineering material or conceal the failed cross-protocol evidence.

## Additional data search

[NIMS CDS](https://cds.nims.go.jp/) is a credible next source for creep rupture
data, but its service explicitly requires registration and usage application.
No account was created and no restricted dataset was acquired. A public
announcement of data sheet 48B is not the underlying specimen table. Public
BAM RF, SLR and SLC histories remain useful for response development, not a
replicate cohort sufficient to certify a life distribution.
