# Positive-strength constraint for continuous cyclic predictions

This is a follow-up research check, not an engineering material card. PR #188
remains frozen at 3052bf01 while its full CI runs.

## Observed defect in the short-path coefficient set

The [BAM protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319173/), Table 1,
defines a 368 s RF cycle: ramps of 2, 4 and 2 s with two 180 s holds, at
strain amplitude .002 and ramp magnitude .001/s. Simulating that **nominal**
waveform is not reconstructing missing measurements. The measured clock drifts
relative to that ideal waveform; pointwise late-time comparisons require their
own alignment, whereas cycle peaks can be compared without such resampling.

Using the frozen three-cycle RF fit with continuous state, Radau and BDF both
reject the first ramp of cycle 422 because k+R becomes nonpositive. At cycle
100 the Radau tensile peak is 238.5193 MPa versus 237.26 MPa observed; at 200 it
is 228.7084 versus 220.75 MPa. BDF agrees within 1e-6 MPa at these peaks.
An initially good peak prediction therefore does not establish valid full-path
parameters. No negative strength is clipped and no failure life is inferred.

## Derived sufficient invariant

The isotropic equation from [the published recovery law](https://doi.org/10.3390/met10030307)
can be written as

`Rdot = b (Q + H/b - R) pdot + gamma_r(p) |Qr-R|^mr sign(Qr-R)`.

For b>0, pdot>=0, nonnegative recovery and the admitted locally Lipschitz
exponents, both terms point inward at
`L = min(R_initial, Q + H/b, Qr)`. Thus R>=L and `k+L>0` is a sufficient
continuous-path positive-strength condition. It is not a proof of numerical
stability, experimental accuracy, or a complete thermodynamic admissibility
condition. Runtime state-domain checks remain necessary.

## Frozen follow-up experiment, before fitting

Keep RF cycles 1--3 for calibration and the original E/Z ranges, k upper bound,
all other coefficients, initial state and numerical settings. Change only the
k lower bound to `max(50 MPa, -L + 1e-6 MPa)`. The tiny positive margin prevents
roundoff at zero; it is not a design safety margin. The old research screen
remains reproducible; the alternative is explicitly selected with
`--positive-strength`.

Do not use later stresses to choose the bound or tune parameters. Compare the
new predictions on the already inspected RF/SLR records as a model-revision
check, **not fresh blind validation**. Then check the nominal full path with
independent Radau/BDF integrations. Preserve poorer fits and any domain failure.
No source data are relabeled as generated loading histories or independent
lifetime replicates.

## Measured results

The constrained fit converged in nine evaluations: E=155176.4711 MPa,
k=115.99827260494216 MPa and Z=1192.4850 MPa s^(1/3). The k lower bound is
active; the sufficient strength margin is only 1.000004e-6 MPa. This is an
admissible research fit, not an accepted engineering card.

RF time-weighted RMS errors for cycles 1--3 are 8.869, 5.767 and 5.410 MPa;
cycle 10 is 21.298 MPa. SLR frozen-parameter errors are 63.851 MPa in cycle 1
and 30.294 MPa in cycle 10, worse than the original unconstrained fit.
The correction therefore solves a domain defect, not cross-protocol accuracy.

Both Radau and BDF complete 1000 nominal cycles without resets. Their retained
cycle peaks agree within 2e-6 MPa. This verifies numerical agreement for this
path, not agreement with experiments. Selected peak comparisons (MPa):

| Cycle | Nominal constrained prediction | Measured RF peak |
| --- | ---: | ---: |
| 100 | 223.9086 | 237.26 |
| 200 | 213.5128 | 220.75 |
| 500 | 202.2914 | 198.74 |
| 1000 | 198.9290 | 181.74 |

The extension to all 2547 nominal cycles also completes with both integrators.
At the source-defined failure count 2054, the predicted peak is 198.7008 MPa;
at total test count 2547 it is 198.7002 MPa. The law approaches a stable cycle,
not failure. These two source counts must remain distinct. Continuing the
nominal equation beyond measured failure is a mathematical domain check only,
not a claim that the physical specimen remains intact.

The cycle-1000 peak is about 9.46% high. Nominal loading differs from the
recorded waveform, and the model contains no fatigue damage variable. Do not
interpret this difference as a calibrated damage law or infer rupture life.

Evidence files under `evidence/creep/`:

- `2026-11-16-bam-positive-strength.json`: constrained measured-input RF fit.
- `2026-11-16-bam-positive-strength-slr.json`: unchanged-parameter SLR prediction.
- `2026-11-16-bam-positive-strength-long-cycles.json`: two independent integrators;
  all 1000 cycles integrated, seven representative cycle records retained.
- `2026-11-16-bam-positive-strength-full-nominal.json`: extension to all 2547
  nominal cycles, retaining eight records and the separate source failure count.
- `2026-11-16-life-interval-coverage.json`: reproducible synthetic sampling
  diagnostic, separate from P92 evidence.

The first two JSON records retain the original screen's role vocabulary for
reproducibility; their later cycles have already been inspected and are model
revision checks, not a new blind validation. Subsequent tool output explicitly
labels that distinction. The SLR record hashes the unchanged RF record.

## Reproduce without private data

Download the RF and SLR files from the cited Zenodo record; the reader verifies
their published checksums before using observations. Run from the repository
in the FEniCSx environment:

```sh
python -m tools.p92_bam_cyclic_screen /path/Data_RF.lis --positive-strength --radau
python -m tools.p92_bam_cyclic_screen /path/Data_SLR.lis --frozen-rf evidence/creep/2026-11-16-bam-positive-strength.json
python -m tools.p92_nominal_cycles evidence/creep/2026-11-16-bam-positive-strength.json --cycles 1000 --method Radau
python -m tools.p92_nominal_cycles evidence/creep/2026-11-16-bam-positive-strength.json --cycles 1000 --method BDF
python -m tools.life_interval_coverage
```

Use `--cycles 2547` for the full nominal extension. Final local acceptance:
400 targeted P92/life tests pass, correctness lint passes, generated-document
check and strict documentation build pass. PR #188's frozen 3052bf01 head has
all applicable CI green, including installed-wheel macOS acceptance; this
follow-up remains a separate review and does not change release status.

At 2000 synthetic repetitions, the 95% pointwise log-log interval has conditional
coverage 94.65% for n=20 and 94.7% for n=100. At n=5, 368/2000 trials cannot
supply a finite interval at the chosen horizon; reporting only the conditional
100% coverage would conceal this limitation. These are sampling diagnostics,
not simultaneous bands, exact coverage proofs, or P92 specimen scatter.

## Next scientific gate

Do not expand the optimizer to ten poorly conditioned parameters merely to
reduce these errors. Separate recovery/softening calibration needs documented
multiple loading protocols and an untouched specimen validation set. The
current three BAM protocols are not replicate lifetimes; they cannot establish
heat-to-heat or specimen-to-specimen life uncertainty. Continue reporting
source identity, numerical refinement, response error and statistical evidence
as separate quantities.
