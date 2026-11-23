# Check both measurement channels before attributing tiny rates to a mechanism

The nominal-control screen's largest early normalized error is the second
SLR long-hold late stress secant. Its measured stress rises by 0.61 MPa;
the third rises by 0.91 MPa. These exceed the stress-only five-quantum rule,
but a recorded strain quantum of 1e-5 corresponds to 1.552 MPa with the
frozen modulus. The strain channel must not disappear from the interpretation.

The [NIST propagation guidance](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty)
and [JCGM measurement-model guidance](https://www.bipm.org/documents/20126/50065290/JCGM_GUM_6_2020.pdf/d4e77d99-3870-0908-ff37-c1b6a230a337)
require relating the output to the relevant measured inputs. They do not
supply an uncertainty distribution or acceptance tolerance for this dataset.

For fixed E the inferred nonelastic increment is Delta-epsilon-Delta-sigma/E.
Independently perturbing each endpoint by plus/minus one recorded quantum
gives the exact diagnostic half-width `2*q_epsilon + 2*q_sigma/E`. Tests
enumerate all sixteen corners. This is not a probabilistic confidence interval,
an instrument-accuracy certificate, or a measured controller disturbance.
No modulus uncertainty, temperature covariance or extensometer accuracy is
available in this record; the calculation is deliberately limited.

The first three SLR zero-strain long-hold tail increments all fail to retain
their inferred nonelastic sign under that perturbation. By contrast, this
does not erase the substantial SLC post-creep hold discrepancy. The original
response policy and failed checks remain unchanged. Do not rescale tolerances,
silently exclude a failed check, or force the material law to reproduce a
tiny apparent rate without disclosing its input-channel sensitivity.

The next scientific step is to keep nominal-control acceptance distinct from
an observation-consistency assessment, using supported input uncertainty or
independent controller data where available. A sophisticated statistical
model without justified uncertainty inputs would not improve this evidence.
The archived audit is `2026-11-22-control-resolution-audit.json`; no constitutive
parameter, fit, production behavior or qualification is changed.

## Instrument cross-check

The [manufacturer's 632.51F-04 page](https://www.mts.com/en/products/materials/extensometers/water-cooled-high-temperature-extensometers)
confirms a 12 mm gauge and water-cooled high-temperature axial measurement,
consistent with the BAM paper. It states calibration capability to ASTM E83
Class B1 and ISO 9513 Class 0.5. This current product specification is not
the historical specimen's calibration certificate, actual noise, drift, or
a probability distribution. Do not substitute its class number for a
dataset-specific uncertainty or treat the text-file quantum as the complete
measurement uncertainty. The [manufacturer's calibration guidance](https://www.mts.com/en/services/calibration-metrology/extensometer-calibration)
likewise distinguishes traceable calibration from mere product identity.
