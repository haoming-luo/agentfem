# P92 external source admission audit

## Decision

Do not convert a grade name into a shared calibration identity. Keep the
published 600/650 C dwell response, pure-fatigue reproduction and life-table
holdout as distinct research evidence. Engineering interaction life still needs
compatible rupture, fatigue and dwell validation data and an explicit failure
definition. A numerical failure or solver cutback is not a physical rupture test.

## Additional open reference

[Imperial thesis, DOI 10.25560/60640](https://doi.org/10.25560/60640),
*Prediction of long-term static and cyclic creep rupture and crack growth of
grade 92 steels under different stress states*. Public PDF SHA256:
`57abdf756ac1198b3aab53d6b9ebe2e5043976401b49cb8e54d90e53d783c273`.

Sections 3.2 and 4.4 identify the fatigue cohort as ex-service parent material
(540 C, 22,000 h); cross-weld creep specimens belong to another cohort. Two
fatigue tests were scrapped after thermocouple detachment. Neither the cohorts
nor those invalid tests may be silently pooled with the existing references.

Visual checks of Table 5-4 (PDF p114 / printed p89) and Table 8-1 (PDF p175 /
printed p150) reveal inconsistent long-term Norton coefficients: the values
3.8e-18 and 1.7e-23 are associated with opposite temperatures, while the
exponents retain their ordering. Do not choose a table by which fit looks best;
the constants require clarification before constitutive use.

Table 4-3 (PDF p94 / printed p69) was also visually checked. It provides an
independent service-aged fatigue cohort for future transfer assessment, not
same-batch validation. No thesis figures or full text are redistributed here.

## What proceeds without those constants

- Generic prescribed-temperature input with committed-state/time semantics.
- Co-located accepted stress, strain, temperature and state histories.
- Public-data reproduction with explicit source/cohort and failure definitions.
- Domain rejection and restart tests independent of fitted material values.

## What remains a data gate

One traceable target material condition, compatible pure-fatigue and creep
rupture observations, strain-rate and environment ranges, and independent
dwell-life observations. Calibration and validation must remain separate. The
2019/2020 duplicated observations already identified in the life-screen audit
do not supply an independent validation cohort. A local source-access failure
or a paid standard does not justify inventing missing numeric data.

## Calibration entry audit

The research life screen now validates in-memory inputs as well as file reads.
Duplicate/overlapping split IDs, a missing holdout, and invalid observed lives
cannot bypass the data gate through `screen(data)` or `calibrate(data)`.
The existing held-out predictions remain unchanged. The life, pure-fatigue,
cyclic-reference and CI-routing regression passed 87 tests; this is software
verification, not 87 new experimental observations.

## Two-rate follow-up (2026-11-07)

[Seifert, Schlayer and Maier, FVWHT 2025, pp. 36-47](https://opus.hs-offenburg.de/files/11628/SeifertSchlayerMaier_PlastCrpFat_FVWHT_2025.pdf)
is openly licensed CC BY 4.0. The retrieved PDF SHA256 is
`3db99ebff289aa4ec91c67320b6d957d23f0776db23672375856e4f20ac256af`.
Equations on printed pp. 37-38 and the scope on p. 39 were visually checked.

The two-rate Chaboche-Jiang formulation distinguishes long and short response
times, with kinematic recovery and ratcheting control. It offers a mechanism
candidate for paths that a single creep law misses. However, its reported
600 C cohort is BED/519n, not the BAM 620 C cohort; complete numerical
parameters are not supplied in this contribution. The law describes primary
and secondary creep, not tertiary rupture. Do not invent coefficients or call
its response curves an independent lifetime prediction.

The same contribution compares full viscoplastic FEM with elastic-FEM-driven
local postprocessing, including a Neuber approximation. That is an alternative
approximate assessment route, not an equivalent faster solver. Before adopting
it, use co-located histories and compare stress redistribution and accumulated
strain against nonlinear FEM. Missing parameters, source normalization and
applicability remain explicit gates; no new production material follows from
this literature review.
