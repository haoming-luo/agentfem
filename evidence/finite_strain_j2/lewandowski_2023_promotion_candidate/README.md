# Lewandowski beam promotion candidate

This directory is the compact, content-bound evidence archive for the
Lewandowski et al. finite-strain J2 self-weight beam. Its machine-derived
aggregate promotion is **accepted**.

The archived evidence establishes:

- agreement with the independently reexecuted public curve: normalized RMS
  `7.71034e-6`, normalized maximum `2.57725e-5`;
- three-level spatial convergence (`24x4x6`, `30x5x8`, `36x6x10`): final-pair
  normalized RMS `0.7054%`, maximum `1.7861%`, both within the fixed project
  contract;
- serial/four-rank curve equivalence: normalized RMS `1.10189e-15`, maximum
  `4.99276e-15`;
- scale-aware full-state checkpoint/restart equivalence;
- a completed four-rank, central-difference 180-increment path with 180 accepted
  attempts, no cutback, a final downward displacement of `0.109482534 m`, and
  a clean source identity;
- decreasing three-level increment differences (`45`, `90`, `180`): the
  90-to-180 normalized RMS is `0.04460%` and the maximum is `0.21498%`, both
  within the fixed `0.2%` and `0.5%` contracts.

The external comparison, mesh convergence, increment convergence, MPI
equivalence and complete-state restart gates all pass without changing a
tolerance. Every consumed artifact binds the clean executable package tree
`75713f...`; harness-only Git commits may differ without changing that tree.
The fail-closed assessor therefore authorizes this benchmark promotion.

`promotion.json` is the authoritative aggregate. Every candidate curve is
paired with its original `assessment.json`; large checkpoint arrays and
temporary solver state are deliberately excluded.
