# Lewandowski beam promotion evidence

This directory is a compact, content-bound evidence archive for the
Lewandowski et al. finite-strain J2 self-weight beam. The machine-derived
`promotion.json` is **accepted** and authorizes the declared structure-level
verification claim for the ordinary strong-boundary provider.

The archived evidence establishes:

- agreement with the independently reexecuted public curve: normalized RMS
  `7.71034e-6`, normalized maximum `2.57725e-5`;
- three-level spatial convergence (`24x4x6`, `30x5x8`, `36x6x10`): final-pair
  normalized RMS `0.7054%`, maximum `1.7861%`, both within the fixed project
  contract;
- serial/four-rank curve equivalence: normalized RMS `7.5270e-16`, maximum
  `2.1488e-15`;
- scale-aware full-state checkpoint/restart equivalence;
- decreasing three-level increment differences (`45`, `90`, `180`), with the
  final-pair RMS `0.0446%` and maximum `0.2150%`, within the fixed `0.2%` and
  `0.5%` contracts.

The external comparison uses the pinned executable observer at `(1, 0, 0)`;
it does not claim to reproduce the differently described point A in the paper.
The project thresholds are AgentFEM promotion contracts, not tolerances stated
by the authors. No threshold was changed after observing a result.

`promotion.json` is the authoritative aggregate. Every candidate curve is
paired with its original `assessment.json`; large checkpoint arrays and
temporary solver state are deliberately excluded.
