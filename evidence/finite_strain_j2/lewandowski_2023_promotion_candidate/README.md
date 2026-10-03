# Lewandowski beam promotion candidate

This directory is a compact evidence archive for the Lewandowski et al.
finite-strain J2 self-weight beam. It intentionally keeps the aggregate
promotion **incomplete** until every constituent run has been refreshed from
one executable package identity.

The archived evidence establishes:

- agreement with the independently reexecuted public curve: normalized RMS
  `7.71034e-6`, normalized maximum `2.57725e-5`;
- three-level spatial convergence (`24x4x6`, `30x5x8`, `36x6x10`): final-pair
  normalized RMS `0.7054%`, maximum `1.7861%`, both within the fixed project
  contract;
- serial/four-rank curve equivalence: normalized RMS `1.27829e-15`, maximum
  `4.13957e-15`;
- scale-aware full-state checkpoint/restart equivalence;
- a completed four-rank, central-difference 180-increment path with 180 accepted
  attempts, no cutback, a final downward displacement of `0.109482534 m`, and
  a clean source identity;
- decreasing three-level increment differences (`45`, `90`, `180`): the
  90-to-180 normalized RMS is `0.04460%` and the maximum is `0.21498%`, both
  within the fixed `0.2%` and `0.5%` contracts.

The numerical increment-convergence gate and the external comparison now pass
without changing a tolerance. Aggregate promotion remains closed for a
different, provenance-only reason: the archived mesh, rank, restart, 45- and
90-increment runs bind package tree `3f8208...`, while the completed
180-increment run binds the clean package tree `75713f...`. The assessor
therefore refuses to combine them into one content-bound release claim. A
future evidence refresh must rerun the compact lower levels from one package
identity; it does not require another algorithm or a relaxed threshold.

`promotion.json` is the authoritative aggregate. Every candidate curve is
paired with its original `assessment.json`; large checkpoint arrays and
temporary solver state are deliberately excluded.
