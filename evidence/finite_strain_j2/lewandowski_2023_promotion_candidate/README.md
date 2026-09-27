# Lewandowski beam promotion candidate

This directory is a compact, content-bound evidence archive for the
Lewandowski et al. finite-strain J2 self-weight beam. It intentionally records
an **incomplete** promotion candidate rather than converting a near pass into a
capability claim.

The archived evidence establishes:

- agreement with the independently reexecuted public curve: normalized RMS
  `7.71034e-6`, normalized maximum `2.57725e-5`;
- three-level spatial convergence (`24x4x6`, `30x5x8`, `36x6x10`): final-pair
  normalized RMS `0.7054%`, maximum `1.7861%`, both within the fixed project
  contract;
- serial/four-rank curve equivalence: normalized RMS `1.27829e-15`, maximum
  `4.13957e-15`;
- scale-aware full-state checkpoint/restart equivalence;
- decreasing three-level increment differences (`15`, `45`, `90`), with RMS
  `0.1101%` within the `0.2%` contract.

Promotion remains closed because the final-pair increment maximum is `0.6784%`,
above the predeclared `0.5%` contract. The maximum occurs at the sharply curved
yield transition between coarse load nodes. A diagnostic 180-increment prefix
reduced the local 90-to-180 maximum through load factor `0.5444` to about
`0.1204%`, but the full 180-increment path was not accepted as release evidence:
the local runtime entered an abnormally slow PETSc solve regime, and the run was
stopped rather than weakening the gate or archiving a partial path as complete.

`promotion.json` is the authoritative aggregate. Every candidate curve is
paired with its original `assessment.json`; large checkpoint arrays and
temporary solver state are deliberately excluded.
