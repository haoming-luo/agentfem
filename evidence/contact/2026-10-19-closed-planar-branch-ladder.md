# Closed-tool smooth-branch sensitivity, 2026-10-19

Status: diagnostic evidence only. This is a synthetic serial finite-J2 cube
against one reviewed closed L-prism face. It does not reproduce an external
commercial-solver problem, establish mesh convergence, or validate sharp-edge
contact. All nine cases use identical geometry, material, tool motion and
Newton controls; each row changes only the named numerical choice.

Run from the repository root against the branch source:

```sh
PYTHONPATH=src python examples/finite_j2_closed_planar_branch_ladder.py \
  --output /tmp/closed-planar-branch-ladder.json
```

The source version for this record is PR #118 after the certified smooth
branch and per-pair integration mode commits. The output JSON contains the
unrounded values, status, integration claim and local wall time. Wall time is
not an acceptance threshold and cannot be compared across machines. These
values are from a single local run, not an uncertainty estimate.

| Axis | Cells/axis | Increments | Penalty | Maximum PEEQ | Tool work | Maximum sampled penetration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 6 | 5,000 | 0.264448 | 0.00528666 | 0.0111180 |
| mesh | 2 | 6 | 5,000 | 0.275917 | 0.00424685 | 0.00717791 |
| mesh | 4 | 6 | 5,000 | 0.271938 | 0.00383225 | 0.00489645 |
| increment | 1 | 3 | 5,000 | 0.264993 | 0.00546429 | 0.0112321 |
| increment | 1 | 12 | 5,000 | 0.264252 | 0.00515762 | 0.0110617 |
| increment | 1 | 24 | 5,000 | 0.264189 | 0.00508320 | 0.0110339 |
| penalty | 1 | 6 | 2,500 | 0.238251 | 0.00430952 | 0.0148842 |
| penalty | 1 | 6 | 10,000 | 0.273201 | 0.00608767 | 0.00774992 |
| penalty | 1 | 6 | 20,000 | 0.272218 | 0.00668094 | 0.00498298 |

Every run completed and reported a certified current whole-face integration.
That geometric certificate addresses the quadrature branch, **not** the
accuracy of finite element discretization, load stepping, or penalty
enforcement. In particular, tool work changes by about 9.8% from mesh 2 to
mesh 4 and about 9.7% from penalty 10,000 to 20,000. The increment ladder is
more settled but still changes by about 1.4% from 12 to 24. No extrapolated
limit or product validation claim follows from these three levels.
The maximum sampled penetration decreases as the penalty increases, as the
declared penalty law should suggest; the contact footprint also changes, so
these nine points do not isolate a hard-contact solution or a pressure field
convergence rate.
