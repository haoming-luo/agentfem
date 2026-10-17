# Implicit contact on prescribed DOFs and sphere-indentation screen

Status: experimental source candidate, not a Hertz validation or a general
contact promotion. The source branch follows merged PR #116.

## Numerical correction

The serial finite-J2 rigid-pair route previously rejected contact trace DOFs
shared with strong Dirichlet constraints. It now lifts the native contact
tangent by the same prescribed correction `g_bc - u_trial` used by DOLFINx for
the UFL material tangent. Native COO entries on constrained rows and columns
are omitted after the material matrix has had its strong rows/columns set by
DOLFINx. This preserves the total-operator action on free rows and the
identity rows on constrained DOFs. It does not change the frictionless
penalty law, rigid projection or material algorithm.

One active, tilted triangular-tool/J2 test imposes a nonzero displacement on
the contact face, completes, and checks every shared strong row remains an
identity row. A separate sphere-indentation test compares a full block with
the symmetry-reduced quarter block at identical physical resolution: 256
versus 64 hexahedra give the same full-body force to better than `1e-9`
relative difference, and four times the quarter contact potential equals
the full-block potential. These are structural equivalence checks, not
independent physical validation. Existing contact/MPC regression remains
required.

## Independent reference screen

`examples/finite_j2_hertz_contact.py` solves a prescribed rigid-sphere
indentation of a finite, bottom-clamped block. Its very high yield threshold
keeps the finite-strain J2 material in its elastic regime. The report keeps
the model separate from the classical small-strain, elastic-half-space,
exact-nonpenetration Hertz limit
`F = 4 E sqrt(R) delta^(3/2) / (3 (1-nu^2))`.
The [published sphere/half-space comparison](https://www.sciencedirect.com/science/article/pii/S0301679X15006027)
states that reference relation and its domain; it does not certify the finite
block or penalty contact used here.

Commands use the current source checkout and the reviewed FEniCSx runtime:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/finite_j2_hertz_contact.py --quarter --cells 8
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/finite_j2_hertz_contact.py --quarter --cells 8 --increments 16
```

At `E=1000`, `nu=0.3`, `R=2`, indentation `0.02`, penalty `1e5`, four
increments and unit half-width/depth, the half-space reference force is
`5.86080586`. Full-block mesh levels with 256, 2048, 6912 and 16384
hexahedra produce forces `7.99194`, `7.32959`, `7.16618`, and `7.09758`.
The finest screened ratio is still `1.2110`, so this is **not** a passed
Hertz accuracy gate. Finite boundaries, contact quadrature, mesh resolution
and finite penalty remain confounded. A deeper quarter block (half-width and
depth `2`, 4096 hexahedra) gives `6.65969` but changes both boundary distance
and resolution, so it cannot independently identify the discrepancy.

At 512 quarter-block cells, penalties `5e4`, `1e5`, `2e5` give full-body forces
`7.10580`, `7.32959`, `7.38030` and maximum penetrations `0.0015424`,
`0.0010021`, `0.00069635`. At fixed mesh/penalty, 4, 8 and 16 increments
leave the final force unchanged within the nonlinear tolerance. Their
accepted trapezoidal work residual declines `1.8809e-4`, `5.8245e-5`,
`2.0718e-5`; this is a path-discretization diagnostic, not a general first
law for plastic contact.

## Remaining gate

Run independently controlled mesh **and** domain-size refinement with a
resolved contact patch, then penalty and increment sensitivity. Compare
force, pressure footprint, penetration, reactions, tool work and plastic
state against either the valid Hertz limit or a frozen external FE result.
Do not turn the current force ratio into a release tolerance. Active
noncoplanar triangle seams/free edges, multiple implicit pairs, implicit
friction and distributed implicit assembly remain separate work.
