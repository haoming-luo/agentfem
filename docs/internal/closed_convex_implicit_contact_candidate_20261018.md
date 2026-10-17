# Closed-convex sharp rigid tool: bounded implicit contact candidate

Status: source-level, serial, frictionless finite-J2 candidate. This does not
promote arbitrary STL/CAD, general contact, or the 0.4.3 release contract.

## Geometry and Operator

`convex_polyhedral_rigid_surface(...)` accepts a closed, consistently oriented
triangle manifold only if every supplied vertex lies behind every outward
facet plane. It rejects open, inward-oriented and nonconvex tooling before a
solve. This is a geometry certificate for the exact supplied facets, not a
repair, remesh or convex-hull operation.

For a point inside the tool, the signed gap is the maximum of its outward
unit-normal face-plane gaps. A negative value means penetration. The nearest
limiting face is selected by stable facet ID on a tie. Outside, the inactive
branch reports Euclidean distance to the actual triangle surface. Every
Newton trial reprojects at the current prescribed pose. The active residual
is the derivative of the reference-area normal penalty potential on the
selected face; its within-branch tangent has zero geometric Hessian. At a
noncoplanar face switch the potential remains continuous but the force has
distinct one-sided limits. The selected limiting tangent is a semismooth
branch, **not** a classical two-sided derivative or a smoothed normal.

The implementation intentionally uses exhaustive facet queries as a small
correctness reference. It is not a scalable arbitrary-tool search. It shares
the existing rigid pair, finite-J2 Newton, strong-boundary lifting, tool
force/moment work, accepted State, rollback and checkpoint owners. It does not
add a new Step or a geometry-specific material law.

## Evidence and limits

Focused solver-neutral tests check closure/orientation rejection, interior and
exterior signed gaps, a noncoplanar feature switch, fixed-branch energy
gradient, tangent difference, and rigid-motion objectivity. A real DOLFINx
boundary operator checks its current-trial tangent under both fixed and
translated/rotated poses. A finite-J2 `model.step()` case compares the same
planar face supplied as a closed convex tool versus the older open triangle
patch; displacement and accepted tool work agree at the tested resolution.
These are local tests, not industrial contact validation.

The candidate still lacks a structural benchmark that *loads through* a
noncoplanar face switch, an independent force/pressure comparison, mesh and
penalty studies, MPI assembly, friction on sharp faces, and performance
evidence for larger tools. At a medial-axis tie, one limiting branch is
selected deterministically; general convergence across repeated active-set
switches is not proven. Open or nonconvex triangle surfaces retain their
fail-closed active edge/vertex policy. Release language must distinguish
this narrow convex solid contract from Abaqus/Standard general contact.
