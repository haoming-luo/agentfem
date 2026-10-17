# Closed-convex sharp rigid tool: bounded implicit contact candidate

Status: source-level, serial, frictionless finite-J2 candidate. This does not
promote arbitrary STL/CAD, general contact, or the 0.4.3 release contract.

## Geometry and Operator

`convex_polyhedral_rigid_surface(...)` accepts a closed, consistently oriented
triangle manifold only if every supplied vertex lies behind every outward
facet plane. It rejects open, inward-oriented and nonconvex tooling before a
solve. This is a geometry certificate for the exact supplied facets, not a
repair, remesh or convex-hull operation.
`convex_polyhedral_rigid_surface_from_mesh(...)` applies that same certificate
to the reviewed triangle-only mesh import, retaining source hash, format and
coordinate-scale evidence; it does not certify arbitrary STL files as convex.

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
An independent one-dimensional equilibrium also checks the homogeneous
elastic-regime Hencky bar against
`E log(lambda)/lambda + k(lambda - x_tool) = 0`, with the penalty potential
compared separately. This validates a bounded normal-contact limit, not a
curved pressure field or a plastic structure.

A separate twelve-increment finite-J2 fixture moves a convex tool far enough
that at least one *active* slave quadrature point switches between
noncoplanar face normals. The same path is run in an elastic regime and with
evolving plastic State. Every accepted Newton increment converges, plastic
equivalent strain becomes nonzero in the latter, and an interrupted checkpoint
resumed before the switch reproduces the continuous displacement, material
State and final prescribed-tool work. This demonstrates one selected-branch
transition path; it is not a convergence theorem for arbitrary corners, large
sliding, or chattering active sets.

An isolated, locally built `0.4.3` wheel was installed outside the source
checkout and ran `examples/finite_j2_closed_convex_contact.py` through the
ordinary `model.step()` path. One-cell/eight-increment and two-cell/
sixteen-increment cases report right-end displacements
`-0.0009990007775119401` and `-0.0009990007775003966`; the independently
solved scalar equilibrium gives `-0.0009990007774995924`. The examples use
the same uniform deformation, so this is an installed-package and exact-limit
smoke, not a mesh convergence study for a nonuniform contact patch. The wheel
was built for the local Apple Silicon environment, not released or tested on
Windows/Linux by this evidence.
These are local tests, not industrial contact validation.

The candidate still lacks an external structural comparison for a sharp
feature, mesh and penalty studies, MPI assembly, friction on sharp faces,
and performance
evidence for larger tools. At a medial-axis tie, one limiting branch is
selected deterministically; general convergence across repeated active-set
switches is not proven. Open or nonconvex triangle surfaces retain their
fail-closed active edge/vertex policy. Release language must distinguish
this narrow convex solid contract from Abaqus/Standard general contact.
