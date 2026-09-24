# Keep both rotation-free shell edge pairs explicit

## Context

The first embedded fabric membrane plus neighbour-reconstructed bending
composition exposed a physically important failure: prescribing displacement
on an edge does not constrain the boundary slope of a rotation-free curvature
operator. Calling that edge clamped would therefore overstate both the model
and its verification.

Kirchhoff--Love boundary theory has two work-conjugate pairs. Boundary
displacement is paired with an effective boundary force, while boundary normal
rotation is paired with bending moment. The effective force may contain a
boundary derivative of the twisting moment and is not generally identical to
a raw section-force resultant. At each pair, either the kinematic member or
the dynamic member is controlled.

## Decision

AgentFEM records these two choices in
`RotationFreeEdgeBoundarySemantics`. The contract classifies an edge as:

- clamped when displacement and normal rotation are essential;
- simply supported when displacement is essential and bending moment natural;
- rotation guided when force is natural and normal rotation essential;
- free when effective force and bending moment are natural.

The object is inspectable but has no backend payload. A concrete shell
provider must own the definition and discretization of boundary normal
rotation, effective force, bending moment, corner terms, and their virtual
work. Until that lowering passes patch, work, and reaction tests, the contract
reports `unavailable_until_provider_verified`.

## Consequences

- Ordinary solid `DirichletBC` remains an honest displacement constraint and
  is never silently promoted to a shell clamp.
- Loads do not invent a generic edge moment without knowing the shell
  provider's derived rotation.
- Constraints own prescribed kinematics, loads own prescribed dynamic data,
  the Operator defines conjugacy, and the Procedure audits work and reaction.
- The same semantic split can later support C1, weakly enforced, mixed-
  director, and neighbouring-element formulations without changing the public
  engineering meaning.

## Verification gates

1. rigid-body rotation gives zero internal and boundary work;
2. a clamped patch constrains displacement and the provider-defined normal
   rotation;
3. a free edge produces zero effective force and bending moment;
4. applied edge moment closes external work against boundary rotation;
5. corner and twisting-moment terms are counted exactly once;
6. serial and MPI reactions, work, and accepted results agree.

## Primary references

- Duong, Itskov, and Sauer (2022),
  <https://doi.org/10.1002/nme.6937>.
- Ivannikov et al. (2015),
  <https://doi.org/10.1016/j.ijsolstr.2014.05.025>.
- Coradello et al. (2024),
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11615124/>.
