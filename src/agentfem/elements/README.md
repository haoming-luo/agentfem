# Elements

Reusable finite-element identities, discretization compatibility checks, and
explicit execution policies. Basix and DOLFINx own basis construction,
quadrature, degree-of-freedom maps and ordinary assembly; do not duplicate them.

Ordinary cases create fields through `agentfem.fields` and add them to a Model.
Use `elements.describe_element(field.space)` to inspect the actual element and
`mesh.compatibility_matrix()` for the declared topology capabilities. Source
cell names, readable connectivity and verified numerical formulations are
different things. A source shell, hybrid or reduced-integration label must not
silently become a standard displacement solid.

Add code here only when an element-level concept is reusable across
applications. Extend an existing backend-supported route with import, patch,
field-output and recovery tests before declaring another supported formulation.
