# Elements

Reserved namespace for reusable finite-element family descriptions,
interpolation choices, quadrature policies, and integration-rule helpers.

Current AgentFEM workflows should still create DOLFINx spaces through
`agentfem.spaces`. Add code here only when an element-level concept is reusable
across applications.
