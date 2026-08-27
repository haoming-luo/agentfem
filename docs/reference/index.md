# Theory and reference

Reference pages answer precise questions about equations, conventions, output
variables, supported combinations, and public interfaces. Start with the
[user guide](../guide/index.md) when constructing your first model.

## Mathematical and scientific reference

- [Theory and conventions](theory_and_conventions.md) gives the common balance
  laws, kinematics, constitutive equations, procedure distinctions, and result
  locations used by the current platform.
- [RVE homogenization and physical field statistics](rve_homogenization_and_statistics.md)
  defines physical quadrature weights, finite-strain macro averages,
  stress-state conventions, Hill--Mandel evidence and accepted-increment
  convergence records.
- [Scientific function reference](scientific_function_reference.md) is
  generated from reviewed knowledge cards. Each entry records formulas,
  assumptions, inputs, outputs, tests, benchmarks, consumers, and limitations.
- [Scientific operator contracts](../operator_contracts.md) defines the roles
  and composition of \(K\), \(M\), \(C\), \(F\), residuals, tangents, and
  functionals.
- [Output variables and field semantics](../result_field_semantics.md) defines
  names, tensor meanings, field locations, projection/recovery, and scientific
  versus presentation output.

## Workflow and interoperability reference

- [Stable steps and compact output](../step_and_output_architecture.md) defines
  increment, iteration, frame, progress, checkpoint, and visualization
  contracts.
- [Mesh interoperability](../mesh_interoperability.md) documents native and
  imported meshes, source semantics, element mapping, and optional tools.
- [Module map](../module_map.md) identifies the responsible extension point for
  each modeling concept.

## Python API

The [Python API](api.md) provides generated public signatures and concise
call-level summaries. It is a lookup reference; it does not replace the theory,
workflow, or example pages that define when a call is scientifically
appropriate.

## Reference-page standard

A mature material, procedure, element, or output page should state:

1. purpose and applicability;
2. governing equation or definition;
3. required inputs and units;
4. algorithm and state update;
5. available output and result locations;
6. incompatible combinations and known limitations;
7. executable tests, benchmarks, and primary references.

This keeps the documentation useful both for learning a workflow and for
checking one technical definition during an engineering analysis.
