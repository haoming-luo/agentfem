# AgentFEM 0.2.2

AgentFEM 0.2.2 establishes an open scientific-learning interface without
turning the finite-element core into a machine-learning framework. A
laboratory-owned PyTorch, JAX, DeepXDE, or other field solver can now enter the
ordinary AgentFEM lifecycle through one framework-neutral boundary:

```python
step = model.step(
    target=neural_field_spec,
    executor=my_solver,
    executor_name="laboratory.my_solver",
    executor_version="1.0",
    executor_options={"epochs": 2000},
    output="results/neural_field",
)
result = step.solve_result()
```

The executor owns its tensors, architecture, optimizer, devices, and model
state. AgentFEM owns the immutable scientific request, executor identity,
common `SimulationResult`, portable manifest, and verification evidence. No
AgentFEM model inheritance or official learning package is required.

## Scientific-learning contracts

The public `agentfem.learning` namespace now distinguishes four roles:

- surrogates approximate a declared parameter-to-response map;
- neural operators learn a function-to-function map;
- neural-field solvers optimize one physical field problem;
- learned constitutive models update local material state.

`NeuralFieldSpec` records fields, representations, residual or variational
objectives, conditions, sampling, trainable physical parameters, and required
independent checks. The contract can describe PINN, VPINN, Deep-Ritz/DEM,
XDEM, and related methods without claiming that arbitrary UFL forms can be
translated automatically or that their external trainers are bundled.

The optional
[AgentFEM-Learning](https://github.com/haoming-luo/agentfem-learning)
companion provides maintained method bindings, examples, dependency policy,
and benchmark evidence. Its first experimental XDEM subdomain proves the full
`NeuralFieldSpec -> Step -> SimulationResult` path against a Williams Mode-III
reference. User and private providers remain free to use the same core
contract directly.

## Agent-first entry

The README now gives coding agents a direct first instruction before the
manual installation route. The repository already ships the corresponding
agent guide, reusable Skill, machine-readable capabilities, project templates,
health checks, structured results, and verification commands. Human and agent
users therefore operate the same public engineering workflow rather than two
separate products.

## Release evidence

The candidate is accepted only after:

- version, citation, tag, release contract, wheel, and source archive agree;
- the complete serial suite and distributed MPI suites pass;
- checkpoint state remains portable across supported MPI rank counts;
- the built wheel contains the neural-field contract and execution boundary;
- a user-owned executor completes the common Step and result lifecycle in
  serial and MPI tests;
- optional PyTorch dataset and surrogate bridges pass from the candidate
  wheel;
- every installed project template and release-facing FEM workflow executes;
- the documentation, machine entrypoints, and scientific knowledge catalog
  pass their strict checks.

This release creates an integration boundary; it does not claim bundled
general XDEM, universal PINN training, automatic neural operators for arbitrary
meshes, or validity of an external learned model without independent evidence.
