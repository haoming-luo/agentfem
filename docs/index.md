---
title: AgentFEM
hide:
  - navigation
  - toc
  - path
---

<section class="af-hero">
  <div class="af-hero-copy">
    <span class="af-kicker">OPEN-SOURCE · AI-NATIVE · FINITE ELEMENT COMPUTING</span>
    <h1>Finite-element workflows that humans and agents can build together.</h1>
    <p>
      AgentFEM turns engineering intent into readable models, dependable
      simulation workflows, inspectable results, and learning-ready data—without
      hiding the numerical mechanics.
    </p>
    <div class="af-actions">
      <a class="md-button md-button--primary" href="get_started/">Run your first simulation</a>
      <a class="md-button" href="examples/">Explore examples</a>
    </div>
  </div>
  <div class="af-hero-mark">
    <img src="assets/images/AgentFEM_logo_transparent.png" alt="AgentFEM">
  </div>
</section>

## Choose your path

<div class="grid cards af-paths" markdown>

-   **I solve engineering problems**

    Build solid, thermal, dynamic, and time-dependent analyses through a
    workflow that reads like finite-element modeling rather than backend glue.

    [:octicons-arrow-right-24: Start as an engineer](get_started/index.md)

-   **I connect simulation and AI**

    Run reproducible parameter campaigns, collect scientific datasets, connect
    PyTorch or your own models, and retain a high-fidelity FEM fallback.

    [:octicons-arrow-right-24: See simulation-to-learning](guide/simulation_to_learning.md)

-   **I build tools or operate as an agent**

    Use the same public Python workflow and structured CLI as human users, with
    machine-readable results, failures, documentation, and scientific evidence.

    [:octicons-arrow-right-24: Open the agent entry](agents/index.md)

</div>

## One workflow, several scales

```text
Study → Model → Mesh/Regions → Fields → Materials → Loads/Constraints
      → Solution Step → Results/Verification → Campaign/Data/Learning
```

The same model can support one engineering analysis, a distributed parameter
campaign, a reusable dataset, or an agent-operated workflow. Advanced users can
still reach UFL, DOLFINx, PETSc, MPI, and custom constitutive implementations.

## What you can do today

<div class="grid cards af-capabilities" markdown>

-   **Readable engineering models**

    Studies, regions, materials, fields, loads, constraints, solution steps,
    outputs, and verification remain explicit.

-   **Linear and nonlinear solids**

    Linear elasticity, thermoelasticity, Neo-Hookean finite strain, mixed
    displacement-pressure hyperelasticity, and a stateful small-strain J2 path.

-   **Heat, waves, and dynamics**

    Steady and transient heat transfer, implicit structural dynamics, and
    central-difference explicit wave workflows.

-   **Engineering results you can inspect**

    Unified fields, histories, resultants, progress, checkpoints, manifests,
    quality policies, Golden benchmarks, and explicit failure states.

-   **External meshes and parallel execution**

    XDMF, optional Gmsh and meshio routes, Abaqus quadratic tetrahedra and
    equation constraints, PETSc, MPI, and distributed workflows.

-   **Simulation to learning**

    Campaigns, accepted datasets, NumPy/PyTorch adapters, surrogate baselines,
    applicability guards, and FEM fallback.

</div>

## Flagship workflows

<div class="grid cards" markdown>

-   **A clear first solid model**

    A small cantilever shows the complete public workflow without hiding the
    finite-element concepts.

    [:octicons-code-24: View the basic example](examples/index.md#linear-solid-mechanics)

-   **Wave propagation with an inclusion**

    Explicit dynamics, heterogeneous material regions, boundary models, time
    histories, and field output in one reproducible workflow.

    [:octicons-code-24: View the wave example](examples/index.md#waves-and-dynamics)

-   **Imported 3D periodic hyperelasticity**

    Abaqus mesh and equation import, quadratic tetrahedra, distributed periodic
    constraints, finite strain, homogenized response, and visualization output.

    [:octicons-code-24: View the periodic-cell example](examples/index.md#nonlinear-solids)

</div>

## Built for useful openness

AgentFEM is open at every layer: begin with a concise engineering workflow,
inspect the scientific and numerical evidence, and descend into reusable
operators or the FEniCSx kernel when a problem requires it.

[Read the product roadmap](product_roadmap.md){ .md-button }
[Browse the complete reference](reference/index.md){ .md-button }

<div class="af-attribution">
AgentFEM was initiated by Haoming Luo and open-sourced on GitHub in July 2026.
</div>
