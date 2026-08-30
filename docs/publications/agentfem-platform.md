---
title: "AgentFEM: An AI-Native Open-Source Platform for Finite-Element Computing"
description: "The AgentFEM technical report: architecture, founding principles, representative workflows, and the author's vision for AI-native finite-element computing."
citation_title: "AgentFEM: An AI-Native Open-Source Platform for Finite-Element Computing"
citation_author: "Haoming Luo"
citation_publication_date: "2026/08/21"
citation_institution: "Xi'an Thermal Power Research Institute"
citation_pdf_url: "https://haoming-luo.github.io/agentfem/assets/papers/agentfem_platform.pdf"
---

# AgentFEM: An AI-Native Open-Source Platform for Finite-Element Computing

**Haoming Luo**  
Xi'an Thermal Power Research Institute, Xi'an 710032, China  
Technical Report · August 2026

<div class="publication-actions" markdown>

[Read the technical report (PDF)](../assets/papers/agentfem_platform.pdf){ .md-button .md-button--primary }
[View AgentFEM on GitHub](https://github.com/haoming-luo/agentfem){ .md-button }

</div>

This technical report is AgentFEM's founding statement: why the platform
exists, how its architecture embodies that purpose, and where AI-native
finite-element computing should go.

## Abstract

AI agents can increasingly construct solver programs, but dependable
finite-element simulation still requires explicit models, numerical
procedures, and evidence. This report presents AgentFEM, an open-source
platform for finite-element workflows shared by people and agents. Its
readable Python language expresses the engineering model, a reusable
capability layer implements finite-element methods, and FEniCSx, UFL, PETSc,
and MPI provide the numerical foundation. Results record solver status,
checks, and origin, and the same workflow extends to parameter studies and
surrogate modelling.

Three cases demonstrate the present scope: a linear-elastic beam, wave
propagation through an inclusion, and finite-strain homogenization of a
periodic porous cell using C3D10H elements. The current release also provides
a direct path from parameterized simulation to learning-ready data and
guarded surrogate models. AgentFEM is the AI-native CAE platform; agents and
development environments provide ways to work with it. Its broader purpose is
to make dependable finite-element software freely available and to prepare
CAE for a future in which people and AI advance science together.

## What the report presents

- a shared finite-element workflow for humans and AI agents;
- the engineering, finite-element capability, and numerical-kernel layers;
- explicit distinctions among computed, converged, verified, and validated
  results;
- representative static, wave-propagation, and finite-strain applications;
- the path from parameterized simulation to scientific datasets and guarded
  surrogate models.

## Citation

```bibtex
@techreport{luo2026agentfem,
  author      = {Luo, Haoming},
  title       = {AgentFEM: An AI-Native Open-Source Platform for
                 Finite-Element Computing},
  institution = {Xi'an Thermal Power Research Institute},
  year        = {2026},
  month       = {August},
  type        = {Technical Report},
  url         = {https://haoming-luo.github.io/agentfem/publications/agentfem-platform/}
}
```

The technical report describes AgentFEM version 0.2.1. The project continues
to evolve; consult the [release notes](../release_0.3.0.md) for subsequent
capabilities.

**Keywords:** finite-element method; scientific software; AI agents;
computational mechanics; simulation workflows; surrogate modelling; FEniCSx
