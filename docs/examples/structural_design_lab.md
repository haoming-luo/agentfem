---
title: AgentFEM × GINO — interactive structural-design demonstration
description: Explore a 3D support under a 10 kN load and compare mass and deflection. See how finite-element data and learned field models support design exploration.
---

# AgentFEM × GINO: explore structural design

**The load stays the same. How much material can the support lose?**

This browser demonstration turns a familiar engineering trade-off into
something you can inspect: a lighter support uses less material, but its
deflection still has to meet the chosen design target.

[Open the structural-design lab](https://lab.haoming-luo.com/structural-design/){ .md-button .md-button--primary }

No account or installation is required. A desktop browser gives the best
side-by-side view. The interface is in Chinese; the sequence below explains
the controls in English.

## Try this comparison

1. Rotate the initial support to see the two arms and the installation gap.
2. Set **顶部最多下沉**: the maximum downward movement allowed for this comparison.
3. Click **寻找更轻支架结构** to select a lighter candidate meeting the predicted
   displacement condition.
4. Compare the mass and downward movement against the original structure.
5. Switch **看结构 / 看应力 / 看变形** to inspect geometry, stress, or deformation.

The deformation scale enlarges the display only. It does not change the load
or the computed displacement.

## The engineering question

The demonstration uses an approximately 189 mm tall aluminium support under
a **10 kN downward static load**, with its two feet fixed. It illustrates
small-displacement, linear-elastic behavior. The adjustable displacement
limit is a design-exploration target, not a material-strength certification.

The original data were produced at 1 kN and scaled to the displayed load using
linear elasticity; the demonstration's method notes describe the additional
FEM checks. This scaling is not a rule for plasticity, changing contact, or
large deformation.

## Where AgentFEM and GINO enter

AgentFEM supplied finite-element structure and displacement data. The learning
workflow used **88 structures: 64 training, 12 validation, and 12 test cases**,
plus additional confirmation structures. A GINO model with a displacement
correction network then supplied predictions for design exploration.

The public lightweight page uses **precomputed candidate results**, including
48 candidates, so it can run without a Python backend. Custom parameter
choices are matched to a nearby stored candidate; it does not run fresh GINO
inference in the browser. The full local workflow uses Python and the learned
model. Open **数据与方法** for the demonstration's data and method notes.

## The larger idea

The value is a shorter loop between changing a design and understanding its
consequences. Simulation builds the data; learning makes repeated exploration
practical; the interface makes the trade-off visible.

This is candidate screening within a declared design family, not a claim of
global optimization or unrestricted topology generation.

Want to build the underlying workflow?

- [Generate a dataset and fit a first surrogate](simulation_to_surrogate.md).
- [Explore AgentFEM's executable examples](index.md).
- [Connect a compatible AI agent](ai_assisted_fem.md).
