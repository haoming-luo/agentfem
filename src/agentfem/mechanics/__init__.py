# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solid-mechanics solution procedures."""

from .creep import (
    CreepEnergyFrame,
    CreepIncrementInfo,
    CreepPathInfo,
    ImplicitCreepStep,
    implicit_creep_step,
)
from .finite_strain_plasticity import (
    ExperimentalFiniteStrainPlasticityStep,
    FiniteStrainJ2AffineTransaction,
    FiniteStrainJ2StateTransaction,
    FiniteStrainJ2StandardProblem,
    FiniteStrainPlasticityIncrementInfo,
    FiniteStrainPlasticityPathInfo,
    experimental_finite_strain_j2_step,
    finite_strain_j2_affine_problem,
    finite_strain_j2_mixed_affine_problem,
    finite_strain_j2_standard_problem,
)
from .harmonic import (
    DirectHarmonicStep,
    DirectHarmonicSweepStep,
    direct_harmonic_step,
    harmonic_frequency_sweep_step,
)
from .modal import ModalAnalysisStep
from .plasticity import (
    J2IncrementInfo,
    J2LoadPathInfo,
    J2PlasticityStep,
    j2_plasticity_step,
)
from .shell import (
    DirectorShellKinematics,
    FiberCurveKinematics,
    FibrousShellCompatibilityExpressions,
    FibrousShellKinematicsExpressions,
    director_shell_kinematics,
    fiber_curve_kinematics,
    fibrous_shell_compatibility_ufl,
    fibrous_shell_kinematics_ufl,
    surface_deformation_gradient,
)
from .viscoelasticity import (
    HarmonicViscoelasticStep,
    QuasistaticViscoelasticStep,
    ViscoelasticEnergyFrame,
    ViscoelasticIncrementInfo,
    ViscoelasticPathInfo,
    ViscoelasticQuadratureState,
    harmonic_viscoelastic_step,
    quasistatic_viscoelastic_step,
)

__all__ = [
    "CreepEnergyFrame",
    "CreepIncrementInfo",
    "CreepPathInfo",
    "DirectHarmonicStep",
    "DirectHarmonicSweepStep",
    "DirectorShellKinematics",
    "FiberCurveKinematics",
    "FibrousShellCompatibilityExpressions",
    "FibrousShellKinematicsExpressions",
    "ImplicitCreepStep",
    "ExperimentalFiniteStrainPlasticityStep",
    "FiniteStrainJ2AffineTransaction",
    "FiniteStrainJ2StateTransaction",
    "FiniteStrainJ2StandardProblem",
    "FiniteStrainPlasticityIncrementInfo",
    "FiniteStrainPlasticityPathInfo",
    "J2IncrementInfo",
    "J2LoadPathInfo",
    "J2PlasticityStep",
    "HarmonicViscoelasticStep",
    "ModalAnalysisStep",
    "QuasistaticViscoelasticStep",
    "ViscoelasticEnergyFrame",
    "ViscoelasticIncrementInfo",
    "ViscoelasticPathInfo",
    "ViscoelasticQuadratureState",
    "implicit_creep_step",
    "experimental_finite_strain_j2_step",
    "finite_strain_j2_affine_problem",
    "finite_strain_j2_mixed_affine_problem",
    "finite_strain_j2_standard_problem",
    "direct_harmonic_step",
    "harmonic_frequency_sweep_step",
    "harmonic_viscoelastic_step",
    "j2_plasticity_step",
    "director_shell_kinematics",
    "fiber_curve_kinematics",
    "fibrous_shell_compatibility_ufl",
    "fibrous_shell_kinematics_ufl",
    "surface_deformation_gradient",
    "quasistatic_viscoelastic_step",
]
