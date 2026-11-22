"""Classify a repository change into the minimum trustworthy CI tier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess


LEVELS = ("docs", "targeted", "core", "release")
_RANK = {level: index for index, level in enumerate(LEVELS)}

_DOCUMENTATION_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "AGENT_GUIDE.md",
    "AGENTS.md",
    "WORKFLOW.md",
    "CONCEPTS.md",
    "CITATION.cff",
    "mkdocs.yml",
}
_DOCUMENTATION_PREFIXES = ("docs/", "site/")
_RELEASE_FILES = {
    "pyproject.toml",
    "environment.yml",
    "release_gate.py",
}
_RELEASE_PREFIXES = (
    "packaging/runtime/",
    "src/agentfem/release/",
    ".github/workflows/publish-",
    ".github/workflows/runtime-installers",
    ".github/workflows/platform-acceptance",
)
_TARGETED_SOURCE_FILES = {
    "tools/p92_db_cohort_screen.py",
    "evidence/creep/p92-db-cohort.json",
    "tools/p92_local_evaluation_benchmark.py",
    "evidence/creep/2026-11-07-p92-local-evaluation.json",
    "tools/p92_rate_fatigue_screen.py",
    "evidence/creep/p92-service-aged-fatigue.json",
    "tools/p92_nonuniform_structure.py",
    "evidence/creep/2026-11-07-p92-nonuniform.json",
    "tools/p92_structural_cycle_screen.py",
    "evidence/creep/2026-11-06-p92-structural-cycles.json",
    "evidence/creep/2026-11-06-p92-435cycles-coarse.json",
    "evidence/creep/2026-11-06-p92-435cycles-refined.json",
    "tools/p92_rupture_screen.py",
    "tools/p92_prior_damage_audit.py",
    "tools/p92_creep_equation_reference.py",
    "tools/p92_prior_equation_screen.py",
    "evidence/creep/p92-prior-density-figure.json",
    "evidence/creep/p92-rupture-table.json",
    "tools/p92_cyclic_figure_screen.py",
    "evidence/creep/p92-cyclic-figure-points.json",
    "tools/p92_pure_fatigue_screen.py",
    "evidence/creep/p92-pure-fatigue.json",
    "tools/p92_life_screen.py",
    "evidence/creep/p92-life-table.json",
    "tools/p92_tensor_normalization.py",
    "tools/p92_tensor_material_reference.py",
    "tools/p92_dwell_figure_screen.py",
    "tools/p92_published_recovery_reference.py",
    "evidence/creep/p92-published-figure-points.json",
    "promotion_gate.py",
    "tools/agent_trial_acceptance.py",
    "tools/prepare_agent_trial.py",
    "tools/promote_agent_trial.py",
    "src/agentfem/campaigns.py",
    "src/agentfem/cli.py",
    "src/agentfem/datasets.py",
    "src/agentfem/feedback.py",
    "src/agentfem/manifests.py",
    "src/agentfem/project.py",
    "src/agentfem/surrogates.py",
    "src/agentfem/upgrades.py",
}
_TARGETED_PREFIXES = (
    ".github/",
    "examples/",
    "services/",
    "src/agentfem/campaigns/",
    "src/agentfem/datasets/",
    "src/agentfem/integrations/",
    "src/agentfem/knowledge/",
    "src/agentfem/surrogates/",
)
_SOURCE_TEST_MAP = {
    "ci_validation_scope.py": ("tests/test_ci_validation_scope.py",),
    "tools/p92_db_cohort_screen.py": ("tests/test_p92_db_cohort_screen.py",),
    "evidence/creep/p92-db-cohort.json": ("tests/test_p92_db_cohort_screen.py",),
    "tools/p92_local_evaluation_benchmark.py": ("tests/test_p92_tensor_normalization.py",),
    "evidence/creep/2026-11-07-p92-local-evaluation.json": ("tests/test_p92_tensor_normalization.py",),
    "tools/p92_rate_fatigue_screen.py": ("tests/test_p92_rate_fatigue_screen.py",),
    "evidence/creep/p92-service-aged-fatigue.json": ("tests/test_p92_rate_fatigue_screen.py",),
    "evidence/creep/2026-11-07-p92-nonuniform.json": ("tests/test_p92_nonuniform_structure.py",),
    "tools/p92_nonuniform_structure.py": ("tests/test_p92_nonuniform_structure.py",),
    "tools/p92_structural_cycle_screen.py": ("tests/test_p92_structural_cycle_screen.py",),
    "evidence/creep/2026-11-06-p92-structural-cycles.json": ("tests/test_p92_structural_cycle_screen.py",),
    "evidence/creep/2026-11-06-p92-435cycles-coarse.json": ("tests/test_p92_structural_cycle_screen.py",),
    "evidence/creep/2026-11-06-p92-435cycles-refined.json": ("tests/test_p92_structural_cycle_screen.py",),
    "tools/p92_rupture_screen.py": ("tests/test_p92_rupture_screen.py",),
    "tools/p92_prior_damage_audit.py": ("tests/test_p92_prior_damage_audit.py",),
    "tools/p92_creep_equation_reference.py": ("tests/test_p92_creep_equation_reference.py", "tests/test_p92_prior_equation_screen.py"),
    "tools/p92_prior_equation_screen.py": ("tests/test_p92_prior_equation_screen.py",),
    "evidence/creep/p92-prior-density-figure.json": ("tests/test_p92_prior_equation_screen.py",),
    "evidence/creep/p92-prior-damage.json": ("tests/test_p92_prior_damage_audit.py",),
    "evidence/creep/p92-rupture-table.json": ("tests/test_p92_rupture_screen.py",),
    "tools/p92_cyclic_figure_screen.py": ("tests/test_p92_cyclic_figure_screen.py",),
    "evidence/creep/p92-cyclic-figure-points.json": ("tests/test_p92_cyclic_figure_screen.py",),
    "tools/p92_pure_fatigue_screen.py": ("tests/test_p92_pure_fatigue_screen.py",),
    "evidence/creep/p92-pure-fatigue.json": ("tests/test_p92_pure_fatigue_screen.py",),
    "tools/p92_life_screen.py": ("tests/test_p92_life_screen.py",),
    "evidence/creep/p92-life-table.json": ("tests/test_p92_life_screen.py",),
    "tools/p92_tensor_normalization.py": (
        "tests/test_p92_tensor_normalization.py", "tests/test_p92_tensor_structure.py",
        "tests/test_p92_nonuniform_structure.py",
    ),
    "tools/p92_tensor_material_reference.py": (
        "tests/test_p92_tensor_structure.py", "tests/test_p92_structural_cycle_screen.py",
        "tests/test_p92_nonuniform_structure.py",
    ),
    "tools/p92_dwell_figure_screen.py": ("tests/test_p92_dwell_figure_screen.py",),
    "tools/p92_published_recovery_reference.py": (
        "tests/test_p92_nonuniform_structure.py",
        "tests/test_p92_structural_cycle_screen.py",
        "tests/test_p92_published_recovery_reference.py", "tests/test_p92_dwell_figure_screen.py",
        "tests/test_p92_tensor_normalization.py", "tests/test_p92_tensor_structure.py",
    ),
    "evidence/creep/p92-published-figure-points.json": ("tests/test_p92_dwell_figure_screen.py",),
    "promotion_gate.py": ("tests/test_promotion_gate.py",),
    "tools/agent_trial_acceptance.py": ("tests/test_agent_trial_acceptance.py",),
    "tools/prepare_agent_trial.py": ("tests/test_prepare_agent_trial.py",),
    "tools/promote_agent_trial.py": ("tests/test_agent_trial_promotion.py",),
    "src/agentfem/campaigns/": ("tests/test_campaigns.py",),
    "src/agentfem/datasets/": (
        "tests/test_datasets.py",
        "tests/test_field_datasets.py",
    ),
    "src/agentfem/surrogates/": ("tests/test_surrogates.py",),
    "src/agentfem/feedback.py": ("tests/test_feedback.py",),
    "src/agentfem/community.py": ("tests/test_community.py",),
    "src/agentfem/project.py": ("tests/test_project_cli.py",),
    "src/agentfem/cli.py": (
        "tests/test_project_cli.py",
        "tests/test_common_workflows.py",
    ),
}
# Research-only scalar controls have known transitive consumers. Do not use a
# tools/* wildcard: an unknown numerical driver still needs the broader gate.
_P92_CONTROL_TESTS = (
    "tests/test_p92_recovery_shape.py",
    "tests/test_p92_20261122_state_kinetics_evidence.py",
    "tests/test_p92_critical_recovery.py",
    "tests/test_p92_hold_state_discriminator.py",
    "tests/test_p92_static_modes.py",
    "tests/test_p92_stress_ripple.py",
    "tests/test_p92_observed_recovery_budget.py",
    "tests/test_p92_common_memory.py",
    "tests/test_p92_modulus_probe.py",
    "tests/test_p92_cross_resistance.py",
    "tests/test_p92_recovery_budget_sensitivity.py",
    "tests/test_p92_inviscid_branch.py",
    "tests/test_p92_history_kinetics_split.py",
    "tests/test_p92_common_prefix.py",
    "tests/test_p92_matched_response_report.py",
    "tests/test_p92_autonomous_controls.py",
    "tests/test_p92_hold_flow_budget.py",
    "tests/test_p92_primary_source_audit.py",
    "tests/test_p92_observed_stress_replay.py",
    "tests/test_p92_observation_readiness.py",
    "tests/test_p92_local_feasibility.py",
    "tests/test_p92_recorded_hold_audit.py",
    "tests/test_p92_control_resolution_audit.py",
    "tests/test_p92_recoverable_switch_audit.py",
    "tests/test_p92_recoverable_backstress.py",
    "tests/test_p92_recoverable_mobility.py",
    "tests/test_p92_recovery_capacity.py",
    "tests/test_p92_activated_flow.py",
    "tests/test_p92_fit_subspace_audit.py",
    "tests/test_p92_recovery_bandwidth.py",
    "tests/test_p92_recovery_replacement.py",
    "tests/test_p92_history_memory_audit.py",
    "tests/test_p92_mechanism_evidence.py",
    "tests/test_p92_static_strength.py",
    "tests/test_p92_transient_strength.py",
    "tests/test_p92_subthreshold_probe.py",
    "tests/test_p92_recoverable_kinetics.py",
    "tests/test_p92_primary_creep.py",
    "tests/test_p92_directional_recovery.py",
    "tests/test_p92_residual_conflicts.py",
    "tests/test_p92_static_memory_joint.py",
    "tests/test_p92_static_memory.py",
    "tests/test_p92_resistance_recovery_reference.py",
    "tests/test_p92_resistance_recovery_frozen.py",
    "tests/test_p92_cross_drag_reference.py",
    "tests/test_p92_equal_stress_hold.py",
    "tests/test_p92_coupled_recovery_reference.py",
    "tests/test_p92_shared_drag_reference.py",
    "tests/test_p92_recoverable_spectrum.py",
    "tests/test_p92_reversal_memory_reference.py",
    "tests/test_p92_gate_objective.py",
    "tests/test_p92_recoverable_reference.py",
    "tests/test_p92_segment_budget.py",
    "tests/test_p92_two_rate_joint.py",
    "tests/test_p92_hold_observations.py",
    "tests/test_p92_two_rate_frozen.py",
    "tests/test_p92_two_rate_controls.py",
    "tests/test_p92_two_rate_probe.py",
    "tests/test_p92_two_rate_reference.py",
    "tests/test_p92_hold_mechanisms.py",
    "tests/test_p92_reversal_sensitivity.py",
    "tests/test_p92_response_screen.py",
    "tests/test_p92_waveform_controls.py",
    "tests/test_p92_transient_history.py",
    "tests/test_p92_recovery_discriminator.py",
    "tests/test_p92_candidate_report.py", "tests/test_p92_measured_controls.py",
    "tests/test_p92_bam_source.py", "tests/test_p92_loading_path.py",
    "tests/test_p92_bounded_softening.py", "tests/test_p92_mixed_control.py",
    "tests/test_p92_ratcheting_screen.py", "tests/test_p92_two_mechanism.py",
    "tests/test_p92_two_mechanism_fit.py",
)
_P92_CONTROL_INPUTS = (
    "tools/p92_critical_recovery_reference.py",
    "tools/p92_critical_recovery_probe.py",
    "tools/p92_hold_state_discriminator.py",
    "tools/p92_static_modes_reference.py",
    "tools/p92_static_modes_probe.py",
    "tools/p92_stress_ripple_audit.py",
    "tools/p92_observed_recovery_budget.py",
    "tools/p92_common_memory_reference.py",
    "tools/p92_common_memory_probe.py",
    "tools/p92_modulus_probe.py",
    "tools/p92_cross_resistance_reference.py",
    "tools/p92_cross_resistance_probe.py",
    "tools/p92_recovery_budget_sensitivity.py",
    "tools/p92_inviscid_branch_reference.py",
    "tools/p92_inviscid_branch_probe.py",
    "tools/p92_history_kinetics_split.py",
    "tools/p92_common_prefix.py",
    "tools/p92_matched_response_report.py",
    "tools/p92_autonomous_controls.py",
    "tools/p92_hold_flow_budget.py",
    "tools/p92_primary_source_audit.py",
    "tools/p92_observed_stress_replay.py",
    "tools/p92_observation_readiness.py",
    "evidence/creep/2026-11-22-critical-recovery-matched.json",
    "evidence/creep/2026-11-22-critical-recovery-frozen.json",
    "evidence/creep/2026-11-22-hold-state-discriminator.json",
    "evidence/creep/2026-11-22-static-modes-probe.json",
    "evidence/creep/2026-11-22-recoverable-modes-probe.json",
    "evidence/creep/2026-11-22-static-exchange-probe.json",
    "evidence/creep/2026-11-22-kelvin-time-exchange-probe.json",
    "evidence/creep/2026-11-22-stress-ripple-audit.json",
    "evidence/creep/2026-11-22-mixed-control-audit.json",
    "evidence/creep/2026-11-22-observed-recovery-budget.json",
    "evidence/creep/2026-11-22-common-memory-probe.json",
    "evidence/creep/2026-11-22-slow-common-memory-probe.json",
    "evidence/creep/2026-11-22-modulus-probe.json",
    "evidence/creep/2026-11-22-source-modulus-probe.json",
    "evidence/creep/2026-11-22-cross-resistance-probe.json",
    "evidence/creep/2026-11-22-recovery-budget-sensitivity.json",
    "evidence/creep/2026-11-22-inviscid-branch-probe.json",
    "evidence/creep/2026-11-22-history-kinetics-split.json",
    "evidence/creep/2026-11-22-hold-state-absolute-jacobian.json",
    "evidence/creep/2026-11-22-common-prefix.json",
    "evidence/creep/2026-11-22-recovery-shape-matched.json",
    "evidence/creep/2026-11-22-recovery-shape-frozen.json",
    "evidence/creep/2026-11-22-recovery-shape-nominal-rate.json",
    "evidence/creep/2026-11-22-recovery-shape-control-matched.json",
    "evidence/creep/2026-11-22-recovery-shape-control-frozen.json",
    "evidence/creep/2026-11-22-recovery-shape-control-nominal-rate.json",
    "evidence/creep/2026-11-22-recovery-shape-paired.json",
    "evidence/creep/2026-11-22-signed-hold-flow-budget.json",
    "evidence/creep/2026-11-22-primary-source-audit.json",
    "evidence/creep/2026-11-22-observation-readiness.json",
    "evidence/creep/2026-11-22-observed-stress-replay.json",
    "evidence/creep/2026-11-22-observed-stress-three-cycles.json",
    "evidence/creep/2026-11-22-observed-stress-ten-cycles.json",
    "evidence/creep/2026-11-22-observed-stress-three-cycles-energy-failure.json",
    "evidence/creep/2026-11-22-observed-stress-ten-cycles-energy-failure.json",
    "evidence/creep/2026-11-22-baseline-signed-feasibility.json",
    "evidence/creep/2026-11-22-mobility-signed-feasibility.json",
    "evidence/creep/2026-11-22-seed-recorded-hold.json",
    "evidence/creep/2026-11-22-baseline-recorded-hold.json",
    "evidence/creep/2026-11-22-mobility-recorded-hold.json",
    "evidence/creep/2026-11-22-mobility-derivative-detail.json",
    "tools/p92_recorded_hold_audit.py",
    "tests/test_p92_recorded_hold_audit.py",
    "tools/p92_local_feasibility.py",
    "evidence/creep/2026-11-22-baseline-local-feasibility.json",
    "evidence/creep/2026-11-22-mobility-local-feasibility.json",
    "tests/test_p92_local_feasibility.py",
    "tools/p92_control_resolution_audit.py",
    "tests/test_p92_control_resolution_audit.py",
    "evidence/creep/2026-11-22-control-resolution-audit.json",
    "tools/p92_recoverable_switch_audit.py",
    "tests/test_p92_recoverable_switch_audit.py",
    "evidence/creep/2026-11-22-baseline-switch-audit.json",
    "evidence/creep/2026-11-22-mobility-switch-audit.json",
    "tools/p92_activated_flow_reference.py",
    "tools/p92_activated_flow_probe.py",
    "tests/test_p92_activated_flow.py",
    "evidence/creep/2026-11-22-activated-flow-probe.json",
    "evidence/creep/2026-11-22-control-replay-audit.json",
    "evidence/creep/2026-11-22-static-memory-fit-space.json",
    "evidence/creep/2026-11-22-recoverable-kinetics-fit-space.json",
    "tools/p92_fit_subspace_audit.py",
    "tools/p92_recovery_bandwidth_reference.py",
    "tools/p92_recovery_bandwidth_probe.py",
    "tools/p92_recovery_replacement_reference.py",
    "tools/p92_recovery_replacement_probe.py",
    "tools/p92_history_memory_audit.py",
    "tools/p92_recoverable_backstress_reference.py",
    "tools/p92_recoverable_backstress_probe.py",
    "tools/p92_recoverable_mobility_reference.py",
    "tools/p92_recoverable_mobility_probe.py",
    "tools/p92_recovery_capacity_audit.py",
    "tests/test_p92_recovery_capacity.py",
    "evidence/creep/2026-11-22-recovery-capacity.json",
    "evidence/creep/2026-11-22-static-memory-replacement-matched.json",
    "evidence/creep/2026-11-22-static-memory-replacement-frozen.json",
    "evidence/creep/2026-11-22-static-memory-replacement-residual.json",
    "evidence/creep/2026-11-22-static-memory-replacement-fit-space.json",
    "tests/test_p92_recoverable_mobility.py",
    "evidence/creep/2026-11-22-recoverable-mobility-probe.json",
    "evidence/creep/2026-11-22-recoverable-resistance-probe.json",
    "evidence/creep/2026-11-22-recoverable-mobility-matched.json",
    "evidence/creep/2026-11-22-recoverable-mobility-frozen.json",
    "evidence/creep/2026-11-22-recoverable-mobility-residual.json",
    "evidence/creep/2026-11-22-recoverable-mobility-hold-partition.json",
    "tests/test_p92_recoverable_backstress.py",
    "evidence/creep/2026-11-22-recoverable-backstress-probe.json",
    "evidence/creep/2026-11-22-shared-history-memory-bounds.json",
    "evidence/creep/2026-11-22-baseline-history-memory-bounds.json",
    "evidence/creep/2026-11-22-shared-memory-contributions.json",
    "evidence/creep/2026-11-22-baseline-memory-contributions.json",
    "evidence/creep/2026-11-22-recoverable-kinetics-matched.json",
    "evidence/creep/2026-11-22-recoverable-kinetics-frozen.json",
    "evidence/creep/2026-11-22-recoverable-kinetics-residual.json",
    "evidence/creep/2026-11-22-baseline-hold-partition.json",
    "evidence/creep/2026-11-22-recoverable-kinetics-probe.json",
    "evidence/creep/2026-11-22-static-strength-probe.json",
    "evidence/creep/2026-11-22-transient-strength-probe.json",
    "evidence/creep/2026-11-22-subthreshold-probe.json",
    "evidence/creep/2026-11-22-recovery-bandwidth-probe.json",
    "evidence/creep/2026-11-22-recovery-replacement-probe.json",
    "evidence/creep/2026-11-22-transient-strength-matched.json",
    "evidence/creep/2026-11-22-transient-strength-frozen.json",
    "evidence/creep/2026-11-22-transient-strength-residual.json",
    "evidence/creep/2026-11-22-transient-strength-fit-space.json",
    "tools/p92_static_strength_probe.py",
    "tools/p92_transient_strength_reference.py",
    "tools/p92_transient_strength_probe.py",
    "tools/p92_subthreshold_probe.py",
    "tools/p92_recoverable_kinetics_probe.py",
    "tools/p92_primary_creep_reference.py",
    "evidence/creep/2026-11-22-primary-exp-matched.json",
    "evidence/creep/2026-11-22-primary-exp-frozen.json",
    "evidence/creep/2026-11-22-primary-exp-residual.json",
    "evidence/creep/2026-11-21-static-memory-matched-baseline.json",
    "evidence/creep/2026-11-21-static-memory-matched-shared.json",
    "evidence/creep/2026-11-21-static-memory-matched-baseline-frozen.json",
    "evidence/creep/2026-11-21-static-memory-matched-shared-frozen.json",
    "tools/p92_static_memory_probe.py",
    "evidence/creep/2026-11-21-static-memory-probe.json",
    "evidence/creep/2026-11-21-static-memory-own-frozen.json",
    "evidence/creep/2026-11-21-static-memory-shared-frozen.json",
    "tools/p92_resistance_recovery_reference.py",
    "tools/p92_resistance_recovery_probe.py",
    "tools/p92_resistance_recovery_frozen.py",
    "evidence/creep/2026-11-21-resistance-recovery-probe.json",
    "evidence/creep/2026-11-21-resistance-strength-frozen.json",
    "evidence/creep/2026-11-21-resistance-mobility-frozen.json",
    "tools/p92_cross_drag_reference.py",
    "tools/p92_cross_drag_probe.py",
    "evidence/creep/2026-11-21-cross-drag-probe.json",
    "evidence/creep/2026-11-21-recovery-matched-linear.json",
    "evidence/creep/2026-11-21-recovery-matched-nonlinear.json",
    "evidence/creep/2026-11-21-recovery-matched-linear-frozen.json",
    "evidence/creep/2026-11-21-recovery-matched-nonlinear-frozen.json",
    "evidence/creep/2026-11-21-equal-stress-hold.json",
    "evidence/creep/2026-11-21-spectrum-rate-budget.json",
    "tools/p92_equal_stress_hold.py",
    "tools/p92_spectrum_rate_budget.py",
    "tools/p92_coupled_recovery_reference.py",
    "tools/p92_coupling_probe.py",
    "evidence/creep/2026-11-21-energetic-coupling.json",
    "evidence/creep/2026-11-21-spectrum-extrema-check.json",
    "tools/p92_spectrum_recovery_joint.py",
    "evidence/creep/2026-11-22-directional-matched.json",
    "evidence/creep/2026-11-22-directional-frozen.json",
    "evidence/creep/2026-11-22-directional-residual-conflicts.json",
    "tools/p92_residual_conflicts.py",
    "evidence/creep/2026-11-22-early-residual-conflicts.json",
    "evidence/creep/2026-11-21-spectrum-static-power.json",
    "evidence/creep/2026-11-21-shared-drag-joint.json",
    "evidence/creep/2026-11-21-shared-drag-frozen.json",
    "evidence/creep/2026-11-21-shared-drag-dimensional-start.json",
    "evidence/creep/2026-11-21-shared-drag-dimensional-start-slow.json",
    "evidence/creep/2026-11-21-shared-drag-dimensional-balanced.json",
    "evidence/creep/2026-11-21-shared-drag-published-check.json",
    "tools/p92_shared_drag_reference.py",
    "tools/p92_shared_drag_benchmark.py",
    "tools/p92_shared_drag_probe.py",
    "tools/p92_shared_drag_joint.py",
    "tools/p92_recoverable_spectrum.py",
    "tools/p92_spectrum_probe.py",
    "tools/p92_spectrum_history.py",
    "tools/p92_reversal_memory_reference.py",
    "tools/p92_reversal_memory_probe.py",
    "evidence/creep/2026-11-21-spectrum-reversal-memory.json",
    "evidence/creep/2026-11-21-spectrum-joint.json",
    "evidence/creep/2026-11-21-spectrum-frozen.json",
    "evidence/creep/2026-11-21-spectrum-history.json",
    "evidence/creep/2026-11-21-spectrum-hardening.json",
    "tools/p92_reversal_slope_comparison.py",
    "tools/p92_response_panels.py",
    "evidence/creep/2026-11-21-spectrum-probe-summary.json",
    "evidence/creep/2026-11-21-sensitive-slopes.json",
    "tools/p92_gate_objective.py",
    "tools/p92_gate_report.py",
    "tools/p92_full_sensitivity.py",
    "tools/p92_sensitive_joint.py",
    "evidence/creep/2026-11-21-full-sensitivity.json",
    "evidence/creep/2026-11-21-sensitive-joint.json",
    "evidence/creep/2026-11-21-sensitive-frozen.json",
    "evidence/creep/2026-11-21-gate-backbone.json",
    "evidence/creep/2026-11-21-gate-backbone-report.json",
    "evidence/creep/2026-11-21-gate-recoverable.json",
    "evidence/creep/2026-11-21-gate-recoverable-report.json",
    "evidence/creep/2026-11-21-recoverable-history.json",
    "evidence/creep/2026-11-21-recoverable-strength.json",
    "tools/p92_recoverable_reference.py",
    "tools/p92_recoverable_probe.py",
    "tools/p92_recoverable_joint.py",
    "tools/p92_recoverable_frozen.py",
    "tools/p92_segment_budget.py",
    "evidence/creep/2026-11-21-aligned-segment-budget.json",
    "evidence/creep/2026-11-21-recoverable-probe.json",
    "evidence/creep/2026-11-21-recoverable-joint.json",
    "evidence/creep/2026-11-21-recoverable-frozen.json",
    "evidence/creep/2026-11-21-recoverable-complete-numerics.json",
    "tools/p92_two_rate_joint.py",
    "evidence/creep/2026-11-21-two-rate-joint.json",
    "evidence/creep/2026-11-21-two-rate-joint-frozen.json",
    "evidence/creep/2026-11-21-two-rate-zero-threshold-probe.json",
    "evidence/creep/2026-11-21-two-rate-flow-complete-numerics.json",
    "tools/p92_hold_observations.py",
    "tools/p92_two_rate_frozen.py",
    "evidence/creep/2026-11-21-two-rate-flow-fit.json",
    "evidence/creep/2026-11-21-two-rate-flow-frozen.json",
    "evidence/creep/2026-11-21-two-rate-ratcheting-probe.json",
    "evidence/creep/2026-11-21-recorded-reversal-slopes.json",
    "tools/p92_two_rate_controls.py",
    "tools/p92_two_rate_probe.py",
    "tools/p92_two_rate_sensitivity.py",
    "tools/p92_two_rate_history_probe.py",
    "evidence/creep/2026-11-21-two-rate-unfitted.json",
    "evidence/creep/2026-11-21-two-rate-recovery-fit.json",
    "evidence/creep/2026-11-21-two-rate-history-probe.json",
    "tools/p92_two_rate_reference.py",
    "tools/p92_hold_mechanisms.py",
    "evidence/creep/2026-11-21-hold-mechanisms.json",
    "evidence/creep/2026-11-21-flow-shape-fit.json",
    "evidence/creep/2026-11-21-flow-shape-frozen.json",
    "evidence/creep/2026-11-21-joint-shape-fit.json",
    "evidence/creep/2026-11-21-joint-shape-frozen.json",
    "tools/p92_reversal_sensitivity.py",
    "tools/p92_reversal_identification.py",
    "evidence/creep/2026-11-21-waveform-time.json",
    "evidence/creep/2026-11-21-waveform-kinematic.json",
    "evidence/creep/2026-11-21-reversal-sensitivity.json",
    "evidence/creep/2026-11-21-reversal-fit.json",
    "evidence/creep/2026-11-21-reversal-frozen.json",
    "tools/p92_response_screen.py",
    "tools/p92_waveform_controls.py", "tools/p92_waveform_experiment.py",
    "evidence/creep/2026-11-17-kinematic-shape.json",
    "evidence/creep/2026-11-17-time-shape.json",
    "tools/p92_transient_history.py",
    "evidence/creep/2026-11-17-transient-history.json",
    "evidence/creep/2026-11-17-slow-time-identification.json",
    "evidence/creep/2026-11-17-both-time-identification.json",
    "tools/p92_recovery_discriminator.py",
    "evidence/creep/2026-11-17-recovery-discriminator.json",
    "evidence/creep/2026-11-17-measured-controls-fit.json",
    "tools/p92_measured_controls.py", "tools/p92_measured_fit.py",
    "tools/p92_bounded_softening.py", "tools/p92_mixed_control.py",
    "tools/p92_ratcheting_screen.py", "tools/p92_two_mechanism.py",
    "tools/p92_two_mechanism_screen.py", "tools/p92_two_mechanism_fit.py",
    "evidence/creep/2026-11-17-dual-fit-scaled.json",
    "evidence/creep/2026-11-17-dual-fit-budget.json",
    "evidence/creep/2026-11-17-two-mechanism.json",
    "evidence/creep/2026-11-17-ratcheting-discriminator.json",
)
_TARGETED_SOURCE_FILES.update(_P92_CONTROL_INPUTS)
_SOURCE_TEST_MAP.update(dict.fromkeys(_P92_CONTROL_INPUTS, _P92_CONTROL_TESTS))
_SOURCE_TEST_MAP["tools/p92_published_recovery_reference.py"] += _P92_CONTROL_TESTS

_CORE_SOURCE_TEST_MAP = {
    "src/agentfem/results/journal.py": (
        "tests/test_accepted_history_journal.py",
        "tests/test_transient_history_journal.py",
    ),
    "src/agentfem/_step_response_hex.py": (
        "tests/test_response_material_step.py", "tests/test_response_deformable_step.py",
        "tests/test_owned_explicit_state.py", "tests/test_response_substeps.py",
        "tests/test_contact_overlap_execution.py",
        "tests/test_transient_history_journal.py",
    ),
    "src/agentfem/mechanics/_deformable_contact_explicit.py": (
        "tests/test_contact_overlap_execution.py", "tests/test_deformable_contact_explicit.py",
        "tests/test_response_deformable_step.py",
    ),
    "src/agentfem/constitutive/_owned_explicit_state.py": (
        "tests/test_owned_explicit_state.py", "tests/test_response_material_step.py",
        "tests/test_response_substeps.py",
    ),
    "src/agentfem/elements/explicit_execution.py": (
        "tests/test_explicit_hex_execution.py", "tests/test_owned_explicit_state.py",
        "tests/test_hex_validity_blocks.py",
    ),
    "src/agentfem/elements/_hex_validity.py": (
        "tests/test_hex_validity_blocks.py", "tests/test_hex_validity.py",
        "tests/test_explicit_hex_execution.py",
    ),
    "src/agentfem/_hex_native.c": (
        "tests/test_hex_native.py",
        "tests/test_hex_validity_blocks.py", "tests/test_hex_validity.py",
        "tests/test_explicit_hex_execution.py",
        "tests/test_bulk_viscosity.py", "tests/test_bulk_viscosity_lifecycle.py",
        "tests/test_orientation.py", "tests/test_small_matrix.py",
    ),
    "src/agentfem/_orientation.py": (
        "tests/test_orientation.py", "tests/test_hex_native.py",
    ),
    "src/agentfem/_small_matrix.py": (
        "tests/test_small_matrix.py", "tests/test_hex_native.py",
    ),
    "src/agentfem/boundary_models/q1_execution.py": (
        "tests/test_q1_execution.py", "tests/test_closed_q1_surface.py",
        "tests/test_response_deformable_step.py",
    ),
    "src/agentfem/_q1_surface_search.py": (
        "tests/test_q1_surface_search.py", "tests/test_q1_execution.py",
        "tests/test_closed_q1_surface.py", "tests/test_q1_exact_order.py",
    ),
    "src/agentfem/time/substeps.py": (
        "tests/test_response_substeps.py", "tests/test_owned_explicit_state.py",
    ),
    "src/agentfem/_exact_rational.py": (
        "tests/test_exact_rational.py", "tests/test_q1_exact_order.py",
        "tests/test_q1_ray_exact.py", "tests/test_q1_swept_separation.py",
    ),
    "src/agentfem/assembly.py": ("tests/test_lumped_mass_contract.py",),
    "src/agentfem/_step_builders_contact.py": (
        "tests/test_rigid_obstacle_contact.py",
    ),
    "src/agentfem/_api_contract.py": (
        "tests/test_documentation.py",
        "tests/test_ir.py",
        "tests/test_project_cli.py",
    ),
    "src/agentfem/_model_validation.py": (
        "tests/test_element_contracts.py",
        "tests/test_validation.py",
    ),
    "src/agentfem/_solver_lifecycle.py": (
        "tests/test_prepared_linear_problem.py",
        "tests/test_harmonic.py",
    ),
    "src/agentfem/_hybrid_nonlinear.py": ("tests/test_hybrid_nonlinear.py",),
    "src/agentfem/_nonlinear_problems.py": (
        "tests/test_common_workflows.py",
        "tests/test_p1_platform.py",
    ),
    "src/agentfem/_transient_problems.py": (
        "tests/test_histories.py",
        "tests/test_common_workflows.py",
        "tests/test_first_order_operator_lifecycle.py",
        "tests/test_implicit_dynamics_lifecycle.py",
        "tests/test_time_inputs.py",
        "tests/test_transient_heat_decay_workflow.py",
        "tests/test_transient_restart.py",
        "tests/test_transient_history_journal.py",
    ),
    "src/agentfem/checkpointing.py": ("tests/test_transient_restart.py",),
    "src/agentfem/fatigue_fracture.py": (
        "tests/test_checkpoint_capabilities.py",
        "tests/test_fatigue_fracture.py",
    ),
    "src/agentfem/boundary_models/mechanical.py": (
        "tests/test_parallel_affine.py",
        "tests/test_rigid_obstacle_contact.py",
    ),
    "src/agentfem/boundary_models/dolfinx_adapter.py": (
        "tests/test_dolfinx_boundary_surface_adapter.py",
        "tests/test_rigid_surface_search.py",
    ),
    "src/agentfem/boundary_models/dolfinx_contact_trace.py": (
        "tests/test_dolfinx_contact_trace.py",
        "tests/test_explicit_contact_procedure.py",
        "tests/test_lumped_mass_contract.py",
    ),
    "src/agentfem/boundary_models/dolfinx_explicit_contact.py": (
        "tests/test_dolfinx_explicit_contact.py",
        "tests/test_explicit_contact_procedure.py",
        "tests/test_contact_numeric_state.py",
        "tests/test_finite_hex_contact_composition.py",
        "tests/test_checkpoint_capabilities.py",
    ),
    "src/agentfem/boundary_models/contact_work.py": (
        "tests/test_contact_work_state.py",
        "tests/test_dolfinx_explicit_contact.py",
        "tests/test_finite_hex_contact_composition.py",
    ),
    "src/agentfem/boundary_models/rigid.py": (
        "tests/test_rigid_obstacle_contact.py",
        "tests/test_rigid_surface_contract.py",
        "tests/test_rigid_surface_search.py",
    ),
    "src/agentfem/boundary_models/search.py": (
        "tests/test_rigid_surface_contract.py",
        "tests/test_rigid_surface_search.py",
    ),
    "src/agentfem/boundary_models/__init__.py": (
        "tests/test_dolfinx_boundary_surface_adapter.py",
        "tests/test_engineering_workflows.py",
        "tests/test_rigid_obstacle_contact.py",
        "tests/test_rigid_surface_contract.py",
        "tests/test_rigid_surface_search.py",
    ),
    "src/agentfem/input_effects.py": (
        "tests/test_implicit_dynamics_lifecycle.py",
        "tests/test_time_inputs.py",
        "tests/test_transient_restart.py",
    ),
    "src/agentfem/release/": (
        "tests/test_packaging.py",
        "tests/test_release_goldens.py",
    ),
    "src/agentfem/time/": (
        "tests/test_dynamics.py",
        "tests/test_sprint_numerics.py",
        "tests/test_time_inputs.py",
        "tests/test_transient_restart.py",
    ),
    "src/agentfem/constitutive/user_material.py": ("tests/test_user_material.py",),
    "src/agentfem/constitutive/__init__.py": (
        "tests/test_constitutive_models.py",
        "tests/test_user_material.py",
    ),
    "src/agentfem/constitutive/material_driver.py": (
        "tests/test_user_material.py",
        "tests/test_material_batch_evidence.py",
        "tests/test_material_stress_conversion.py",
        "tests/test_finite_strain_plasticity.py",
        "tests/test_finite_strain_j2_material_map.py",
        "tests/test_history_material_temperature.py",
    ),
    "src/agentfem/mechanics/_prescribed_temperature.py": (
        "tests/test_rate_material_history_units.py",
        "tests/test_history_material_temperature.py",
        "tests/test_small_strain_temperature.py",
        "tests/test_thermal_history_time_units.py",
    ),
    "src/agentfem/histories.py": (
        "tests/test_histories.py",
        "tests/test_thermal_history_time_units.py",
    ),
    "src/agentfem/mechanics/finite_strain_material.py": (
        "tests/test_finite_strain_material_step.py",
        "tests/test_history_plane_stress.py",
        "tests/test_history_material_temperature.py",
        "tests/test_finite_material_point_histories.py",
        "tests/test_thermal_history_time_units.py",
    ),
    "src/agentfem/results/material_points.py": (
        "tests/test_material_point_histories.py",
        "tests/test_finite_material_point_histories.py",
    ),
    "src/agentfem/mechanics/_incremental_runtime.py": (
        "tests/test_incremental_runtime.py",
        "tests/test_small_strain_material_step.py",
        "tests/test_p1_platform.py",
        "tests/test_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/_finite_hex_explicit.py": (
        "tests/test_finite_hex_explicit.py",
        "tests/test_finite_hex_interface.py",
        "tests/test_isotropic_interface_objectivity.py",
        "tests/test_finite_uniform_hex_dolfinx.py",
        "tests/test_finite_hex_energy.py",
        "tests/test_finite_hex_cyclic_energy.py",
        "tests/test_finite_hex_step.py",
    ),
    "src/agentfem/mechanics/_finite_hex_energy.py": (
        "tests/test_finite_hex_energy.py",
        "tests/test_finite_hex_cyclic_energy.py",
        "tests/test_finite_hex_step.py",
    ),
    "src/agentfem/_step_finite_uniform_hex.py": (
        "tests/test_finite_hex_step.py",
        "tests/test_finite_hex_bond_contact.py",
        "tests/test_finite_hex_orthotropic_provider.py",
        "tests/test_finite_hex_contact_composition.py",
        "tests/test_finite_hex_material_envelope_step.py",
        "tests/test_finite_hex_external_material.py",
        "tests/test_finite_hex_cyclic_energy.py",
        "tests/test_transient_restart.py",
    ),
    "src/agentfem/constitutive/stability.py": (
        "tests/test_material_stability_envelope.py",
        "tests/test_finite_hex_material_envelope_step.py",
        "tests/test_finite_hex_stability_envelope.py",
    ),
    "src/agentfem/results/_finite_hex.py": (
        "tests/test_finite_hex_step.py",
        "tests/test_finite_hex_external_material.py",
    ),
    "src/agentfem/mechanics/small_strain_material.py": (
        "tests/test_small_strain_material_step.py",
        "tests/test_small_strain_restore_transaction.py",
        "tests/test_small_strain_physical_time.py",
        "tests/test_small_strain_point_histories.py",
        "tests/test_small_strain_temperature.py",
        "tests/test_thermal_history_time_units.py",
        "tests/test_learned_constitutive.py",
    ),
    "src/agentfem/mechanics/plasticity.py": (
        "tests/test_external_inelastic_benchmark.py",
        "tests/test_p1_platform.py",
        "tests/test_creep_prescribed_trial.py",
    ),
    "src/agentfem/mechanics/creep.py": (
        "tests/test_rate_material_checkpoint_clock.py",
        "tests/test_rate_material_history_units.py",
        "tests/test_external_inelastic_benchmark.py",
        "tests/test_p1_platform.py",
        "tests/test_creep_prescribed_trial.py",
        "tests/test_p92_sinh_screen.py",
    ),
    "src/agentfem/mechanics/viscoelasticity.py": (
        "tests/test_rate_material_checkpoint_clock.py",
        "tests/test_viscoelasticity.py", "tests/test_rate_material_history_units.py",
    ),
    "src/agentfem/mechanics/finite_strain_plasticity.py": (
        "tests/test_finite_strain_j2_mixed.py",
        "tests/test_finite_strain_j2_periodic.py",
        "tests/test_finite_strain_j2_standard.py",
    ),
    "src/agentfem/mechanics/harmonic.py": (
        "tests/test_harmonic.py",
        "tests/test_harmonic_backend_evidence.py",
    ),
    "src/agentfem/events.py": (
        "tests/test_solvers.py",
        "tests/test_results.py",
        "tests/test_harmonic.py",
    ),
    "src/agentfem/elements/": (
        "tests/test_element_contracts.py",
        "tests/test_validation.py",
        "tests/test_hex_validity.py",
        "tests/test_hex_validity_blocks.py",
        "tests/test_uniform_hex.py",
        "tests/test_finite_uniform_hex.py",
        "tests/test_finite_uniform_hex_material.py",
        "tests/test_finite_uniform_hex_dolfinx.py",
        "tests/test_finite_hex_explicit.py",
        "tests/test_uniform_hex_global.py",
        "tests/test_uniform_hex_step.py",
        "tests/test_uniform_hex_work.py",
    ),
    "src/agentfem/elements/_finite_uniform_hex.py": (
        "tests/test_finite_hex_stability_envelope.py",
        "tests/test_finite_hex_material_envelope_step.py",
    ),
    "src/agentfem/_step_uniform_hex.py": (
        "tests/test_uniform_hex_step.py",
        "tests/test_uniform_hex_work.py",
        "tests/test_uniform_hex_global.py",
        "tests/test_transient_restart.py",
    ),
    "src/agentfem/fracture": ("tests/test_fracture_v5.py",),
    "src/agentfem/mesh/": (
        "tests/test_element_contracts.py",
        "tests/test_mesh_formats.py",
        "tests/test_mesh_quality.py",
        "tests/test_mixed_cell_topologies.py",
    ),
    "src/agentfem/operators/identity.py": (
        "tests/test_dynamics.py",
        "tests/test_harmonic.py",
        "tests/test_provenance.py",
    ),
    "src/agentfem/operators/core.py": (
        "tests/test_dynamics.py",
        "tests/test_lumped_mass_contract.py",
        "tests/test_operators.py",
    ),
    "src/agentfem/operators/": (
        "tests/test_operators.py",
        "tests/test_common_workflows.py",
    ),
    "src/agentfem/problems.py": (
        "tests/test_common_workflows.py",
        "tests/test_results.py",
    ),
    "src/agentfem/_execution_environment.py": (
        "tests/test_execution_environment.py",
        "tests/test_project_execution.py",
    ),
    "src/agentfem/results/accepted_export.py": (
        "tests/test_project_execution.py",
        "tests/test_xdmf_layout.py",
        "tests/test_xdmf_cell_ordering.py",
    ),
    "src/agentfem/results": ("tests/test_results.py",),
    "src/agentfem/solvers.py": (
        "tests/test_solvers.py",
        "tests/test_prepared_linear_problem.py",
        "tests/test_common_workflows.py",
        "tests/test_affine_equilibrium_reuse.py",
        "tests/test_affine_normal_augmentation_research.py",
        "tests/test_affine_newton_policy.py",
    ),
    "src/agentfem/_affine_equilibrium.py": (
        "tests/test_affine_equilibrium_reuse.py",
        "tests/test_affine_j2_automatic_restart.py",
        "tests/test_affine_normal_augmentation_research.py",
        "tests/test_affine_newton_policy.py",
    ),
    "src/agentfem/mechanics/_affine_normal_augmentation.py": (
        "tests/test_affine_normal_augmentation_research.py",
        "tests/test_periodic_contact_fixture.py",
    ),
    "examples/finite_j2_periodic_augmented_plane.py": (
        "tests/test_periodic_contact_fixture.py",
    ),
    "src/agentfem/boundary_models/_normal_work_path.py": (
        "tests/test_affine_normal_augmentation_research.py",
        "tests/test_fixed_normal_work.py",
    ),
    "src/agentfem/boundary_models/_augmented_plane_trial_provider.py": (
        "tests/test_affine_normal_augmentation_research.py",
        "tests/test_finite_j2_normal_augmentation.py",
        "tests/test_translating_normal_augmentation.py",
    ),
    "src/agentfem/step_providers.py": (
        "tests/test_common_workflows.py",
        "tests/test_validation.py",
    ),
    "src/agentfem/_architecture_contract.py": ("tests/test_architecture_contract.py",),
}
_CORE_SOURCE_MPI_TEST_MAP = {
    "src/agentfem/mechanics/_affine_normal_augmentation.py": (
        "tests/test_finite_strain_j2_contact_mpi_boundary.py",
        "tests/test_parallel_affine.py",
    ),
    "src/agentfem/boundary_models/_augmented_plane_trial_provider.py": (
        "tests/test_finite_strain_j2_contact_mpi_boundary.py",
    ),
    "src/agentfem/_nonmatching": ("tests/test_parallel_nonmatching.py",),
    "src/agentfem/_interface_boundary.py": ("tests/test_parallel_nonmatching.py",),
    "src/agentfem/_interface_pairing.py": ("tests/test_parallel_nonmatching.py",),
    "src/agentfem/_elastic_cohesive.py": ("tests/test_parallel_nonmatching.py",),
    "src/agentfem/assembly.py": ("tests/test_lumped_mass_contract.py",),
    "src/agentfem/_step_builders_contact.py": (
        "tests/test_rigid_obstacle_contact.py",
    ),
    "src/agentfem/_model_validation.py": ("tests/test_element_contracts.py",),
    "src/agentfem/_hybrid_nonlinear.py": ("tests/test_parallel_mixed.py",),
    "src/agentfem/_nonlinear_problems.py": (
        "tests/test_parallel_affine.py",
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/_transient_problems.py": (
        "tests/test_first_order_operator_lifecycle.py",
        "tests/test_parallel_transient.py",
    ),
    "src/agentfem/checkpointing.py": ("tests/test_parallel_transient.py",),
    "src/agentfem/fatigue_fracture.py": ("tests/test_parallel_cohesive.py",),
    "src/agentfem/boundary_models/mechanical.py": (
        "tests/test_rigid_obstacle_contact.py",
    ),
    "src/agentfem/boundary_models/dolfinx_adapter.py": (
        "tests/test_dolfinx_boundary_surface_adapter.py",
        "tests/test_parallel_rigid_surface_search.py",
    ),
    "src/agentfem/boundary_models/dolfinx_contact_trace.py": (
        "tests/test_dolfinx_contact_trace.py",
        "tests/test_explicit_contact_procedure.py",
        "tests/test_lumped_mass_contract.py",
    ),
    "src/agentfem/boundary_models/dolfinx_explicit_contact.py": (
        "tests/test_dolfinx_explicit_contact.py",
        "tests/test_explicit_contact_procedure.py",
        "tests/test_parallel_finite_hex_contact.py",
    ),
    "src/agentfem/boundary_models/contact_work.py": (
        "tests/test_dolfinx_explicit_contact.py",
    ),
    "src/agentfem/boundary_models/rigid.py": (
        "tests/test_rigid_obstacle_contact.py",
    ),
    "src/agentfem/boundary_models/search.py": (
        "tests/test_parallel_rigid_surface_search.py",
    ),
    "src/agentfem/boundary_models/__init__.py": (
        "tests/test_dolfinx_boundary_surface_adapter.py",
        "tests/test_parallel_rigid_surface_search.py",
        "tests/test_rigid_obstacle_contact.py",
    ),
    "src/agentfem/input_effects.py": ("tests/test_parallel_transient.py",),
    "src/agentfem/time/": ("tests/test_parallel_transient.py",),
    "src/agentfem/mechanics/_finite_hex_explicit.py": (
        "tests/test_parallel_finite_hex.py",
        "tests/test_parallel_finite_hex_step.py",
        "tests/test_parallel_nonmatching.py",
    ),
    "src/agentfem/mechanics/_finite_hex_energy.py": (
        "tests/test_parallel_finite_hex.py",
        "tests/test_parallel_finite_hex_step.py",
    ),
    "src/agentfem/_step_finite_uniform_hex.py": (
        "tests/test_parallel_finite_hex_step.py",
        "tests/test_parallel_finite_hex_contact.py",
        "tests/test_parallel_nonmatching.py",
    ),
    "src/agentfem/constitutive/stability.py": (
        "tests/test_parallel_finite_hex_step.py",
    ),
    "src/agentfem/results/_finite_hex.py": (
        "tests/test_parallel_finite_hex_step.py",
    ),
    "src/agentfem/constitutive/material_driver.py": (
        "tests/test_parallel_inelastic.py",
        "tests/test_material_batch_evidence.py",
    ),
    "src/agentfem/mechanics/_incremental_runtime.py": (
        "tests/test_parallel_inelastic.py",
        "tests/test_parallel_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/small_strain_material.py": (
        "tests/test_parallel_learning.py",
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/mechanics/plasticity.py": ("tests/test_parallel_inelastic.py",),
    "src/agentfem/mechanics/creep.py": ("tests/test_parallel_inelastic.py",),
    "src/agentfem/mechanics/viscoelasticity.py": (
        "tests/test_parallel_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/finite_strain_plasticity.py": (
        "tests/test_parallel_inelastic.py",
        "tests/test_parallel_mixed.py",
    ),
    "src/agentfem/mechanics/harmonic.py": ("tests/test_parallel_viscoelasticity.py",),
    "src/agentfem/elements/": (
        "tests/test_element_contracts.py",
        "tests/test_parallel_uniform_hex.py",
        "tests/test_parallel_finite_hex.py",
    ),
    "src/agentfem/_step_uniform_hex.py": ("tests/test_parallel_uniform_hex.py",),
    "src/agentfem/fracture": ("tests/test_parallel_cohesive.py",),
    "src/agentfem/mesh/": (
        "tests/test_element_contracts.py",
        "tests/test_mixed_cell_topologies.py",
    ),
    "src/agentfem/operators/identity.py": ("tests/test_parallel_operator_identity.py",),
    "src/agentfem/operators/core.py": (
        "tests/test_lumped_mass_contract.py",
        "tests/test_parallel_operator_identity.py",
    ),
    "src/agentfem/solvers.py": (
        "tests/test_parallel_affine.py",
        "tests/test_parallel_results.py",
    ),
    "src/agentfem/problems.py": (
        "tests/test_parallel_results.py",
        "tests/test_parallel_transient.py",
    ),
    "src/agentfem/results": ("tests/test_parallel_results.py",),
}
_CORE_SOURCE_MPI_DRIVER_MAP = {
    "src/agentfem/mechanics/viscoelasticity.py": ("viscoelastic-restart",),
    "tests/portable_viscoelastic_step_driver.py": ("viscoelastic-restart",),
    "src/agentfem/_nonlinear_problems.py": ("nonlinear-contact-restart",),
    "src/agentfem/_step_builders_contact.py": ("nonlinear-contact-restart",),
    "src/agentfem/boundary_models/mechanical.py": (
        "nonlinear-contact-restart",
    ),
    "src/agentfem/boundary_models/rigid.py": (
        "nonlinear-contact-restart",
    ),
    "src/agentfem/boundary_models/__init__.py": (
        "nonlinear-contact-restart",
    ),
    "src/agentfem/checkpointing.py": (
        "cyclic-field-restart",
        "global-cyclic-restart",
        "nonlinear-contact-restart",
    ),
    "src/agentfem/fatigue_fracture.py": (
        "cyclic-field-restart",
        "global-cyclic-restart",
    ),
    "src/agentfem/mechanics/finite_strain_plasticity.py": (
        "mixed-finite-strain-j2-restart",
    ),
    "src/agentfem/results/_accepted_energy.py": (
        "nonlinear-contact-restart",
    ),
    "src/agentfem/results/_nonlinear_step.py": (
        "nonlinear-contact-restart",
    ),
    "tests/portable_nonlinear_contact_driver.py": (
        "nonlinear-contact-restart",
    ),
    "tests/portable_cyclic_field_driver.py": ("cyclic-field-restart",),
    "tests/portable_global_cyclic_driver.py": ("global-cyclic-restart",),
    "tests/portable_mixed_finite_strain_j2_driver.py": (
        "mixed-finite-strain-j2-restart",
    ),
}
_ML_PREFIXES = (
    "src/agentfem/learning",
    "src/agentfem/datasets",
    "src/agentfem/surrogates",
    "tests/test_datasets.py",
    "tests/test_surrogates.py",
)


@dataclass(frozen=True)
class ValidationScope:
    level: str
    tests: tuple[str, ...] = ()
    mpi_tests: tuple[str, ...] = ()
    mpi_drivers: tuple[str, ...] = ()
    ml: bool = False

    @property
    def numerical(self) -> bool:
        return self.level != "docs"

    def outputs(self) -> dict[str, str]:
        return {
            "level": self.level,
            "numerical": str(self.numerical).lower(),
            "ml": str(self.ml).lower(),
            "tests": " ".join(self.tests),
            "mpi_tests": " ".join(self.mpi_tests),
            "mpi_drivers": " ".join(self.mpi_drivers),
        }


def _at_least(current: str, candidate: str) -> str:
    return candidate if _RANK[candidate] > _RANK[current] else current


def _path_level(path: str) -> str:
    if path in _DOCUMENTATION_FILES or path.startswith(_DOCUMENTATION_PREFIXES):
        return "docs"
    if (
        path.startswith("src/agentfem/release/")
        and ".dev" in Path(path).name
        and path.endswith(".json")
    ):
        return "core"
    if path in _RELEASE_FILES or path.startswith(_RELEASE_PREFIXES):
        return "release"
    if path.startswith("tests/"):
        filename = path.removeprefix("tests/")
        is_direct_test = (
            "/" not in filename
            and filename.startswith("test_")
            and filename.endswith(".py")
        )
        if is_direct_test and not filename.startswith("test_parallel_"):
            return "targeted"
        # Drivers, fixtures, Goldens and explicitly distributed modules do not
        # provide useful evidence when selected as a serial pytest path.
        return "core"
    if path in _TARGETED_SOURCE_FILES or path.startswith(_TARGETED_PREFIXES):
        return "targeted"
    if path.startswith("src/agentfem/"):
        return "core"
    if path in {"build_docs.py", "build_knowledge.py", "ci_validation_scope.py"}:
        return "targeted"
    # Unknown build or repository files fail safe instead of silently reducing
    # scientific validation.
    return "core"


def classify_changes(
    paths: list[str] | tuple[str, ...],
    *,
    requested: str = "auto",
) -> ValidationScope:
    """Return the minimum CI tier that covers ``paths``.

    Manual requests are authoritative.  Automatic classification is
    conservative for unknown paths, while test-only corrections remain in the
    targeted tier and therefore do not replay the complete MPI evidence chain.
    """

    if requested != "auto":
        if requested not in LEVELS:
            raise ValueError(f"Unsupported validation level: {requested!r}.")
        return ValidationScope(
            requested,
            ml=requested == "release",
        )

    normalized = tuple(
        dict.fromkeys(str(Path(path)).replace("\\", "/") for path in paths if path)
    )
    if not normalized:
        return ValidationScope("core")

    level = "docs"
    for path in normalized:
        level = _at_least(level, _path_level(path))
    selected_tests = {
        path
        for path in normalized
        if path.startswith("tests/test_") and path.endswith(".py")
    }
    mpi_tests: set[str] = set()
    mpi_drivers: set[str] = set()
    unmapped_core_source = False
    if level in {"targeted", "core"}:
        for path in normalized:
            for prefix, mapped_tests in _SOURCE_TEST_MAP.items():
                if path == prefix or path.startswith(prefix):
                    selected_tests.update(mapped_tests)
            for prefix, mapped_tests in _CORE_SOURCE_TEST_MAP.items():
                if path == prefix or path.startswith(prefix):
                    selected_tests.update(mapped_tests)
            for prefix, mapped_tests in _CORE_SOURCE_MPI_TEST_MAP.items():
                if path == prefix or path.startswith(prefix):
                    mpi_tests.update(mapped_tests)
            for prefix, mapped_drivers in _CORE_SOURCE_MPI_DRIVER_MAP.items():
                if path == prefix or path.startswith(prefix):
                    mpi_drivers.update(mapped_drivers)
            if (
                level == "core"
                and path.startswith("src/agentfem/")
                and _path_level(path) == "core"
                and not any(
                    path == prefix or path.startswith(prefix)
                    for prefix in _CORE_SOURCE_TEST_MAP
                )
            ):
                unmapped_core_source = True
    if unmapped_core_source:
        # An unknown core owner cannot be covered by an invented partial set;
        # fail safe to the complete release evidence ladder.
        level = "release"
        selected_tests.clear()
        mpi_tests.clear()
        mpi_drivers.clear()
    tests = tuple(sorted(selected_tests))
    ml = level == "release" or any(path.startswith(_ML_PREFIXES) for path in normalized)
    return ValidationScope(
        level,
        tests=tests,
        mpi_tests=tuple(sorted(mpi_tests)),
        mpi_drivers=tuple(sorted(mpi_drivers)),
        ml=ml,
    )


def changed_paths(base: str, head: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--requested", default="auto")
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()

    if arguments.requested == "auto" and (not arguments.base or not arguments.head):
        scope = ValidationScope("core")
    else:
        paths = (
            changed_paths(arguments.base, arguments.head)
            if arguments.base and arguments.head
            else []
        )
        scope = classify_changes(paths, requested=arguments.requested)

    outputs = scope.outputs()
    text = "\n".join(f"{key}={value}" for key, value in outputs.items()) + "\n"
    if arguments.github_output is None:
        print(text, end="")
    else:
        with arguments.github_output.open("a", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
