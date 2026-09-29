from pathlib import Path

import pytest

from agentfem import checkpointing
from agentfem._nonlinear_problems import (
    AffineNonlinearVariationalProblem,
    IncrementalNonlinearVariationalProblem,
)
from agentfem._transient_problems import (
    ExplicitDynamicsStep,
    FirstOrderTransientStep,
    ImplicitDynamicsStep,
)
from agentfem.fatigue_fracture import FieldStateTransaction, GlobalCyclicFatigueStep
from agentfem.mechanics.creep import ImplicitCreepStep
from agentfem.mechanics.finite_strain_plasticity import (
    FiniteStrainJ2StandardProblem,
)
from agentfem.mechanics.harmonic import DirectHarmonicSweepStep
from agentfem.mechanics.plasticity import J2PlasticityStep
from agentfem.mechanics.small_strain_material import SmallStrainMaterialStep
from agentfem.mechanics.viscoelasticity import QuasistaticViscoelasticStep


def _capabilities(step_type):
    return step_type.checkpoint_capabilities(object.__new__(step_type))


@pytest.mark.parametrize(
    "step_type",
    (
        ExplicitDynamicsStep,
        ImplicitDynamicsStep,
        FirstOrderTransientStep,
        IncrementalNonlinearVariationalProblem,
        AffineNonlinearVariationalProblem,
        J2PlasticityStep,
        ImplicitCreepStep,
        QuasistaticViscoelasticStep,
        SmallStrainMaterialStep,
        FiniteStrainJ2StandardProblem,
        DirectHarmonicSweepStep,
        FieldStateTransaction,
        GlobalCyclicFatigueStep,
    ),
)
def test_checkpoint_owners_declare_typed_capabilities(step_type):
    capabilities = _capabilities(step_type)

    assert isinstance(capabilities, checkpointing.CheckpointCapabilities)
    assert capabilities.schemas
    assert capabilities.state_components
    assert capabilities.identity_scope


def test_payload_scope_is_independent_of_rank_count_portability():
    harmonic = _capabilities(DirectHarmonicSweepStep)
    transient = _capabilities(ExplicitDynamicsStep)

    assert harmonic.rank_count_portability == "supported"
    assert harmonic.payload_scope == "progress_ledger"
    assert harmonic.full_restart is False
    assert transient.payload_scope == "full_restart_state"
    assert transient.rank_count_portability == "requires_portable_policy"
    assert transient.summary()["effective_rank_count_portable"] is None


def test_policy_summary_reports_effective_rank_count_claim(tmp_path):
    capabilities = _capabilities(ExplicitDynamicsStep)
    local = checkpointing.every(3, directory=tmp_path, portable=False)
    portable = checkpointing.every(3, directory=tmp_path, portable=True)

    assert capabilities.summary(policy=local)["effective_rank_count_portable"] is False
    assert (
        capabilities.summary(policy=portable)["effective_rank_count_portable"]
        is True
    )


def test_cycle_checkpoint_declares_cross_rank_count_portability(tmp_path):
    capabilities = _capabilities(GlobalCyclicFatigueStep)
    policy = checkpointing.CheckpointPolicy(
        every=1,
        directory=Path(tmp_path),
        portable=True,
    )

    capabilities.validate_policy(policy)
    assert capabilities.full_restart is True
    assert capabilities.summary(policy=policy)["effective_rank_count_portable"] is True


def test_capabilities_of_requires_an_explicit_declaration():
    class MissingDeclaration:
        pass

    with pytest.raises(TypeError, match="does not declare checkpoint capabilities"):
        checkpointing.capabilities_of(MissingDeclaration())


def test_capability_contract_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="payload scope"):
        checkpointing.CheckpointCapabilities(
            schemas=("example.v1",),
            boundary="accepted_step",
            payload_scope="portable",
            state_components=("state",),
            atomic_publication=True,
            rank_count_portability="supported",
            identity_scope=("model",),
        )


def test_partition_bound_contract_rejects_portable_policy(tmp_path):
    capabilities = checkpointing.CheckpointCapabilities(
        schemas=("example.v1",),
        boundary="manual_state",
        payload_scope="field_state",
        state_components=("field",),
        atomic_publication=True,
        rank_count_portability="unsupported",
        identity_scope=("partition layout",),
    )
    policy = checkpointing.every(1, directory=tmp_path, portable=True)

    with pytest.raises(ValueError, match="original MPI rank count and partition"):
        capabilities.validate_policy(policy)
