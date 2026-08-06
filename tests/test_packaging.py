from agentfem import release_gate


def test_release_gate_rejects_build_machine_bytecode_and_cache_members():
    members = {
        "agentfem/models.py",
        "agentfem/__pycache__/models.cpython-311.pyc",
        "agentfem/old.pyo",
    }

    assert release_gate._forbidden_distribution_members(members) == [
        "agentfem/__pycache__/models.cpython-311.pyc",
        "agentfem/old.pyo",
    ]
