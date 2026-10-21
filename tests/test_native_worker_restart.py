# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Worker tuning is operational only for declared partition-invariant kernels."""
from contextlib import ExitStack

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import elements, fields, models, studies
from test_strain_history_runtime import manifest, runtime


def build(rt, storage="owned_arrays"):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 3, 2, 1, cell_type=mesh.CellType.hexahedron)
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(rt.material)
    step = model.step(target=u, dt=1e-4, steps=4, time_increment_ceiling=1e-3,
        stress_free_initial_state=True, progress=False, material_storage=storage,
        execution_kernel=rt.kernel, element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=1., hourglass_scale=.1, kinematics="finite_strain"),
        bulk_viscosity=elements.BulkViscosityPolicy(2., 2., .5))
    xyz = u.value.function_space.tabulate_dof_coordinates()
    step.state.v.value.x.array[:] = (.001 * xyz).ravel()
    return step


@pytest.mark.parametrize("storage", ["fields", "owned_arrays"])
@pytest.mark.parametrize("old,new", [
    (dict(workers=1, cell_workers=1), dict(workers=3, cell_workers=2, tile_size=2)),
    (dict(workers=3, cell_workers=2, tile_size=2), dict(workers=1, cell_workers=1)),
])
def test_native_worker_change_preserves_restart_and_actual_provenance(tmp_path, storage, old, new):
    path = manifest(tmp_path)
    with ExitStack() as owners:
        a = owners.enter_context(runtime(path, **old))
        b = owners.enter_context(runtime(path, **new))
        assert a.kernel.summary() != b.kernel.summary()
        assert a.kernel.checkpoint_summary() == b.kernel.checkpoint_summary()
        reference = build(a, storage)
        reference.run()
        partial = build(a, storage)
        partial.run(until_step=2)
        checkpoint = partial.save_checkpoint(tmp_path / "accepted")
        resumed = build(b, storage)
        assert resumed.material_residual.identity != partial.material_residual.identity
        resumed.load_checkpoint(checkpoint)
        resumed.run()
        for field in ("u", "v", "a"):
            np.testing.assert_array_equal(getattr(resumed.state, field).value.x.array,
                                          getattr(reference.state, field).value.x.array)
        actual = resumed.material_residual.snapshot()
        expected = reference.material_residual.snapshot()
        assert actual.pop("identity") != expected.pop("identity")
        assert actual == expected


def test_legacy_native_checkpoint_remains_exact_and_can_be_resaved(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    with ExitStack() as owners:
        a = owners.enter_context(runtime(path, workers=1))
        b = owners.enter_context(runtime(path, workers=3, cell_workers=2))
        partial = build(a)
        partial.run(until_step=2)
        legacy_record = partial.material_residual.snapshot()
        legacy_record["schema"] = "agentfem.private-response-hex.v4"
        legacy_record.pop("restart_identity")
        with monkeypatch.context() as patch:
            patch.setattr(partial.material_residual, "checkpoint_snapshot", lambda: legacy_record)
            checkpoint = partial.save_checkpoint(tmp_path / "legacy")
        different = build(b)
        with pytest.raises(ValueError, match="identity mismatch"):
            different.load_checkpoint(checkpoint)
        assert different.completed_steps == 0
        np.testing.assert_array_equal(different.state.u.value.x.array, 0.)
        original = build(a)
        original.load_checkpoint(checkpoint)
        upgraded = original.save_checkpoint(tmp_path / "upgraded")
        different.load_checkpoint(upgraded)
        different.run()
        original.run()
        np.testing.assert_array_equal(different.state.u.value.x.array, original.state.u.value.x.array)
        for name in ("state", "cauchy_stress"):
            np.testing.assert_array_equal(different.accepted_material_fields()[name],
                                          original.accepted_material_fields()[name])


def test_compatibility_declaration_is_optional_frozen_and_keeps_numerical_identity():
    from agentfem.elements import ExplicitHexBodyKernel
    declaration = {"implementation": {"revision": 1}}
    kernel = ExplicitHexBodyKernel(lambda *a: None, {"provider": "custom", "workers": 1},
                                   restart_identity=declaration)
    before = kernel.checkpoint_summary()
    declaration["implementation"]["revision"] = 2
    assert kernel.checkpoint_summary() == before
    assert ExplicitHexBodyKernel(lambda *a: None, {"provider": "custom"}).checkpoint_summary() is None
    with pytest.raises(TypeError, match="restart_identity"):
        ExplicitHexBodyKernel(lambda *a: None, {"provider": "custom"}, restart_identity={})


@pytest.mark.parametrize("changes", [{"maximum_stretch_increment": .02},
                                    {"property_names": ["mu", "lambda"]}])
def test_native_restart_rejects_changed_scientific_identity_before_acceptance(tmp_path, changes):
    path = manifest(tmp_path)
    with runtime(path) as original:
        partial = build(original)
        partial.run(until_step=2)
        checkpoint = partial.save_checkpoint(tmp_path / "accepted")
    # Numerical increment policy or material parameter encoding is not tuning.
    changed_path = manifest(tmp_path, **changes)
    with runtime(changed_path, workers=3) as changed:
        target = build(changed)
        with pytest.raises(ValueError, match="identity mismatch"):
            target.load_checkpoint(checkpoint)
        assert target.completed_steps == 0
        np.testing.assert_array_equal(target.state.u.value.x.array, 0.)


@pytest.mark.parametrize("change", [{"schema": "unknown"}, {"restart_identity": "0" * 64},
                                   {"identity": "not-a-digest"}])
def test_invalid_compatibility_record_does_not_mutate_state(tmp_path, change):
    with runtime(manifest(tmp_path)) as rt:
        step = build(rt)
        before = step.material_residual.snapshot()
        invalid = {**before, **change}
        with pytest.raises(ValueError, match="identity mismatch"):
            step.material_residual.restore(invalid)
        assert step.material_residual.snapshot() == before


def test_builtin_cell_workers_resume_but_arithmetic_backend_change_rejects(tmp_path):
    pytest.importorskip("agentfem._body_pipeline")
    from test_response_material_step import contact_step
    options = {"bulk_viscosity": elements.BulkViscosityPolicy(1., 10., .5)}
    original = contact_step(execution_kernel=elements.explicit_hex_kernel(
        backend="native", cell_workers=1), **options)
    original.run(until_step=2)
    checkpoint = original.save_checkpoint(tmp_path / "cell_workers")
    tuned = contact_step(execution_kernel=elements.explicit_hex_kernel(
        backend="native", cell_workers=3), **options)
    tuned.load_checkpoint(checkpoint)
    original.run()
    tuned.run()
    np.testing.assert_array_equal(tuned.state.u.value.x.array, original.state.u.value.x.array)
    assert tuned.material_residual.viscosity_work == original.material_residual.viscosity_work
    different = contact_step(execution_kernel=elements.explicit_hex_kernel(backend="numpy"), **options)
    with pytest.raises(ValueError, match="identity mismatch"):
        different.load_checkpoint(checkpoint)
    assert different.completed_steps == 0
