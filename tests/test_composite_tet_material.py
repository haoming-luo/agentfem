# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic constitutive bridge: native J2, committed history and mesh dynamics."""

import numpy as np
import pytest
from mpi4py import MPI
from agentfem import constitutive, fields, models, studies, elements, mesh
from agentfem.elements.composite_tet import composite_tet10, composite_tet11, _EDGES
from agentfem.elements._composite_material import CompositeMaterialBatch


def material(yield_stress=0.1):
    return constitutive.explicit_small_strain_j2(
        constitutive.J2LinearIsotropicHardening(
            young=100, poisson=0.25, yield_stress=yield_stress, hardening_modulus=2
        ),
        density=1,
    )


def local_element():
    x = np.vstack((np.zeros(3), np.eye(3)))
    return composite_tet10(np.vstack((x, x[_EDGES].mean(1))))


def test_independent_center_rigid_modes_and_tying():
    e = local_element()
    e11 = composite_tet11(e.coordinates)
    C = material().elastic_envelope
    T = np.zeros((33, 30))
    T[:30] = np.eye(30)
    for i in range(3):
        T[30 + i, 12 + i : 30 : 3] = 1 / 6
    for mean in (False, True):
        K = e11.stiffness(C, mean_dilatation=mean)
        vals = np.linalg.eigvalsh(K)
        assert np.count_nonzero(vals < vals[-1] * 1e-10) == 6
        np.testing.assert_allclose(
            T.T @ K @ T,
            e.stiffness(C, mean_dilatation=mean, stabilize=True),
            atol=2e-13,
        )
    np.testing.assert_allclose(
        T.T @ e11.mass(2, lumped=False) @ T, e.mass(2, lumped=False), atol=1e-15
    )
    assert np.all(e11.mass(2) > 0)
    # Projection alone cannot stabilize the released centre.
    vals = np.linalg.eigvalsh(e11.stiffness(C, stabilize=False))
    assert np.count_nonzero(vals < vals[-1] * 1e-10) == 9


def test_j2_force_and_trial_commit_restart():
    e = local_element()
    m = material()
    b = CompositeMaterialBatch([e], m)
    strain = np.diag([0.02, -0.003, 0.001])
    u = (e.coordinates @ strain.T).reshape(1, 30)
    f = b.evaluate(u, 0.1)
    state = b.trial[1].copy()
    np.testing.assert_allclose(b.evaluate(u, 0.1), f)
    np.testing.assert_allclose(b.trial[1], state)
    assert np.all(b.states == 0)
    b.commit()
    assert b.states[:, 9].min() > 0
    snap = b.snapshot()
    native = m.provider.material.update(strain)
    np.testing.assert_allclose(
        b.stress, np.broadcast_to(native.stress, b.stress.shape), atol=1e-13
    )
    b.evaluate(-u, 0.2)
    b.rollback()
    assert b.snapshot() == snap
    other = CompositeMaterialBatch([e], m)
    other.restore(snap)
    np.testing.assert_allclose(b.evaluate(-u, 0.2), other.evaluate(-u, 0.2))
    assert np.all(b.trial[4] >= b.dissipation)


def build(comm=MPI.COMM_SELF, yield_stress=0.1, independent=False, mean=False):
    d = mesh.cuboid((0, 0, 0), (1, 1, 1), (1, 1, 1), comm=comm, cell_type="tetrahedron")
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=d
    )
    u = m.field(
        fields.displacement(
            d,
            element_policy=elements.composite_tet10_policy(
                independent_center=independent, mean_dilatation=mean
            ),
        )
    )
    m.material(material(yield_stress))
    m.clamp(u, on=mesh.boundary_region(d, lambda x: np.isclose(x[0], 0)))
    m.nodal_force((10, 0, 0), at=(1, 1, 1))
    return m.step(target=u, dt=0.001, steps=15, progress=False, save_every=1)


@pytest.mark.parametrize(
    "independent,mean", [(False, False), (True, False), (True, True), (False, True)]
)
def test_j2_step_restart_and_output(tmp_path, independent, mean):
    import h5py

    s = build(independent=independent, mean=mean)
    result = s.solve_result(output=tmp_path / "j2.xdmf", field_variables=("S", "SENER"))
    assert s.residual.internal.material_batch.states[:, 9].max() > 0
    np.testing.assert_allclose(
        result.fields["STATE_PEEQ"].field.ravel(),
        s.residual.internal.material_batch.states[:, 9],
    )
    a = build(independent=independent, mean=mean)
    a.run(until_step=7)
    a.save_checkpoint(tmp_path / "checkpoint")
    b = build(independent=independent, mean=mean)
    b.load_checkpoint(tmp_path / "checkpoint")
    b.solve_result()
    np.testing.assert_allclose(s.state.u.x.array, b.state.u.x.array, atol=1e-14)
    assert (
        s.residual.internal.material_batch.snapshot()
        == b.residual.internal.material_batch.snapshot()
    )
    with h5py.File(tmp_path / "j2.h5") as h:
        batch = s.residual.internal.material_batch
        expected = np.einsum(
            "cq,cqj->cj",
            batch.weights / batch.volume[:, None],
            batch.stress.reshape(-1, batch.points_per_cell, 9),
        )
        np.testing.assert_allclose(
            h["Frames/0015/Cell/S"][:], np.repeat(expected, 12, axis=0)
        )
    assert (
        s.history_records[-1]["accounted_internal_kinetic_energy"]
        >= s.history_records[-1]["total_mechanical_energy"]
    )


@pytest.mark.parametrize(
    "independent,mean", [(False, False), (True, False), (True, True), (False, True)]
)
def test_j2_failure_rolls_back_whole_step(independent, mean):
    s = build(independent=independent, mean=mean)
    s.run(until_step=7)
    snap = s.residual.snapshot()
    u = s.state.u.x.array.copy()
    provider = s.residual.internal.material_batch.material.provider
    original = provider.update

    def fail(point):
        raise RuntimeError("injected material failure")

    provider.update = fail
    with pytest.raises(Exception, match="injected material failure"):
        s.run(until_step=8)
    assert s.residual.snapshot() == snap
    np.testing.assert_array_equal(s.state.u.x.array, u)
    provider.update = original
    s.solve_result()


@pytest.mark.parametrize(
    "independent,mean", [(False, False), (True, False), (True, True), (False, True)]
)
def test_j2_mpi_matches_serial(tmp_path, independent, mean):
    comm = MPI.COMM_WORLD
    s = build(comm, independent=independent, mean=mean)
    s.solve_result()
    serial = build(independent=independent, mean=mean)
    serial.solve_result()
    internal = s.residual.internal
    owned = len(internal.mass_diagonal) // 3
    x = s.state.u.function_space.tabulate_dof_coordinates()[:owned]
    sx = serial.state.u.function_space.tabulate_dof_coordinates()
    indices = np.argmin(np.linalg.norm(x[:, None] - sx[None], axis=2), axis=1)
    np.testing.assert_allclose(
        s.state.u.x.array[: owned * 3].reshape(-1, 3),
        serial.state.u.x.array.reshape(-1, 3)[indices],
        atol=2e-13,
    )
    assert internal.energies()["material_dissipation"] == pytest.approx(
        serial.residual.internal.energies()["material_dissipation"], abs=1e-13
    )


@pytest.mark.parametrize(
    "independent,mean", [(False, False), (True, False), (True, True), (False, True)]
)
def test_elastic_limit_matches_linear_step(independent, mean):
    s = build(yield_stress=1e9, independent=independent, mean=mean)
    internal = s.residual.internal
    batch = internal.material_batch
    s.solve_result()
    reference = build(yield_stress=1e9, independent=independent, mean=mean)
    reference.residual.internal.material_batch = None
    reference.solve_result()
    np.testing.assert_allclose(s.state.u.x.array, reference.state.u.x.array, atol=2e-15)
    assert np.all(batch.states == 0)
    assert batch.dissipation.sum() == 0


def test_consistent_material_force_derivative():
    e = local_element()
    m = material()
    b = CompositeMaterialBatch([e], m)
    u = (e.coordinates @ np.diag([0.02, -0.003, 0.001])).reshape(1, 30)
    b.evaluate(u, 0.1)
    native = m.provider.material.update(np.diag([0.02, -0.003, 0.001]))
    pairs = b.pairs
    C = np.array(
        [[native.algorithmic_tangent[i, j, k, l] for k, l in pairs] for i, j in pairs]
    )
    B = e.strain_matrices()
    K = e.volume / 4 * np.einsum("qai,ab,qbj->ij", B, C, B)
    direction = np.random.default_rng(3).normal(size=(1, 30))
    h = 1e-7
    fd = (b.evaluate(u + h * direction, 0.1) - b.evaluate(u - h * direction, 0.1)) / (
        2 * h
    )
    np.testing.assert_allclose(fd.ravel(), K @ direction.ravel(), rtol=3e-7, atol=1e-8)


def test_material_envelope_rejection_is_atomic():
    from dataclasses import replace

    e = local_element()
    m = material()
    b = CompositeMaterialBatch([e], m)
    original = m.provider.update

    def too_stiff(point):
        r = original(point)
        return replace(r, consistent_tangent=2 * m.elastic_envelope)

    m.provider.update = too_stiff
    snap = b.snapshot()
    with pytest.raises(ValueError, match="stability envelope"):
        b.evaluate(np.zeros((1, 30)), 0.1)
    assert b.snapshot() == snap
    assert b.trial is None


def test_material_checkpoint_validation_is_atomic():
    import copy

    e = local_element()
    b = CompositeMaterialBatch([e], material())
    snap = b.snapshot()
    bad = copy.deepcopy(snap)
    bad["stress"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        b.restore(bad)
    assert b.snapshot() == snap


def test_internal_mode_static_condensation_energy():
    e = local_element()
    enriched = composite_tet11(e.coordinates)
    K = enriched.stiffness(material().elastic_envelope, mean_dilatation=True)
    recovery = -np.linalg.solve(K[30:, 30:], K[30:, :30])
    reduced = K[:30, :30] + K[:30, 30:] @ recovery
    u = np.random.default_rng(72).normal(size=30)
    full = np.r_[u, recovery @ u]
    np.testing.assert_allclose(full @ K @ full, u @ reduced @ u, rtol=1e-13)
    np.testing.assert_allclose((K @ full)[30:], 0, atol=2e-13)
    tied = np.r_[u, u.reshape(10, 3)[4:].mean(0)]
    assert full @ K @ full <= tied @ K @ tied + 1e-12
    # Interior inertia is real; cannot replace dynamic elimination by this K.
    assert enriched.mass(1)[30:].min() > 0


@pytest.mark.parametrize(
    "independent,mean", [(False, False), (True, False), (True, True)]
)
def test_whole_mesh_cyclic_material_path(independent, mean):
    from agentfem import amplitudes

    d = mesh.cuboid(
        (0, 0, 0), (1, 1, 1), (1, 1, 1), comm=MPI.COMM_SELF, cell_type="tetrahedron"
    )
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=d
    )
    u = m.field(
        fields.displacement(
            d,
            element_policy=elements.composite_tet10_policy(
                independent_center=independent, mean_dilatation=mean
            ),
        )
    )
    law = material()
    m.material(law)
    m.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2), value=0.0)
    for xval in np.unique(u.space.tabulate_dof_coordinates()[:, 0]):
        amp = amplitudes.Amplitude(
            f"axial_{xval}",
            lambda t, xval=xval: xval * 0.02 * np.sin(2 * np.pi * t / 0.04),
        )
        m.fix(
            u, on=lambda x, xval=xval: np.isclose(x[0], xval), components=0, value=amp
        )
    s = m.step(target=u, dt=0.001, steps=40, progress=False)
    old = constitutive.J2PlasticState()
    peeq = []
    energy = []
    for i in range(1, 41):
        s.run(until_step=i)
        strain = np.diag([0.02 * np.sin(2 * np.pi * i * 0.001 / 0.04), 0, 0])
        ref = law.provider.material.update(strain, old)
        old = ref.state
        b = s.residual.internal.material_batch
        np.testing.assert_allclose(
            b.stress, np.broadcast_to(ref.stress, b.stress.shape), atol=2e-12
        )
        np.testing.assert_allclose(
            b.states[:, 9], old.equivalent_plastic_strain, atol=2e-13
        )
        peeq.append(b.states[:, 9].mean())
        energy.append(b.dissipation.mean())
    assert peeq[-1] > peeq[9] > 0
    assert np.min(np.diff(energy)) >= -1e-14
    assert energy[-1] > 0


def test_j2_mpi_single_rank_failure_rolls_back():
    comm = MPI.COMM_WORLD
    s = build(comm, independent=True, mean=True)
    s.run(until_step=7)
    batch = s.residual.internal.material_batch
    owner = comm.allreduce(comm.rank if len(batch.states) else comm.size, op=MPI.MIN)
    snap = s.residual.snapshot()
    u, v, a = (getattr(s.state, n).x.array.copy() for n in ("u", "v", "a"))
    provider = batch.material.provider
    original = provider.update
    calls = 0

    def fail_late(point):
        nonlocal calls
        calls += 1
        if comm.rank == owner and calls == 17:
            raise RuntimeError("single-rank interior material failure")
        return original(point)

    provider.update = fail_late
    with pytest.raises(Exception, match="single-rank interior material failure"):
        s.run(until_step=8)
    assert s.residual.snapshot() == snap
    assert batch.trial is None
    for name, old in zip(("u", "v", "a"), (u, v, a)):
        np.testing.assert_array_equal(getattr(s.state, name).x.array, old)
    provider.update = original
    s.solve_result()
    reference = build(comm, independent=True, mean=True)
    reference.solve_result()
    np.testing.assert_array_equal(s.state.u.x.array, reference.state.u.x.array)
    assert batch.snapshot() == reference.residual.internal.material_batch.snapshot()
