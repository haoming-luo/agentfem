"""Plane-stress reduction keeps full material physics and planar derivatives distinct."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agentfem import constitutive as c
from agentfem.constitutive.plane_stress import PlaneStressReduction, reduce_plane_stress

spec = importlib.util.spec_from_file_location(
    "plane_stress_history_example",
    Path(__file__).parents[1] / "examples/history_material/material.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
HistoryMaterial = example.HistoryMaterial
POLICY = PlaneStressReduction(symmetry="reflection_symmetric")
INDICES = (0, 1, 3, 4)


def point(material, F=None, *, old_F=None, state=None, dt=0.1, time=None):
    return c.MaterialPointInput(
        np.eye(3) if old_F is None else old_F,
        np.diag((1.1, 0.98, 1.0)) if F is None else F,
        dt if time is None else time,
        dt,
        [],
        np.zeros(material.state_schema.size) if state is None else state,
        material.state_schema,
    )


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_local_reduction_against_independent_green_strain_solution(mechanism):
    material = HistoryMaterial(mechanism=mechanism)
    previous_F, state = np.eye(3), np.zeros(material.state_schema.size)
    for index, stretch in enumerate((1.08, 1.14, 1.14, 1.03, 0.97, 1.06), start=1):
        F = np.array(((stretch, 0.04, 0), (0.01, 1.0, 0), (0, 0, previous_F[2, 2])))
        request = point(
            material, F, old_F=previous_F, state=state, dt=0.1, time=index * 0.1
        )
        before = request.state_old.copy()
        response = reduce_plane_stress(material, request, reduction=POLICY)
        E2 = (F[:2, :2].T @ F[:2, :2] - np.eye(2)) / 2
        b = (
            material.branch / (1 + 0.1 / material.relaxation_time)
            if mechanism == "viscoelastic"
            else 0
        )
        Q33 = state[8] if mechanism == "viscoelastic" else 0
        E33 = (2 * b * Q33 - material.lame * np.trace(E2)) / (
            material.lame + 2 * material.shear + 2 * b
        )
        np.testing.assert_allclose(
            response.deformation_gradient[2, 2], np.sqrt(1 + 2 * E33), rtol=1e-10
        )
        assert abs(response.first_piola_stress[2, 2]) <= response.tolerance
        np.testing.assert_array_equal(request.state_old, before)
        previous_F = response.deformation_gradient
        state = response.material_response.state_new
    assert state[-1] == pytest.approx(0.6)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
@pytest.mark.parametrize("unloading", [False, True])
def test_condensed_algorithmic_tangent_by_independent_reduced_differences(
    mechanism, unloading
):
    material = HistoryMaterial(mechanism=mechanism)
    first = reduce_plane_stress(
        material, point(material, np.diag((1.25, 1.0, 1.0))), reduction=POLICY
    )
    request = point(
        material,
        np.array(
            (
                (1.05 if unloading else 1.35, 0.05, 0),
                (0.02, 0.98, 0),
                (0, 0, first.deformation_gradient[2, 2]),
            )
        ),
        old_F=first.deformation_gradient,
        state=first.material_response.state_new,
    )
    reduced = reduce_plane_stress(material, request, reduction=POLICY)
    differences = np.zeros((4, 4))
    for j, component in enumerate(INDICES):
        perturbation = np.zeros((3, 3))
        perturbation.flat[component] = 1e-6
        plus = reduce_plane_stress(
            material,
            replace(
                request,
                deformation_gradient_new=request.deformation_gradient_new
                + perturbation,
            ),
            reduction=POLICY,
        )
        minus = reduce_plane_stress(
            material,
            replace(
                request,
                deformation_gradient_new=request.deformation_gradient_new
                - perturbation,
            ),
            reduction=POLICY,
        )
        differences[:, j] = (
            (plus.first_piola_stress - minus.first_piola_stress) / 2e-6
        ).ravel()[list(INDICES)]
    np.testing.assert_allclose(
        reduced.in_plane_tangent, differences, rtol=2e-6, atol=1e-7
    )
    # The original full derivative is intentionally retained, not overwritten.
    full = reduced.material_response.consistent_tangent[np.ix_(INDICES, INDICES)]
    assert np.linalg.norm(full - reduced.in_plane_tangent) > 1e-2
    evidence = c.check_plane_stress_tangent(material, request, reduction=POLICY)
    assert evidence.accepted and evidence.relative_error < 2e-6
    assert evidence.summary()["component_order"] == ["11", "12", "21", "22"]
    assert evidence.summary()["storage"] == "matrix_4x4"


class NeoHookeanHistory:
    name = "neo_hookean_with_clock"
    tangent_convention = c.MaterialTangentConvention.first_piola_deformation_gradient()
    state_schema = c.MaterialStateSchema(name, (c.MaterialStateVariable("AGE"),))

    def __init__(self, scale=1.0):
        self.mu, self.lame = 50 * scale, 100 * scale

    def summary(self):
        return {"kind": self.name, "mu": self.mu, "lame": self.lame, "version": 1}

    def update(self, p):
        F = p.deformation_gradient_new
        inverse = np.linalg.inv(F).T
        J = np.linalg.det(F)
        b = self.lame * np.log(J) - self.mu
        P = self.mu * F + b * inverse
        A = (
            self.mu * np.einsum("ik,JL->iJkL", np.eye(3), np.eye(3))
            + self.lame * np.einsum("iJ,kL->iJkL", inverse, inverse)
            - b * np.einsum("iL,kJ->iJkL", inverse, inverse)
        )
        return c.MaterialPointOutput(
            P @ F.T / J,
            A.reshape(9, 9),
            p.state_old + p.time_increment,
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
        )


@pytest.mark.parametrize("stretch", [0.7, 0.95, 1.01, 1.3, 1.7])
def test_specialized_neo_hookean_reference_and_stress_conversion(stretch):
    material = NeoHookeanHistory()
    request = point(material, np.array(((stretch, 0.1, 0), (0.04, 1.04, 0), (0, 0, 1))))
    reduced = reduce_plane_stress(material, request, reduction=POLICY)
    # lambda=100, mu=50 -> E=400/3 and nu=1/3.
    properties = c.neo_hookean_plane_stress(young=400 / 3, poisson=1 / 3)
    F2 = request.deformation_gradient_new[:2, :2]
    np.testing.assert_allclose(
        reduced.deformation_gradient[2, 2],
        c.plane_stress_thickness_stretch_value(F2, properties),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        reduced.first_piola_stress[:2, :2],
        c.plane_stress_first_piola_value(F2, properties),
        rtol=1e-10,
        atol=1e-9,
    )
    F = reduced.deformation_gradient
    np.testing.assert_allclose(
        reduced.material_response.cauchy_stress,
        reduced.first_piola_stress @ F.T / np.linalg.det(F),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("scale", [1e-200, 1e-12, 1.0, 1e12, 1e200])
def test_reduction_criterion_is_not_tied_to_a_stress_unit(scale):
    reduced = reduce_plane_stress(
        NeoHookeanHistory(scale), point(NeoHookeanHistory(scale)), reduction=POLICY
    )
    reference = reduce_plane_stress(
        NeoHookeanHistory(), point(NeoHookeanHistory()), reduction=POLICY
    )
    np.testing.assert_allclose(
        reduced.deformation_gradient, reference.deformation_gradient, rtol=1e-12
    )
    np.testing.assert_allclose(
        reduced.in_plane_tangent / scale, reference.in_plane_tangent, rtol=1e-12
    )


def test_positive_logarithmic_stretch_recovers_from_bad_initial_guess():
    material = NeoHookeanHistory()
    reduced = reduce_plane_stress(
        material, point(material, np.diag((1.2, 1.0, 0.08))), reduction=POLICY
    )
    assert reduced.deformation_gradient[2, 2] > 0
    assert reduced.iterations > 2


def test_no_hidden_history_update_during_local_newton():
    material = NeoHookeanHistory()
    original = material.update
    seen = []

    def update(request):
        seen.append((request.state_old.copy(), request.deformation_gradient_old.copy()))
        return original(request)

    material.update = update
    request = point(material)
    result = reduce_plane_stress(material, request, reduction=POLICY)
    assert len(seen) > 1
    for state, old_F in seen:
        np.testing.assert_array_equal(state, request.state_old)
        np.testing.assert_array_equal(old_F, request.deformation_gradient_old)
    assert result.material_response.state_new[0] == pytest.approx(
        request.time_increment
    )


@pytest.mark.parametrize(
    "failure", ["nonplanar", "shear", "coupling", "singular", "iterations"]
)
def test_unsupported_or_failed_reduction_leaves_request_unchanged(failure):
    material = NeoHookeanHistory()
    request = point(material)
    policy = POLICY
    if failure == "nonplanar":
        F = request.deformation_gradient_new.copy()
        F[0, 2] = 0.01
        request = replace(request, deformation_gradient_new=F)
    elif failure == "iterations":
        policy = replace(POLICY, maximum_iterations=1)
    else:
        original = material.update

        def broken(p):
            response = original(p)
            if failure == "shear":
                stress = response.cauchy_stress.copy()
                stress[0, 2] = stress[2, 0] = 1.0
                return replace(response, cauchy_stress=stress)
            A = response.consistent_tangent.copy()
            if failure == "coupling":
                A[2, 0] = 1.0
            else:
                A[8, 8] = 0
            return replace(response, consistent_tangent=A)

        material.update = broken
    before = request.state_old.copy()
    with pytest.raises((ValueError, RuntimeError)):
        reduce_plane_stress(material, request, reduction=policy)
    np.testing.assert_array_equal(request.state_old, before)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"symmetry": "anisotropic"},
        {"symmetry": "reflection_symmetric", "relative_tolerance": 0},
        {"symmetry": "reflection_symmetric", "maximum_iterations": True},
        {"symmetry": "reflection_symmetric", "relative_tolerance": True},
        {"symmetry": "reflection_symmetric", "absolute_tolerance": False},
    ],
)
def test_explicit_policy_contract(options):
    with pytest.raises((TypeError, ValueError)):
        PlaneStressReduction(**options)


def test_public_reduced_tangent_check_detects_wrong_derivative_without_mutating_old_state():
    material = NeoHookeanHistory()
    original = material.update

    def wrong(p):
        response = original(p)
        tangent = response.consistent_tangent.copy()
        tangent[0, 0] *= 1.2
        return replace(response, consistent_tangent=tangent)

    material.update = wrong
    request = point(material)
    before = request.state_old.copy()
    evidence = c.check_plane_stress_tangent(material, request, reduction=POLICY)
    assert not evidence.accepted and evidence.relative_error > 0.01
    np.testing.assert_array_equal(request.state_old, before)


@pytest.mark.parametrize(
    "options",
    [
        {"relative_step": 0},
        {"relative_step": np.nan},
        {"relative_step": 1e-30},
        {"tolerance": 0},
        {"tolerance": np.inf},
    ],
)
def test_public_reduced_tangent_check_rejects_invalid_controls(options):
    material = NeoHookeanHistory()
    with pytest.raises(ValueError):
        c.check_plane_stress_tangent(
            material, point(material), reduction=POLICY, **options
        )


def structure(
    *,
    mechanism="viscoelastic",
    material=None,
    degree=1,
    nonuniform=False,
    incrementation=None,
    policy=POLICY,
    size=2,
    checkpoint=None,
    output_every=1,
):
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from agentfem import amplitudes, fields, mesh, models, steps, studies

    domain = dmesh.create_unit_square(MPI.COMM_SELF, size, size)
    model = models.create(
        study=studies.static_solid(
            dimension=2, assumption="plane_stress", nonlinear=True
        ),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain, degree=degree))
    material = HistoryMaterial(mechanism=mechanism) if material is None else material
    model.material(material)
    if nonuniform:
        model.clamp(u, on=mesh.face(domain, axis="x", value=0))
        model.traction((2.0, 0.3), on=mesh.face(domain, axis="x", value=1))
    else:
        for axis in range(2):
            model.fix(u, on=mesh.face(domain, axis="xy"[axis], value=0), component=axis)
            model.fix(
                u,
                on=mesh.face(domain, axis="xy"[axis], value=1),
                component=axis,
                value=0.12 if axis == 0 else 0.0,
            )
    step = model.step(
        target=u,
        duration=2.0,
        time_unit="s",
        plane_stress=policy,
        incrementation=incrementation or steps.fixed(8),
        progress=False,
        amplitude=amplitudes.tabular((0.0, 0.5, 0.75, 1.0), (0.0, 1.0, 1.0, 0.2)),
        quadrature_degree=2 * degree,
        checkpoint=checkpoint,
        output_every=output_every,
    )
    return step, material


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
@pytest.mark.parametrize("degree", [1, 2])
def test_structural_patch_keeps_true_3d_gradient_and_material_response(
    mechanism, degree
):
    step, material = structure(mechanism=mechanism, degree=degree)
    old_F, state = np.eye(3), np.zeros(material.state_schema.size)

    def compare(current):
        nonlocal old_F, state
        station = current.accepted_increments[-1]
        F = np.diag(
            (1 + 0.12 * current.amplitude(station["coordinate"]), 1.0, old_F[2, 2])
        )
        reduced = reduce_plane_stress(
            material,
            point(material, F, old_F=old_F, state=state, dt=station["dt"]),
            reduction=POLICY,
        )
        expected = reduced.material_response
        count = len(current._accepted_F)
        np.testing.assert_allclose(
            current._accepted_F,
            np.broadcast_to(reduced.deformation_gradient, (count, 3, 3)),
            atol=2e-9,
        )
        np.testing.assert_allclose(
            current.state.committed_state_vectors(),
            np.broadcast_to(expected.state_new, (count, len(state))),
            atol=2e-9,
        )
        np.testing.assert_allclose(
            current.response.cauchy_stress.values,
            np.broadcast_to(expected.cauchy_stress, (count, 3, 3)),
            atol=2e-8,
        )
        assert np.max(abs(current.response.first_piola_stress.values[:, 2, 2])) < 1e-8
        old_F, state = reduced.deformation_gradient, expected.state_new

    step.solve(on_accept=compare)
    assert not np.allclose(step._accepted_F[:, 2, 2], 1)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_plane_stress_restart_continuity_and_atomic_corruption(tmp_path, mechanism):
    whole, _ = structure(mechanism=mechanism)
    whole.solve()
    first, _ = structure(mechanism=mechanism)
    first.solve(until=0.5)
    path = first.save_checkpoint(tmp_path / "half")
    resumed, material = structure(mechanism=mechanism)
    original_update = material.update
    material.update = lambda p: (_ for _ in ()).throw(
        AssertionError("restore must not integrate")
    )
    resumed.load_checkpoint(path)
    material.update = original_update
    resumed.solve()
    for name in whole._responses:
        np.testing.assert_allclose(
            resumed._responses[name].x.array,
            whole._responses[name].x.array,
            rtol=1e-10,
            atol=1e-10,
        )
    np.testing.assert_allclose(resumed._accepted_F, whole._accepted_F, atol=1e-12)
    np.testing.assert_allclose(
        resumed.state.committed_state_vectors(),
        whole.state.committed_state_vectors(),
        atol=1e-12,
    )
    with np.load(path) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data["F"][:, 2, 2] *= 1.1
    broken = tmp_path / "corrupt.npz"
    np.savez(broken, **data)
    before = resumed._snapshot()
    with pytest.raises(ValueError, match="effective plane-stress gradient"):
        resumed.load_checkpoint(broken)
    np.testing.assert_array_equal(resumed._accepted_F, before["F"])
    np.testing.assert_array_equal(resumed.solution.x.array, before["U"])


def test_global_plane_stress_failed_local_update_rolls_back_all_fields():
    step, material = structure()
    step.solve(until=0.5)
    before = step._snapshot()
    original = material.update

    def fail(p):
        if p.time > 1:
            raise RuntimeError("deliberate constitutive failure")
        return original(p)

    material.update = fail
    with pytest.raises(RuntimeError, match="rejected"):
        step.solve()
    np.testing.assert_array_equal(step.solution.x.array, before["U"])
    np.testing.assert_array_equal(step._accepted_F, before["F"])
    for name in step._responses:
        np.testing.assert_array_equal(
            step._responses[name].x.array, before["response"][name]
        )
    material.update = original
    step.solve()
    assert step.time == pytest.approx(2.0)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
@pytest.mark.parametrize("unloading", [False, True])
def test_nonuniform_reduced_global_tangent_matches_residual_differences(
    mechanism, unloading
):
    import dolfinx.fem.petsc as fp

    step, _ = structure(mechanism=mechanism, nonuniform=True)
    step.solve(until=0.5)
    snapshot = step._snapshot()
    start, target = 0.5, 0.625
    step._apply_loading(target)
    # Stay strictly inside a smooth history branch. A centered difference at
    # the previous maximum crosses the damage switch and is not either
    # branch's algorithmic derivative. Do not loosen the tangent tolerance.
    step.solution.x.array[:] *= 0.8 if unloading else 1.2
    step.solution.x.scatter_forward()
    # Compare the same fixed-old-state update and constrained residual.
    baseline = step._evaluate(target, start)
    if mechanism == "damage":
        E = (np.swapaxes(step._trial_F, 1, 2) @ step._trial_F - np.eye(3)) / 2
        magnitude = np.linalg.norm(E, axis=(1, 2))
        old_maximum = step.state.committed_state_vectors()[:, 0]
        assert (
            np.all(magnitude < old_maximum)
            if unloading
            else np.all(magnitude > old_maximum)
        )
    A = fp.assemble_matrix(step.tangent_form, bcs=step.bcs)
    A.assemble()
    direction = np.random.default_rng(42).normal(size=step.solution.x.array.size)
    for bc in step.bcs:
        direction[bc.dof_indices()[0]] = 0
    vector = baseline.residual.duplicate()
    vector.array[:] = direction
    exact = vector.duplicate()
    A.mult(vector, exact)
    base = step.solution.x.array.copy()
    values = []
    for sign in (1.0, -1.0):
        step.solution.x.array[:] = base + sign * 1e-6 * direction
        step.solution.x.scatter_forward()
        evaluation = step._evaluate(target, start)
        values.append(evaluation.residual.array_r.copy())
        evaluation.residual.destroy()
    # _evaluate returns -R with essential rows eliminated.
    np.testing.assert_allclose(
        -(values[0] - values[1]) / 2e-6, exact.array_r, rtol=3e-6, atol=1e-6
    )
    for obj in (baseline.residual, A, vector, exact):
        obj.destroy()
    step._restore(snapshot)
    step.solve()


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_traction_controlled_structure_against_independent_scalar_equilibrium(
    mechanism,
):
    from scipy.optimize import brentq
    from mpi4py import MPI
    from dolfinx import mesh as dmesh
    from agentfem import amplitudes, fields, mesh, models, steps, studies

    domain = dmesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    model = models.create(
        study=studies.static_solid(
            dimension=2, assumption="plane_stress", nonlinear=True
        ),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain))
    material = HistoryMaterial(mechanism=mechanism)
    model.material(material)
    # Roller constraints admit exact homogeneous lateral contraction.
    model.fix(u, on=mesh.face(domain, axis="x", value=0), component=0)
    model.fix(u, on=mesh.face(domain, axis="y", value=0), component=1)
    model.traction((6.0, 0.0), on=mesh.face(domain, axis="x", value=1))
    cycle = amplitudes.tabular((0.0, 0.5, 0.75, 1.0), (0.0, 1.0, 1.0, 0.2))
    step = model.step(
        target=u,
        duration=2.0,
        time_unit="s",
        amplitude=cycle,
        plane_stress=POLICY,
        incrementation=steps.fixed(8),
        progress=False,
    )
    Q, kappa = np.zeros(3), 0.0

    def compare(current):
        nonlocal Q, kappa
        station = current.accepted_increments[-1]
        ratio = station["dt"] / material.relaxation_time
        b = material.branch / (1 + ratio) if mechanism == "viscoelastic" else 0.0
        target = 6 * cycle(station["coordinate"])

        def analytical(stretch):
            E11 = (stretch**2 - 1) / 2
            transverse = (2 * b * Q[1] - material.lame * E11) / (
                2 * material.lame + 2 * material.shear + 2 * b
            )
            E = np.array((E11, transverse, transverse))
            S11 = (
                material.lame * E.sum()
                + 2 * material.shear * E11
                + 2 * b * (E11 - Q[0])
            )
            if mechanism == "damage":
                S11 *= np.exp(-material.damage_rate * max(kappa, np.linalg.norm(E)))
            return stretch * S11, E

        stretch = brentq(
            lambda value: analytical(value)[0] - target, 0.95, 1.2, xtol=1e-14
        )
        _, E = analytical(stretch)
        expected_F = np.diag(np.sqrt(1 + 2 * E))
        np.testing.assert_allclose(
            current._accepted_F,
            np.broadcast_to(expected_F, current._accepted_F.shape),
            atol=2e-9,
        )
        P = current.response.first_piola_stress.values
        np.testing.assert_allclose(P[:, 0, 0], target, atol=2e-8)
        np.testing.assert_allclose(P[:, 1:, 1:], 0.0, atol=2e-8)
        if mechanism == "viscoelastic":
            Q = (Q + ratio * E) / (1 + ratio)
            expected_state = np.r_[np.diag(Q).ravel(), station["time"]]
        else:
            kappa = max(kappa, np.linalg.norm(E))
            expected_state = np.r_[kappa, station["time"]]
        state = current.state.committed_state_vectors()
        np.testing.assert_allclose(
            state, np.broadcast_to(expected_state, state.shape), atol=2e-9
        )

    step.solve(on_accept=compare)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_reduced_output_warps_one_grid_with_stress_and_thickness(tmp_path, mechanism):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    step, _ = structure(mechanism=mechanism)
    path = tmp_path / "results.xdmf"
    result = step.solve_result(output=path)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    cells = step.state.domain.topology.index_map(2).size_local
    for station in step.accepted_increments:
        reader.UpdateTimeStep(station["time"])
        grid = reader.GetOutputDataObject(0)
        assert grid.IsA("vtkUnstructuredGrid") and grid.GetNumberOfCells() == cells
        F = vtk_to_numpy(grid.GetCellData().GetArray("F_CELL")).reshape(-1, 3, 3)
        stress = vtk_to_numpy(grid.GetCellData().GetArray("S_CELL")).reshape(-1, 3, 3)
        np.testing.assert_allclose(stress[:, 2, 2], 0.0, atol=1e-8)
        assert np.all(F[:, 2, 2] > 0.0)
        np.testing.assert_allclose(
            vtk_to_numpy(grid.GetCellData().GetArray("AGE_CELL")), station["time"]
        )
        warp = vtk.vtkWarpVector()
        warp.SetInputData(grid)
        warp.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, step.solution.name
        )
        warp.Update()
        assert warp.GetOutput().GetNumberOfCells() == cells
        for name in ("F_CELL", "S_CELL", "P_CELL", "AGE_CELL"):
            assert warp.GetOutput().GetCellData().GetArray(name) is not None
    assert result.fields["F"].location == "quadrature_points"
    assert result.fields["S"].processing["stress_measure"] == "cauchy"


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_in_plane_superposed_rotation_preserves_reduced_history(mechanism):
    material = HistoryMaterial(mechanism=mechanism)
    p = point(material)
    first = reduce_plane_stress(material, p, reduction=POLICY)
    p = point(
        material,
        np.array(
            ((1.18, 0.02, 0), (0.03, 0.98, 0), (0, 0, first.deformation_gradient[2, 2]))
        ),
        old_F=first.deformation_gradient,
        state=first.material_response.state_new,
    )
    angle = 0.8
    rotation = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        )
    )
    reference = reduce_plane_stress(material, p, reduction=POLICY)
    transformed = reduce_plane_stress(
        material,
        replace(
            p,
            deformation_gradient_old=rotation @ p.deformation_gradient_old,
            deformation_gradient_new=rotation @ p.deformation_gradient_new,
        ),
        reduction=POLICY,
    )
    np.testing.assert_allclose(
        transformed.first_piola_stress,
        rotation @ reference.first_piola_stress,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        transformed.material_response.state_new,
        reference.material_response.state_new,
        atol=1e-13,
    )


def test_plane_stress_material_requested_cutback_and_reduction_identity(tmp_path):
    from agentfem import steps

    class LimitedMaterial(HistoryMaterial):
        def update(self, p):
            assert p.time == pytest.approx(p.state_old[-1] + p.time_increment)
            return replace(
                super().update(p),
                suggested_time_scale=0.5 if p.time_increment > 0.5 + 1e-12 else 1.0,
            )

    step, _ = structure(
        material=LimitedMaterial(),
        incrementation=steps.automatic(initial=0.5, minimum=0.01, maximum=0.5),
    )
    step.solve()
    assert any(not r["accepted"] for r in step.attempted_increments)
    assert all(r["dt"] <= 0.5 + 1e-12 for r in step.accepted_increments)
    control, _ = structure(material=LimitedMaterial(), incrementation=steps.fixed(1))
    for row in step.accepted_increments:
        control.solve(until=row["coordinate"])
    np.testing.assert_allclose(step._accepted_F, control._accepted_F, atol=1e-11)
    np.testing.assert_allclose(
        step.state.committed_state_vectors(),
        control.state.committed_state_vectors(),
        atol=1e-11,
    )
    path = step.save_checkpoint(tmp_path / "checkpoint")
    other, _ = structure(
        material=LimitedMaterial(),
        incrementation=steps.automatic(initial=0.5, minimum=0.01, maximum=0.5),
        policy=replace(POLICY, relative_tolerance=1e-9),
    )
    with pytest.raises(ValueError, match="identity differs"):
        other.load_checkpoint(path)
    assert other.time == 0.0


def test_nonuniform_damage_manufactured_solution_spatial_refinement():
    """Independent equilibrium: u=(a*x*x,0), E33=-lambda*E11/(lambda+2mu)."""
    import ufl
    from mpi4py import MPI
    from dolfinx import fem, mesh as dmesh
    from agentfem import fields, mesh, models, steps, studies

    errors = []
    amplitude = 0.04
    for size, degree in ((2, 1), (4, 1), (8, 1), (2, 2)):
        domain = dmesh.create_unit_square(MPI.COMM_SELF, size, size)
        model = models.create(
            study=studies.static_solid(
                dimension=2, assumption="plane_stress", nonlinear=True
            ),
            mesh=domain,
        )
        u = model.field(fields.displacement(domain, degree=degree))
        material = HistoryMaterial(mechanism="damage")
        model.material(material)
        model.fix(u, on=mesh.face(domain, axis="x", value=0), component=0)
        model.fix(
            u, on=mesh.face(domain, axis="x", value=1), component=0, value=amplitude
        )
        for value in (0.0, 1.0):
            model.fix(u, on=mesh.face(domain, axis="y", value=value), component=1)
        x = ufl.SpatialCoordinate(domain)
        stretch = 1 + 2 * amplitude * x[0]
        E = (stretch**2 - 1) / 2
        beta = material.lame / (material.lame + 2 * material.shear)
        modulus = material.lame * (1 - beta) + 2 * material.shear
        rate = material.damage_rate * np.sqrt(1 + beta**2)
        derivative = modulus * ufl.exp(-rate * E) * (E + stretch**2 * (1 - rate * E))
        model.body_force(ufl.as_vector((-2 * amplitude * derivative, 0.0)))
        step = model.step(
            target=u,
            duration=1.0,
            time_unit="s",
            plane_stress=POLICY,
            incrementation=steps.fixed(4),
            quadrature_degree=6,
            progress=False,
        )
        step.solve()
        exact = ufl.as_vector((amplitude * x[0] ** 2, 0.0))
        error = np.sqrt(
            fem.assemble_scalar(
                fem.form(ufl.inner(u.value - exact, u.value - exact) * ufl.dx)
            )
        )
        errors.append(error)
        # Check that the target is on the loading branch, not an unloaded
        # state carrying a larger historical damage driver.
        effective = step._accepted_F
        E_current = (np.swapaxes(effective, 1, 2) @ effective - np.eye(3)) / 2
        np.testing.assert_allclose(
            step.state.committed_state_vectors()[:, 0],
            np.linalg.norm(E_current, axis=(1, 2)),
            atol=2e-10,
        )
    assert errors[0] / errors[1] > 3.5, errors
    assert errors[1] / errors[2] > 3.5, errors
    assert errors[3] < 1e-8, errors


def test_plane_stress_viscous_time_refinement_against_exact_coupled_ode():
    """An independent continuous-time reference, not repeated material.update."""
    material = HistoryMaterial()
    speed, end = 0.1, 1.0
    E11 = speed * end + 0.5 * speed**2 * end**2
    D = material.lame + 2 * material.shear + 2 * material.branch

    def convolution(rate):
        tail = -np.expm1(-rate * end)
        return speed * (end - tail / rate) + 0.5 * speed**2 * (
            end**2 - 2 * end / rate + 2 * tail / rate**2
        )

    Q11 = convolution(1 / material.relaxation_time)
    rate = (material.lame + 2 * material.shear) / (material.relaxation_time * D)
    Q33 = -material.lame / (material.lame + 2 * material.shear) * convolution(rate)
    E33 = (2 * material.branch * Q33 - material.lame * E11) / D
    expected = np.array((Q11, Q33, np.sqrt(1 + 2 * E33)))
    errors = []
    for count in (8, 16, 32, 64):
        old_F, state = np.eye(3), np.zeros(material.state_schema.size)
        for number in range(1, count + 1):
            dt, time = end / count, number * end / count
            request = replace(
                point(
                    material,
                    np.diag((1 + speed * time, 1.0, old_F[2, 2])),
                    old_F=old_F,
                    state=state,
                    dt=dt,
                ),
                time=time,
            )
            reduced = reduce_plane_stress(material, request, reduction=POLICY)
            old_F, state = (
                reduced.deformation_gradient,
                reduced.material_response.state_new,
            )
        numerical = np.array((state[0], state[8], old_F[2, 2]))
        errors.append(np.linalg.norm(numerical - expected))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 1.8) & (ratios < 2.2)), (errors, ratios)


@pytest.mark.parametrize("chunk", [1, 3, 1024])
def test_reduced_batch_partition_and_late_failure_are_transactional(chunk):
    from mpi4py import MPI
    from dolfinx import mesh

    domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    material = HistoryMaterial()
    state = c.MaterialQuadratureState.create(domain, material.state_schema, degree=2)
    count = len(state.committed_state_vectors())
    gradients = np.tile(np.eye(3), (count, 1, 1))
    gradients[:, 0, 0] = np.linspace(1.05, 1.2, count)
    options = dict(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=gradients,
        time=0.1,
        time_increment=0.1,
        plane_stress=POLICY,
    )
    result = c.update_material_points(
        material, state, max_batch_points=chunk, **options
    )
    reference = c.update_material_points(
        material, state, max_batch_points=count, **options
    )
    assert result.scalar_fallback_points == count and result.provider_batch_calls == 0
    np.testing.assert_allclose(
        result.plane_stress.deformation_gradient,
        reference.plane_stress.deformation_gradient,
    )
    np.testing.assert_allclose(
        result.plane_stress.in_plane_tangent, reference.plane_stress.in_plane_tangent
    )
    np.testing.assert_allclose(result.state_new, reference.state_new)
    state.commit()
    before = state.committed_state_vectors().copy()
    options.update(
        deformation_gradient_old=result.plane_stress.deformation_gradient, time=0.2
    )
    original = material.update
    seen = []

    def failing(p):
        seen.append(p.deformation_gradient_new[0, 0])
        if p.deformation_gradient_new[0, 0] > 1.15:
            raise RuntimeError("deliberate later point failure")
        return original(p)

    material.update = failing
    with pytest.raises(RuntimeError, match="later point failure"):
        c.update_material_points(
            material, state, max_batch_points=chunk, commit=True, **options
        )
    assert min(seen) < 1.1 and max(seen) > 1.15
    np.testing.assert_array_equal(state.committed_state_vectors(), before)
    np.testing.assert_array_equal(state.trial_state_vectors(), before)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_interrupted_output_and_restart_join_preserve_every_accepted_field(
    tmp_path, mechanism
):
    import xml.etree.ElementTree as ET
    import h5py
    from agentfem import results

    full, _ = structure(mechanism=mechanism)
    full_path = tmp_path / "full.xdmf"
    full.solve_result(output=full_path)
    partial, material = structure(mechanism=mechanism)
    update = material.update

    def stop(p):
        if p.time > 1.0:
            raise RuntimeError("deliberate interruption")
        return update(p)

    material.update = stop
    first_path = tmp_path / "first.xdmf"
    with pytest.raises(RuntimeError, match="rejected"):
        partial.solve_result(output=first_path)
    assert partial.time == 1.0
    # The index exists and all referenced heavy data can be opened before any
    # continuation is attempted. A later run must not be needed to repair it.
    assert len(ET.parse(first_path).findall(".//Time")) == 4

    def saved_fields(path):
        frames = []
        for frame in ET.parse(path).findall(".//Grid[@GridType='Uniform']"):
            values = {}
            for attribute in frame.findall("Attribute"):
                item = attribute.find("DataItem")
                filename, dataset = item.text.strip().split(":", 1)
                with h5py.File(path.parent / filename) as archive:
                    values[attribute.get("Name")] = archive[dataset][:]
            frames.append((float(frame.find("Time").get("Value")), values))
        return frames

    first_frames = saved_fields(first_path)
    path = partial.save_checkpoint(tmp_path / "restart")
    resumed, _ = structure(mechanism=mechanism)
    resumed.load_checkpoint(path)
    second_path = tmp_path / "second.xdmf"
    result = resumed.solve_result(output=second_path)
    assert result.metadata["field_output"]["output_scope"] == "continuation_segment"
    assert result.metadata["restart"]
    merged = results.join_xdmf_segments(
        [first_path, second_path],
        tmp_path / "joined.xdmf",
        expected_times=[row["time"] for row in full.accepted_increments],
    )
    reference = saved_fields(full_path)
    joined = saved_fields(merged)
    assert len(joined) == 8
    assert len(first_frames) == 4
    for (time, fields), (ref_time, ref_fields) in zip(joined, reference, strict=True):
        assert time == ref_time and fields.keys() == ref_fields.keys()
        for name in fields:
            np.testing.assert_allclose(
                fields[name], ref_fields[name], rtol=1e-9, atol=1e-10
            )


def test_finite_j2_local_reduction_against_independent_uniaxial_logarithmic_return():
    """Real constitutive law; this does not alter its specialized Step admission."""
    young, poisson, yield_stress, hardening = 1000.0, 0.3, 2.0, 25.0
    material = c.finite_strain_j2_logarithmic(
        young=young,
        poisson=poisson,
        yield_stress=yield_stress,
        hardening_modulus=hardening,
    )
    old_F, state = np.eye(3), material.state_schema.initial_state()
    plastic_log_stretch, accumulated = 0.0, 0.0
    for index, epsilon in enumerate(
        (0.0005, 0.004, 0.02, 0.04, 0.04, 0.01, -0.015, -0.03, 0.005), start=1
    ):
        trial = young * (epsilon - plastic_log_stretch)
        gamma = max(abs(trial) - yield_stress - hardening * accumulated, 0.0) / (
            young + hardening
        )
        plastic_log_stretch += np.sign(trial) * gamma
        accumulated += gamma
        kirchhoff = young * (epsilon - plastic_log_stretch)
        transverse = np.exp(-poisson * kirchhoff / young - 0.5 * plastic_log_stretch)
        F = np.diag((np.exp(epsilon), transverse, old_F[2, 2]))
        request = point(material, F, old_F=old_F, state=state, time=index * 0.1)
        reduced = reduce_plane_stress(material, request, reduction=POLICY)
        np.testing.assert_allclose(
            reduced.deformation_gradient[2, 2], transverse, rtol=2e-9
        )
        J = np.exp((1 - 2 * poisson) * kirchhoff / young)
        expected_stress = np.diag((kirchhoff / J, 0.0, 0.0))
        np.testing.assert_allclose(
            reduced.material_response.cauchy_stress,
            expected_stress,
            rtol=1e-8,
            atol=1e-7,
        )
        expected_plastic = np.diag(
            np.exp(
                (
                    plastic_log_stretch,
                    -0.5 * plastic_log_stretch,
                    -0.5 * plastic_log_stretch,
                )
            )
        )
        np.testing.assert_allclose(
            reduced.material_response.state_new[:9].reshape(3, 3),
            expected_plastic,
            atol=2e-9,
        )
        assert reduced.material_response.state_new[9] == pytest.approx(
            accumulated, rel=1e-8, abs=1e-10
        )
        old_F, state = reduced.deformation_gradient, reduced.material_response.state_new
    check_F = old_F.copy()
    check_F[0, 0] *= 1.01
    evidence = c.check_plane_stress_tangent(
        material,
        point(material, check_F, old_F=old_F, state=state, time=1.0),
        reduction=POLICY,
    )
    assert evidence.accepted, evidence.summary()
