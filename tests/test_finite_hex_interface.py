# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private finite bulk/isotropic interface composition, not public promotion."""

import numpy as np
import pytest
from dolfinx import fem

from agentfem import constitutive, fracture, interfaces
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual
from test_nonmatching_global import _blocks
from test_quadrilateral_global import _trace


def setup(stiffness=100, tangent=None):
    domain = _blocks(1, 2, cell_type="hexahedron")
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1e9, hardening_modulus=0
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    internal = FiniteUniformHexResidual(
        u, response, density=2, hourglass_modulus=40, hourglass_scale=0.1
    )
    negative, ni = _trace(u.function_space, False)
    positive, pi = _trace(u.function_space, True)
    pair = interfaces.pair_reference_traces(negative, positive, tolerance=1e-10)
    interface = fracture.nonmatching_cohesive_force(
        pair,
        u,
        interfaces.elastic_cohesive(
            normal_stiffness=stiffness,
            tangential_stiffness=stiffness if tangent is None else tangent,
        ),
        negative_dofs=ni,
        positive_dofs=pi,
    )
    return FiniteHexExplicitResidual(
        internal, law, omega_squared_bound=1e6, cohesive=interface
    )


def test_anisotropic_interface_is_rejected():
    with pytest.raises(NotImplementedError, match="isotropic"):
        setup(tangent=50)


def test_combined_global_force_objectivity_and_checkpoint_identity():
    residual = setup()
    u = residual.internal.displacement
    x = u.function_space.tabulate_dof_coordinates()
    side = np.zeros(len(x))
    for nodes in residual.internal.cell_nodes:
        side[nodes] = np.sign(x[nodes, 2].mean())
    current = 1.02 * x + side[:, None] * np.array([0.02, 0.01, 0.03])
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    forces = []
    energies = []
    saved = residual.snapshot()
    for points in (current, current @ rotation.T + [2, 3, 4]):
        u.x.array[:] = (points - x).ravel()
        residual.update_time(1e-5)
        vector = residual.assemble_vector()
        force = vector.array.reshape(-1, 3).copy()
        vector.destroy()
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.cross(points, force).sum(axis=0), 0, atol=1e-10)
        forces.append(force)
        energies.append(
            residual.cohesive.assembler.evaluate(
                *residual.cohesive._values()
            ).stored_energy
        )
        residual.restore(saved)
    np.testing.assert_allclose(forces[1], forces[0] @ rotation.T, atol=1e-10)
    assert energies[1] == pytest.approx(energies[0], rel=1e-12)
    with pytest.raises(ValueError, match="identity"):
        setup(stiffness=101).restore(saved)


def test_combined_nonfinite_force_rolls_back_material_and_interface(monkeypatch):
    residual = setup()
    before = residual.snapshot()

    def invalid(vector):
        vector.array[0] = np.nan

    monkeypatch.setattr(residual.cohesive, "add_to_vector", invalid)
    residual.update_time(1e-5)
    with pytest.raises(ValueError, match="Non-finite combined"):
        residual.assemble_vector()
    for name, field in residual._fields().items():
        np.testing.assert_array_equal(field.x.array, before["fields"][name])
    residual.restore(before)
    assert residual.snapshot() == before


def dynamic_step():
    from agentfem import problems, state, time
    from agentfem.diagnostics import MechanicalEnergyMonitor

    residual = setup()
    history = state.second_order_state(residual.internal.displacement)
    residual.internal.displacement = history.u.value
    residual.cohesive.displacement = history.u.value
    x = history.u.value.function_space.tabulate_dof_coordinates()
    velocity = 1e-3 * x
    # Interface nodes belong to one side only. Set once, not once per cell.
    for nodes in residual.internal.cell_nodes:
        velocity[nodes, 2] = 1e-3 * x[nodes, 2] + 1e-4 * np.sign(x[nodes, 2].mean())
    history.v.value.x.array[:] = velocity.ravel()
    return problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(
            state=history, mass=residual.internal
        ),
        history_monitor=MechanicalEnergyMonitor(mass=residual.internal.mass_diagonal),
        residual=residual,
        dt=1e-5,
        steps=4,
        progress=False,
    )


def test_combined_explicit_restart_and_post_interface_commit_failure(
    tmp_path, monkeypatch
):
    reference = dynamic_step()
    reference.run()
    step = dynamic_step()
    step.run(until_step=2)
    checkpoint = step.save_checkpoint(tmp_path / "combined")
    resumed = dynamic_step()
    resumed.load_checkpoint(checkpoint)
    saved = resumed.residual.snapshot()
    nodal = resumed.state.snapshot()
    original = resumed.residual.cohesive.commit

    def fail_after_commit():
        original()
        raise RuntimeError("injected interface post-commit")

    monkeypatch.setattr(resumed.residual.cohesive, "commit", fail_after_commit)
    with pytest.raises(RuntimeError, match="interface post-commit"):
        resumed.run()
    assert resumed.completed_steps == 2
    assert resumed.residual.snapshot() == saved
    for name, value in nodal["fields"].items():
        np.testing.assert_array_equal(resumed.state.snapshot()["fields"][name], value)
    monkeypatch.setattr(resumed.residual.cohesive, "commit", original)
    resumed.run()
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(resumed.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
        )
