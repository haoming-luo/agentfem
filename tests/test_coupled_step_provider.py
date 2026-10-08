# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from pathlib import Path
from mpi4py import MPI
import numpy as np
import pytest
from agentfem import checkpointing, constraints, loads, models, studies, units
from agentfem.step_providers import step_capability
from test_thermoelastic_procedure import make_step


@contextmanager
def coupled_model():
    with make_step() as source:
        model = models.Model(
            study=studies.coupled_thermoelastic(),
            mesh=source.u.space.mesh,
            unit_system=units.si(),
        )
        u, theta = model.field(source.u), model.field(source.theta)
        model.material(source.material)
        for axis in range(3):
            model.constraint(
                constraints.component_dirichlet(
                    u,
                    axis,
                    marker=lambda x, axis=axis: np.isclose(x[axis], 0),
                    value=0.0,
                )
            )
        heat = model.load(loads.heat_source(10.0, target=theta))
        yield model, u, theta, heat


def test_normal_step_route_and_contract(tmp_path):
    with coupled_model() as (model, u, theta, heat):
        options = dict(dt=0.1, steps=3, temperature_departure=theta, heat_loads=(heat,))
        capability = step_capability(model, target=u, options=options)
        assert capability["ready"]
        assert capability["provider"]["name"] == "staggered_thermoelastic"
        with model.step(target=u, **options) as step:
            result = step.solve_result()
            assert result.metadata["execution_context"]
            assert step.input_identity["declared_inputs"]["complete"]
            expected = 3.0 / (100 + 9 * (1000 / 1.5) * 0.002**2 * 300)
            np.testing.assert_allclose(theta.value.x.array, expected, atol=2e-10)
        assert len(model.steps) == 1


@pytest.mark.parametrize("extra", [dict(history=()), dict(update_load=lambda t: None)])
def test_unsupported_workflow_options_are_not_silently_consumed(extra):
    with coupled_model() as (model, u, theta, heat):
        with pytest.raises(TypeError, match="Unsupported Step option"):
            model.step(
                target=u,
                temperature_departure=theta,
                dt=0.1,
                steps=2,
                heat_loads=(heat,),
                **extra,
            )
        assert not model.steps


def test_registered_heat_cannot_be_silently_dropped():
    with coupled_model() as (model, u, theta, heat):
        with pytest.raises(
            (ValueError, RuntimeError), match="assign every registered load"
        ):
            model.step(target=u, temperature_departure=theta, dt=0.1, steps=2)
        assert not model.steps


def test_wrong_unit_system_rejected():
    with coupled_model() as (model, u, theta, heat):
        model.unit_system = units.n_mm_mpa()
        with pytest.raises((ValueError, RuntimeError), match="units.si"):
            model.step(
                target=u,
                temperature_departure=theta,
                dt=0.1,
                steps=2,
                heat_loads=(heat,),
            )
        assert not model.steps


def test_public_restart_binds_actual_natural_load(tmp_path):
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "public-coupled"), root=0)
    with coupled_model() as (model, u, theta, heat):
        with model.step(
            target=u, temperature_departure=theta, dt=0.1, steps=3, heat_loads=(heat,)
        ) as step:
            step.advance()
            step.save_checkpoint(path)
            step.run()
            expected = theta.value.x.array.copy()
    with coupled_model() as (model, u, theta, heat):
        with model.step(
            target=u, temperature_departure=theta, dt=0.1, steps=3, heat_loads=(heat,)
        ) as step:
            step.load_checkpoint(path)
            step.solve_result()
            np.testing.assert_allclose(theta.value.x.array, expected, atol=1e-10)
    with coupled_model() as (model, u, theta, heat):
        model.loads.clear()
        changed = model.load(loads.heat_source(11.0, target=theta))
        with model.step(
            target=u,
            temperature_departure=theta,
            dt=0.1,
            steps=3,
            heat_loads=(changed,),
        ) as step:
            with pytest.raises((ValueError, RuntimeError)):
                step.load_checkpoint(path)
            assert step.completed_steps == 0
            np.testing.assert_array_equal(theta.value.x.array, 0)


def test_public_progress_and_bounded_scheduled_checkpoints(tmp_path):
    directory = MPI.COMM_WORLD.bcast(str(tmp_path / "scheduled"), root=0)

    class Reporter:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    reporter = Reporter()
    with coupled_model() as (model, u, theta, heat):
        with model.step(
            target=u,
            temperature_departure=theta,
            dt=0.1,
            steps=3,
            heat_loads=(heat,),
            progress=reporter,
            print_every=2,
            checkpoint=checkpointing.every(
                1, directory=directory, keep_last=1, portable=True
            ),
        ) as step:
            result = step.solve_result()
            increments = [e for e in reporter.events if e.kind == "time_increment"]
            assert [e.increment for e in increments] == [1, 2, 3]
            assert [e.display for e in increments] == [False, True, True]
            assert all(
                e.metrics["quadratic_balance_absolute"] < 1e-10 for e in increments
            )
            assert len(result.metadata["scheduled_checkpoints"]) == 1
            kept = result.metadata["scheduled_checkpoints"][0]
            assert Path(kept).exists()
            assert len(list(Path(directory).glob("*.checkpoint.json"))) == 1
    with coupled_model() as (model, u, theta, heat):
        with model.step(
            target=u, temperature_departure=theta, dt=0.1, steps=3, heat_loads=(heat,)
        ) as resumed:
            resumed.load_checkpoint(kept)
            assert resumed.completed_steps == 3


def test_rejected_window_never_schedules_checkpoint(tmp_path):
    directory = MPI.COMM_WORLD.bcast(str(tmp_path / "rejected"), root=0)
    with coupled_model() as (model, u, theta, heat):
        with model.step(
            target=u,
            temperature_departure=theta,
            dt=0.1,
            steps=3,
            heat_loads=(heat,),
            max_iterations=1,
            checkpoint=checkpointing.every(1, directory=directory),
        ) as step:
            with pytest.raises(RuntimeError, match="AFM-COUPLING-001"):
                step.solve_result()
            assert not Path(directory).exists()


def test_manual_advance_uses_configured_policy_and_temporary_override():
    with coupled_model() as (model, u, theta, heat):
        with model.step(
            target=u,
            temperature_departure=theta,
            dt=0.1,
            steps=3,
            heat_loads=(heat,),
            max_iterations=1,
        ) as step:
            with pytest.raises(RuntimeError, match="AFM-COUPLING-001"):
                step.advance()
            assert step.completed_steps == 0
            record = step.advance(max_iterations=200)
            assert record["iteration_policy"]["max_iterations"] == 200
            assert step.iteration_options["max_iterations"] == 1
            with pytest.raises(RuntimeError, match="AFM-COUPLING-001"):
                step.advance()
            assert step.completed_steps == 1
