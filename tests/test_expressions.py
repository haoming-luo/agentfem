from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import expressions, fields, mesh, results


def test_scientific_expression_interpolates_scalar_and_tracks_contract():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.scalar_unknown(domain, degree=2, name="T")

    parsed = expressions.expression("exp(-t) * sin(pi*x) * sin(pi*y)")
    expressions.interpolate(temperature, parsed, parameters={"t": 0.25})

    value = results.probe(temperature, at=(0.5, 0.5))
    assert value == pytest.approx(np.exp(-0.25), rel=1.0e-12)
    assert parsed.summary() == {
        "kind": "scientific_expression",
        "source": "exp(-t) * sin(pi*x) * sin(pi*y)",
        "language": "agentfem-math-v1",
    }


def test_scientific_expression_rejects_python_execution_syntax():
    with pytest.raises(expressions.ExpressionError, match="Only one-argument calls"):
        expressions.expression("__import__('os').system('echo unsafe')")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    with pytest.raises(expressions.ExpressionError, match="Unknown name"):
        expressions.as_ufl("secret*x", domain)

    parsed = expressions.expression("x + t")
    with pytest.raises(expressions.ExpressionError, match="parameter name"):
        parsed.evaluate(np.array([[0.5]]), parameters={"x": 2.0})


def test_constant_expression_interpolates_without_a_coordinate_domain():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    scalar = fields.scalar_unknown(domain, degree=1, name="zero")
    vector = fields.vector_unknown(domain, degree=1, dim=2, name="constant")

    expressions.interpolate(scalar, "0.0")
    expressions.interpolate(vector, ["1.0", "-2.0"])

    np.testing.assert_allclose(scalar.value.x.array, 0.0)
    np.testing.assert_allclose(results.probe(vector, at=(0.25, 0.25)), [1.0, -2.0])


def test_scientific_expression_evaluates_numpy_arrays_without_eval():
    parsed = expressions.expression("exp(-t) * sin(pi*x) + y**2")
    points = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.5]])

    values = parsed.evaluate(points, parameters={"t": 0.25})

    expected = np.exp(-0.25) * np.sin(np.pi * points[0]) + points[1] ** 2
    np.testing.assert_allclose(values, expected)


def test_rectilinear_grid_sampling_uses_standard_axis_and_array_order():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (3, 3),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    displacement = fields.vector_unknown(domain, degree=1, dim=2, name="U")
    displacement.value.interpolate(lambda x: np.vstack((x[0], 2.0 * x[1])))

    sampled = results.sample_rectilinear_grid(
        displacement,
        bbox=(0.0, 1.0, 0.0, 1.0),
        shape=(4, 3),
        reduction="magnitude",
    )

    assert sampled.values.shape == (3, 4)
    assert sampled.inside.shape == (3, 4)
    assert sampled.values[-1, -1] == pytest.approx(np.sqrt(5.0))
    assert sampled.summary()["inside_points"] == 12


def test_rectilinear_grid_sampling_marks_irregular_bbox_points_as_missing():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    value = fields.scalar_unknown(domain, degree=1, name="q")
    value.value.x.array[:] = 2.0
    value.value.x.scatter_forward()

    sampled = results.sample_rectilinear_grid(
        value,
        bbox=(-0.5, 1.5, -0.5, 1.5),
        shape=(5, 5),
    )

    assert sampled.values.shape == (5, 5)
    assert np.isnan(sampled.values[0, 0])
    assert sampled.values[2, 2] == pytest.approx(2.0)
    assert sampled.inside[2, 2]
    assert not sampled.inside[0, 0]
