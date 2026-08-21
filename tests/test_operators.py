from __future__ import annotations

import ufl
from mpi4py import MPI
import pytest

from agentfem import fields, mesh, operators
from agentfem.operators.core import OperatorForm, combine, scale


class _Expression:
    def __init__(self, text):
        self.text = text

    def __add__(self, other):
        return _Expression(f"({self.text}+{other.text})")

    def __mul__(self, factor):
        return _Expression(f"({self.text}*{factor})")

    def __rmul__(self, factor):
        return _Expression(f"({factor}*{self.text})")


class _FormExpression(_Expression):
    def __init__(self, text, arity):
        super().__init__(text)
        self.arity = arity

    def arguments(self):
        return tuple(range(self.arity))


def test_operator_sum_retains_composition_history():
    first = OperatorForm(
        name="K_matrix",
        expression=_Expression("K_matrix"),
        kind="regional_stiffness",
        role="matrix",
        family="elasticity",
    )
    second = OperatorForm(
        name="K_inclusion",
        expression=_Expression("K_inclusion"),
        kind="regional_stiffness",
        role="matrix",
        family="elasticity",
    )

    total = combine(first, second, name="K", kind="partitioned_stiffness")
    summary = total.to_ir()

    assert summary["operation"] == "sum"
    assert summary["metadata"]["operand_count"] == 2
    assert summary["parts"][0]["operator"]["name"] == "K_matrix"
    assert summary["parts"][1]["operator"]["name"] == "K_inclusion"


def test_scaled_operator_records_factor_and_source():
    operator = OperatorForm(
        name="C",
        expression=_Expression("C"),
        kind="capacity",
        role="matrix",
        family="heat_capacity",
    )

    scaled = scale(operator, 0.25, name="C_over_dt")

    assert scaled.operation == "scale"
    assert scaled.parts[0]["factor"] == 0.25
    assert scaled.metadata["source_name"] == "C"


def test_operator_and_system_validation_catch_role_mismatch():
    C = OperatorForm("C", _FormExpression("C", 2), "capacity", role="matrix")
    K = OperatorForm("K", _FormExpression("K", 2), "conduction", role="matrix")
    F = OperatorForm("F", _FormExpression("F", 1), "source", role="vector")

    system = operators.first_order_system(C, K, F)
    bad = operators.first_order_system(C, K, K)

    assert system.equation == "C x_dot + K x = F"
    assert system.validate().is_valid
    assert [item.code for item in bad.validate().errors] == ["AFM-SYS-002"]

    mislabeled = OperatorForm(
        "bad",
        _FormExpression("linear_form", 1),
        "bad_matrix",
        role="matrix",
    )
    assert [item.code for item in mislabeled.validate().errors] == ["AFM-OP-001"]


def test_residual_linearization_records_R_to_Kt_relationship():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain)
    residual_form = (
        ufl.inner(
            ufl.grad(displacement.value),
            ufl.grad(displacement.test),
        )
        * ufl.dx
    )

    residual = operators.residual_operator(
        residual_form,
        family="prototype_nonlinear_solid",
    )
    tangent = operators.linearize(residual, displacement)

    assert operators.form_arity(residual) == 1
    assert operators.form_arity(tangent) == 2
    assert residual.validate().is_valid
    assert tangent.validate().is_valid
    assert tangent.operation == "linearize"
    assert tangent.metadata["equation"] == "K_t = dR/du"


def test_standard_boundary_and_proportional_damping_operators_are_checked():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.temperature(domain)
    boundary_measure = ufl.ds(domain=domain)
    robin = operators.robin_operator(
        temperature,
        12.0,
        measure=boundary_measure,
    )
    ambient = operators.robin_source_vector(
        temperature,
        12.0,
        293.15,
        measure=boundary_measure,
    )
    flux = operators.flux_vector(
        100.0,
        temperature,
        measure=boundary_measure,
    )
    capacity = operators.capacity_operator(temperature, 1.0)
    conduction = operators.conduction_operator(temperature, 1.0)
    damping = operators.rayleigh_damping(
        capacity,
        conduction,
        mass_coefficient=0.02,
        stiffness_coefficient=0.001,
    )

    assert operators.form_arity(robin) == 2
    assert operators.form_arity(ambient) == 1
    assert operators.form_arity(flux) == 1
    assert damping.family == "rayleigh_damping"
    assert damping.validate().is_valid
    with pytest.raises(ValueError, match="non-zero"):
        operators.rayleigh_damping(capacity, conduction)


def test_transport_and_reaction_operators_keep_scientific_semantics():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    concentration = fields.scalar_unknown(domain, name="concentration")
    advection = operators.advection_operator(
        concentration.trial,
        concentration.test,
        (2.0, 0.5),
    )
    reaction = operators.reaction_expression(
        concentration.value,
        {"type": "allen_cahn", "lambda": 2.0},
    )
    logistic = operators.reaction_expression(
        concentration.value,
        {"type": "logistic", "rho": 3.0},
    )
    supg = operators.streamline_upwind_operator(
        ufl.dot(
            operators.as_velocity((2.0, 0.5)),
            ufl.grad(concentration.trial),
        ),
        concentration.test,
        (2.0, 0.5),
        domain=domain,
    )
    burgers = operators.burgers_convection_operator(
        concentration.value,
        concentration.trial,
        concentration.test,
    )

    assert advection.validate().is_valid
    assert advection.summary()["metadata"]["velocity"] == (2.0, 0.5)
    assert supg.summary()["metadata"]["method"] == "SUPG"
    assert burgers.validate().is_valid
    assert burgers.family == "nonlinear_transport"
    assert reaction.ufl_shape == ()
    assert logistic.ufl_shape == ()


def test_flow_and_fourth_order_operators_are_public_composable_assets():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    mixed = fields.velocity_pressure(domain)
    velocity, pressure = mixed.trial
    test_velocity, test_pressure = mixed.test
    scalar = fields.scalar_unknown(domain, name="u")

    flow = operators.combine(
        operators.viscous_flow_operator(velocity, test_velocity, 0.1),
        operators.pressure_coupling_operator(pressure, test_velocity),
        operators.incompressibility_operator(velocity, test_pressure),
        name="A_stokes",
        kind="mixed_stokes_operator",
    )
    convection = operators.convective_momentum_operator(
        mixed.velocity.value,
        velocity,
        test_velocity,
    )
    laplacian = operators.split_laplacian_operator(
        scalar.trial,
        scalar.test,
    )

    assert flow.validate().is_valid
    assert convection.family == "incompressible_flow"
    assert laplacian.validate().is_valid
    assert mixed.summary()["stability_family"] == "Taylor-Hood"


def test_advective_intrinsic_time_scale_rejects_zero_velocity():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
    )

    with pytest.raises(ValueError, match="nonzero advection velocity"):
        operators.intrinsic_time_scale(domain, (0.0, 0.0))


def test_advection_operator_accepts_a_symbolic_velocity_field():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
    )
    concentration = fields.scalar_unknown(domain, name="concentration")
    coordinate = ufl.SpatialCoordinate(domain)
    velocity = ufl.as_vector((1.0 + coordinate[0], coordinate[1]))

    advection = operators.advection_operator(
        concentration.trial,
        concentration.test,
        velocity,
    )

    assert advection.validate().is_valid
    assert isinstance(advection.metadata["velocity"], str)


def test_incompressible_flow_unknown_and_operators_are_explicit():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    flow = fields.velocity_pressure(domain)
    velocity, pressure = flow.trial
    test_velocity, test_pressure = flow.test
    parts = (
        operators.viscous_flow_operator(velocity, test_velocity, 0.1),
        operators.pressure_coupling_operator(pressure, test_velocity),
        operators.incompressibility_operator(velocity, test_pressure),
        operators.convective_momentum_operator(
            flow.velocity.value,
            flow.velocity.value,
            test_velocity,
        ),
    )

    assert flow.summary()["stability_family"] == "Taylor-Hood"
    assert flow.summary()["velocity_degree"] == 2
    assert flow.summary()["pressure_degree"] == 1
    assert all(operator.validate().is_valid for operator in parts)
    assert [operator.role for operator in parts] == [
        "matrix",
        "matrix",
        "matrix",
        "residual",
    ]
