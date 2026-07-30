from __future__ import annotations

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
