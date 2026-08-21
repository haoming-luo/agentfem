"""Safe mathematical expressions for scientific model inputs.

The public API deliberately accepts a small mathematical language rather than
Python source.  It is useful for configuration files, experiment tables, and
agent-authored models where ``eval`` would be unsafe and ordinary callables
would not be serializable or inspectable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
import ufl


class ExpressionError(ValueError):
    """Raised when a scientific expression is invalid or unsupported."""


_FUNCTIONS = {
    "sin": ufl.sin,
    "cos": ufl.cos,
    "tan": ufl.tan,
    "tanh": ufl.tanh,
    "exp": ufl.exp,
    "sqrt": ufl.sqrt,
    "log": ufl.ln,
    "ln": ufl.ln,
    "abs": abs,
}

_NUMPY_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "ln": np.log,
    "abs": np.abs,
}

_RESERVED_PARAMETER_NAMES = {"x", "y", "z", "pi"}


@dataclass(frozen=True)
class ScientificExpression:
    """An inspectable expression that can be lowered to UFL.

    Only arithmetic, coordinate names ``x/y/z``, ``t``, ``pi``, declared
    parameters, and a small allow-list of mathematical functions are accepted.
    Attribute access, indexing, comprehensions, and arbitrary function calls
    are rejected before any finite-element form is compiled.
    """

    source: str

    def __post_init__(self) -> None:
        selected = str(self.source).strip()
        if not selected:
            raise ExpressionError("A scientific expression cannot be empty.")
        try:
            tree = ast.parse(selected, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Invalid scientific expression: {selected!r}.") from exc
        _validate(tree)
        object.__setattr__(self, "source", selected)

    def ufl(self, domain, *, parameters: Mapping[str, object] | None = None):
        """Lower the expression to a UFL expression on ``domain``."""

        coordinates = ufl.SpatialCoordinate(domain)
        names = _coordinate_names(coordinates)
        names["pi"] = float(np.pi)
        names["t"] = 0.0
        for name, value in dict(parameters or {}).items():
            if (
                not str(name).isidentifier()
                or name in _FUNCTIONS
                or name in _RESERVED_PARAMETER_NAMES
            ):
                raise ExpressionError(f"Invalid expression parameter name {name!r}.")
            names[str(name)] = value
        tree = ast.parse(self.source, mode="eval")
        return _lower(tree.body, names)

    def evaluate(
        self,
        coordinates,
        *,
        parameters: Mapping[str, object] | None = None,
    ):
        """Evaluate safely on coordinate arrays without Python ``eval`` or JIT.

        ``coordinates`` follows the DOLFINx interpolation convention with
        shape ``(geometric_dimension, number_of_points)``. This route is for
        known coefficient and load fields; :meth:`ufl` remains the route for
        symbolic unknowns and variational forms.
        """

        selected = np.asarray(coordinates)
        if selected.ndim != 2 or selected.shape[0] < 1:
            raise ExpressionError(
                "Expression coordinates must have shape (dimension, points)."
            )
        names = {"x": selected[0], "pi": float(np.pi), "t": 0.0}
        if selected.shape[0] >= 2:
            names["y"] = selected[1]
        if selected.shape[0] >= 3:
            names["z"] = selected[2]
        for name, value in dict(parameters or {}).items():
            if (
                not str(name).isidentifier()
                or name in _FUNCTIONS
                or name in _RESERVED_PARAMETER_NAMES
            ):
                raise ExpressionError(f"Invalid expression parameter name {name!r}.")
            names[str(name)] = _numerical_parameter(value)
        tree = ast.parse(self.source, mode="eval")
        value = _lower_numpy(tree.body, names)
        return _point_values(value, selected.shape[1])

    def summary(self) -> dict[str, object]:
        """Return the serializable public contract."""

        return {
            "kind": "scientific_expression",
            "source": self.source,
            "language": "agentfem-math-v1",
        }


def expression(source: str | Real | ScientificExpression) -> ScientificExpression:
    """Return a validated :class:`ScientificExpression`."""

    if isinstance(source, ScientificExpression):
        return source
    if isinstance(source, Real):
        return ScientificExpression(repr(float(source)))
    return ScientificExpression(str(source))


def as_ufl(source, domain, *, parameters: Mapping[str, object] | None = None):
    """Validate and lower one scalar expression to UFL."""

    return expression(source).ufl(domain, parameters=parameters)


def vector_as_ufl(
    sources: Sequence[str | Real | ScientificExpression],
    domain,
    *,
    parameters: Mapping[str, object] | None = None,
):
    """Validate and lower a vector of scalar expressions to UFL."""

    values = tuple(sources)
    dimension = int(domain.geometry.dim)
    if len(values) != dimension:
        raise ExpressionError(
            f"Vector expression has {len(values)} components; expected {dimension}."
        )
    return ufl.as_vector(
        [as_ufl(value, domain, parameters=parameters) for value in values]
    )


def interpolate(
    target,
    source,
    *,
    parameters: Mapping[str, object] | None = None,
) -> object:
    """Interpolate a validated scalar or vector expression into ``target``."""

    function = getattr(target, "value", target)
    if not hasattr(function, "function_space"):
        raise TypeError("expressions.interpolate target must be a finite-element field.")
    shape = tuple(getattr(function, "ufl_shape", ()))
    if shape:
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
            raise ExpressionError("A vector field requires one expression per component.")
        selected = tuple(expression(value) for value in source)
        if len(selected) != int(shape[0]):
            raise ExpressionError(
                f"Vector expression has {len(selected)} components; expected {shape[0]}."
            )

        def evaluate_points(points):
            return np.vstack(
                [item.evaluate(points, parameters=parameters) for item in selected]
            )

    else:
        selected = expression(source)

        def evaluate_points(points):
            return selected.evaluate(points, parameters=parameters)

    function.interpolate(evaluate_points)
    function.x.scatter_forward()
    return target


def _coordinate_names(coordinates) -> dict[str, object]:
    dimension = int(coordinates.ufl_shape[0])
    values = {"x": coordinates[0]}
    if dimension >= 2:
        values["y"] = coordinates[1]
    if dimension >= 3:
        values["z"] = coordinates[2]
    return values


def _validate(tree: ast.AST) -> None:
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ExpressionError(
                f"Unsupported syntax {type(node).__name__} in scientific expression."
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS
                or node.keywords
                or len(node.args) != 1
            ):
                raise ExpressionError(
                    "Only one-argument calls to "
                    f"{tuple(sorted(_FUNCTIONS))} are allowed."
                )
        if isinstance(node, ast.Constant) and not isinstance(node.value, Real):
            raise ExpressionError("Scientific expression constants must be real numbers.")


def _lower(node: ast.AST, names: Mapping[str, object]):
    if isinstance(node, ast.Constant):
        return ufl.as_ufl(float(node.value))
    if isinstance(node, ast.Name):
        try:
            return names[node.id]
        except KeyError as exc:
            raise ExpressionError(
                f"Unknown name {node.id!r} in scientific expression."
            ) from exc
    if isinstance(node, ast.UnaryOp):
        value = _lower(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _lower(node.left, names)
        right = _lower(node.right, names)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
        }
        return operations[type(node.op)]()
    if isinstance(node, ast.Call):
        return _FUNCTIONS[node.func.id](_lower(node.args[0], names))
    raise ExpressionError(f"Unsupported expression node {type(node).__name__}.")


def _lower_numpy(node: ast.AST, names: Mapping[str, object]):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        try:
            return names[node.id]
        except KeyError as exc:
            raise ExpressionError(
                f"Unknown name {node.id!r} in scientific expression."
            ) from exc
    if isinstance(node, ast.UnaryOp):
        value = _lower_numpy(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _lower_numpy(node.left, names)
        right = _lower_numpy(node.right, names)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
        }
        return operations[type(node.op)]()
    if isinstance(node, ast.Call):
        return _NUMPY_FUNCTIONS[node.func.id](_lower_numpy(node.args[0], names))
    raise ExpressionError(f"Unsupported expression node {type(node).__name__}.")


def _numerical_parameter(value):
    selected = getattr(value, "value", value)
    array = np.asarray(selected)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array


def _point_values(value, points: int) -> np.ndarray:
    selected = np.asarray(value)
    if selected.ndim == 0:
        return np.full(points, float(selected))
    return np.broadcast_to(selected, (points,))


__all__ = [
    "ExpressionError",
    "ScientificExpression",
    "as_ufl",
    "expression",
    "interpolate",
    "vector_as_ufl",
]
