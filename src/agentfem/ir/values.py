"""Scientific coefficient summaries shared by AF-IR-aware public objects."""

from __future__ import annotations

from math import isfinite


def describe_value(value):
    """Return a JSON-safe coefficient value or an explicit opaque marker.

    DOLFINx constants expose their current numerical data through ``value``.
    Arbitrary UFL expressions and callables remain opaque until they have a
    declared semantic extension.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Scientific coefficient values must be finite.")
        return value
    if isinstance(value, complex):
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError("Scientific coefficient values must be finite.")
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (tuple, list)):
        return tuple(describe_value(item) for item in value)

    raw_value = getattr(value, "value", None)
    if raw_value is not None and raw_value is not value:
        return describe_value(raw_value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return describe_value(item_method())
        except (TypeError, ValueError):
            pass
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return describe_value(list_method())
        except (TypeError, ValueError):
            pass
    summary = getattr(value, "summary", None)
    if callable(summary):
        return summary()

    value_type = type(value)
    return {
        "kind": "opaque_coefficient",
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
        "serializable": False,
    }


__all__ = ["describe_value"]
