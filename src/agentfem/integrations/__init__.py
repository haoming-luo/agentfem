"""Versioned adapters for external scientific contracts, loaded on demand."""

from importlib import import_module as _import_module


def __getattr__(name: str):
    if name == "pdeagent_bench":
        module = _import_module(f"{__name__}.pdeagent_bench")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["pdeagent_bench"]
