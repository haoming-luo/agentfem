from __future__ import annotations

from agentfem.backends import (
    BackendAdapter,
    BackendDescriptor,
    available_backends,
    get_backend,
    register_backend,
)


class _RecordingBackend(BackendAdapter):
    def __init__(self):
        self.calls = []

    @property
    def descriptor(self):
        return BackendDescriptor(
            name="recording",
            version="test",
            capabilities=("form_compilation",),
        )

    def compile_form(self, expression):
        self.calls.append(("compile", expression))
        return ("compiled", expression)

    def assemble_matrix(self, expression, *, bcs=None):
        self.calls.append(("matrix", expression, bcs))
        return ("matrix", expression)

    def assemble_vector(self, expression):
        self.calls.append(("vector", expression))
        return ("vector", expression)


def test_fenicsx_is_the_explicit_default_registered_backend():
    assert "fenicsx" in available_backends()
    descriptor = get_backend("fenicsx").descriptor
    assert descriptor.name == "fenicsx"
    assert descriptor.supports("matrix_assembly")


def test_backend_factories_are_lazy_and_explicit():
    created = []

    def factory():
        created.append(True)
        return _RecordingBackend()

    register_backend("recording_test", factory, overwrite=True)
    assert created == []

    backend = get_backend("recording_test")

    assert created == [True]
    assert backend.compile_form("form") == ("compiled", "form")
