"""FEniCSx/DOLFINx lowering adapter."""

from __future__ import annotations

from functools import cached_property

from .base import BackendAdapter, BackendDescriptor


class FEniCSxBackend(BackendAdapter):
    """Current production backend for AgentFEM operator forms."""

    @cached_property
    def descriptor(self) -> BackendDescriptor:
        from .runtime import current_runtime

        runtime = current_runtime()
        try:
            import dolfinx

            version = getattr(dolfinx, "__version__", "unknown")
        except ImportError:
            version = "unavailable"
        return BackendDescriptor(
            name="fenicsx",
            version=str(version),
            capabilities=runtime.capabilities,
            status="available" if version != "unavailable" else "unavailable",
            notes=(
                f"Primary AgentFEM execution backend through {runtime.name}. "
                f"{runtime.notes}"
            ),
        )

    def compile_form(self, expression):
        from agentfem import assembly

        return assembly.make_form(expression)

    def assemble_matrix(self, expression, *, bcs=None):
        from agentfem import assembly

        return assembly.assemble_matrix(self.compile_form(expression), bcs=bcs)

    def assemble_vector(self, expression):
        from agentfem import assembly

        return assembly.assemble_vector(self.compile_form(expression))


__all__ = ["FEniCSxBackend"]
