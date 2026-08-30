"""Runtime capability discovery for FEniCSx execution.

AgentFEM exposes one scientific API while allowing more than one numerical
runtime. PETSc remains the distributed, full-capability route. DOLFINx native
assembly plus SciPy (and optionally PyAMG) provides a useful serial route on
platforms where petsc4py is not packaged, notably native Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
import os


class RuntimeCapabilityError(RuntimeError):
    """Raised before lowering when the active runtime cannot execute a step."""

    code = "AFM-BACKEND-CAPABILITY-001"

    def __init__(self, required, *, operation: str | None = None):
        runtime = current_runtime()
        missing = tuple(item for item in required if item not in runtime.capabilities)
        label = "operation" if operation is None else operation
        super().__init__(
            f"{self.code}: {label} requires {missing!r}, but runtime "
            f"{runtime.name!r} provides {runtime.capabilities!r}. "
            "Use the full FEniCSx/PETSc runtime (Linux, macOS, WSL2, or a "
            "supported cluster), or select a supported procedure."
        )
        self.required = tuple(required)
        self.missing = missing
        self.operation = operation
        self.runtime = runtime


class RuntimeSelectionError(RuntimeError):
    """Raised when an explicitly requested numerical runtime is unavailable."""

    code = "AFM-BACKEND-RUNTIME-001"


@dataclass(frozen=True)
class RuntimeProfile:
    """Detected numerical runtime and its explicitly supported capabilities."""

    name: str
    capabilities: tuple[str, ...]
    packages: dict[str, bool]
    distributed: bool
    notes: str

    def supports(self, *capabilities: str) -> bool:
        return all(item in self.capabilities for item in capabilities)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "packages": dict(self.packages),
            "distributed": self.distributed,
            "notes": self.notes,
        }


def _available(module: str) -> bool:
    """Return whether an optional runtime module can actually be imported.

    ``find_spec`` alone is insufficient for compiled Python packages: on
    Windows a package can be present while one of its DLL dependencies is
    missing.  Treat that state as unavailable so runtime selection fails
    before numerical lowering instead of crashing at the first assembly.
    """

    try:
        if find_spec(module) is None:
            return False
        import_module(module)
        return True
    except (ImportError, ModuleNotFoundError, OSError, ValueError):
        return False


def current_runtime() -> RuntimeProfile:
    """Return the active FEniCSx runtime after import-level capability probes."""

    raw_request = os.environ.get("AGENTFEM_RUNTIME", "").strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "native": "fenicsx-native-serial",
        "fenicsx-native-serial": "fenicsx-native-serial",
        "petsc": "fenicsx-petsc",
        "fenicsx-petsc": "fenicsx-petsc",
    }
    if raw_request not in aliases:
        raise RuntimeSelectionError(
            f"{RuntimeSelectionError.code}: unknown AGENTFEM_RUNTIME "
            f"value {raw_request!r}; choose 'auto', 'fenicsx-petsc', or "
            "'fenicsx-native-serial'."
        )
    requested = aliases[raw_request]
    packages = {
        "dolfinx": _available("dolfinx"),
        "petsc4py": _available("petsc4py"),
        "dolfinx_fem_petsc": _available("dolfinx.fem.petsc"),
        "scipy": _available("scipy"),
        "pyamg": _available("pyamg"),
        "dolfinx_mpc": _available("dolfinx_mpc"),
    }
    if not packages["dolfinx"]:
        if requested != "auto":
            raise RuntimeSelectionError(
                f"{RuntimeSelectionError.code}: AGENTFEM_RUNTIME requests "
                f"{requested!r}, but DOLFINx is unavailable."
            )
        return RuntimeProfile(
            "unavailable", (), packages, False, "DOLFINx is not installed."
        )
    common = [
        "ufl_form_compilation",
        "matrix_assembly",
        "vector_assembly",
        "xdmf_output",
    ]
    force_native = requested == "fenicsx-native-serial"
    force_petsc = requested == "fenicsx-petsc"
    if force_native and not packages["scipy"]:
        raise RuntimeSelectionError(
            f"{RuntimeSelectionError.code}: AGENTFEM_RUNTIME requests the "
            "native serial runtime, but SciPy is unavailable."
        )
    petsc_ready = packages["petsc4py"] and packages["dolfinx_fem_petsc"]
    if force_petsc and not petsc_ready:
        raise RuntimeSelectionError(
            f"{RuntimeSelectionError.code}: AGENTFEM_RUNTIME requests the "
            "PETSc runtime, but petsc4py and the DOLFINx PETSc adapter are "
            "not both importable."
        )
    if petsc_ready and not force_native:
        capabilities = common + [
            "linear_solve",
            "nonlinear_solve",
            "petsc_linear_solve",
            "petsc_nonlinear_solve",
            "mpi_distributed_mesh",
        ]
        if packages["dolfinx_mpc"]:
            capabilities.append("exact_mpc")
        return RuntimeProfile(
            "fenicsx-petsc",
            tuple(capabilities),
            packages,
            True,
            "Full PETSc-backed AgentFEM runtime.",
        )
    if packages["scipy"]:
        capabilities = common + ["linear_solve", "native_serial_linear_solve"]
        if packages["pyamg"]:
            capabilities.append("pyamg_linear_solve")
        return RuntimeProfile(
            "fenicsx-native-serial",
            tuple(capabilities),
            packages,
            False,
            "PETSc-free serial runtime using DOLFINx native assembly and SciPy/PyAMG.",
        )
    return RuntimeProfile(
        "fenicsx-assembly-only",
        tuple(common),
        packages,
        False,
        "DOLFINx is present, but neither PETSc nor SciPy can solve systems.",
    )


def require_capabilities(*capabilities: str, operation: str | None = None) -> None:
    """Fail closed with a stable error when capabilities are unavailable."""

    if not current_runtime().supports(*capabilities):
        raise RuntimeCapabilityError(capabilities, operation=operation)


__all__ = [
    "RuntimeCapabilityError",
    "RuntimeProfile",
    "RuntimeSelectionError",
    "current_runtime",
    "require_capabilities",
]
