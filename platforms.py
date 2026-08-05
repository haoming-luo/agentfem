"""Evidence-based platform and runtime support reporting."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import platform as _platform
from pathlib import Path
import shutil
import sys

from . import dependencies


@dataclass(frozen=True)
class PlatformSupport:
    """One operating-system support decision with explicit limitations."""

    system: str
    route: str
    level: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    @property
    def recommended(self) -> bool:
        return self.level in {"ci_verified", "developer_verified", "recommended"}

    def summary(self) -> dict[str, object]:
        return {
            "system": self.system,
            "route": self.route,
            "level": self.level,
            "recommended": self.recommended,
            "evidence": self.evidence,
            "limitations": self.limitations,
        }

    def format(self) -> str:
        lines = [
            f"AgentFEM platform: {self.route}",
            f"  support level: {self.level}",
        ]
        lines.extend(f"  evidence: {item}" for item in self.evidence)
        lines.extend(f"  limitation: {item}" for item in self.limitations)
        return "\n".join(lines)


@dataclass(frozen=True)
class RuntimeReport:
    """Compact runtime inventory for bug reports and agent inspection."""

    platform: PlatformSupport
    python: str
    machine: str
    packages: dict[str, str | None]
    mpi: dict[str, object]
    optional: tuple[dependencies.DependencyStatus, ...]

    def summary(self) -> dict[str, object]:
        return {
            "platform": self.platform.summary(),
            "python": self.python,
            "machine": self.machine,
            "packages": dict(self.packages),
            "mpi": dict(self.mpi),
            "optional": tuple(item.summary() for item in self.optional),
        }

    def format(self) -> str:
        lines = [self.platform.format(), f"  Python: {self.python}", f"  machine: {self.machine}"]
        lines.extend(
            f"  {name}: {version or 'not installed'}"
            for name, version in self.packages.items()
        )
        lines.append(f"  MPI vendor: {self.mpi['vendor']}")
        lines.append(f"  MPI launcher: {self.mpi['recommended_launcher']}")
        if self.mpi["path_mismatch"]:
            lines.append(
                "  warning: PATH mpiexec differs from the active environment; "
                "AgentFEM will use the environment launcher"
            )
        lines.extend(
            f"  optional {item.package}: "
            f"{item.version if item.available else 'not installed'}"
            for item in self.optional
        )
        return "\n".join(lines)


def support_for(
    system: str,
    *,
    wsl: bool = False,
) -> PlatformSupport:
    """Return the first-release support tier for an operating-system route."""

    selected = str(system).strip().lower()
    if selected == "linux" and wsl:
        return PlatformSupport(
            system="Windows",
            route="Windows via WSL2/Linux",
            level="recommended",
            evidence=(
                "uses the same conda-forge Linux FEniCSx/PETSc/MPI stack as CI",
                "all AgentFEM public workflows are filesystem- and terminal-based",
            ),
            limitations=(
                "GPU, GUI, ParaView, and host-file integration depend on the local WSL setup",
            ),
        )
    if selected == "linux":
        return PlatformSupport(
            system="Linux",
            route="native Linux",
            level="ci_verified",
            evidence=("full serial, MPI, wheel, and release smoke gates run on Ubuntu",),
        )
    if selected == "darwin":
        return PlatformSupport(
            system="macOS",
            route="native macOS with conda-forge/Homebrew-compatible FEniCSx stack",
            level="developer_verified",
            evidence=("maintainer development and release verification run on Apple Silicon",),
            limitations=("MPI launcher must match the mpi4py implementation",),
        )
    if selected == "windows":
        return PlatformSupport(
            system="Windows",
            route="native Windows",
            level="experimental",
            evidence=("FEniCSx 0.11 has conda-forge win-64 builds",),
            limitations=(
                "AgentFEM has no native-Windows CI gate",
                "the current PETSc-based solver path is not available as a standard conda-forge win-64 stack",
                "dolfinx_mpc 0.11 is unavailable on conda-forge win-64, so the distributed Abaqus equation workflow is excluded",
            ),
        )
    return PlatformSupport(
        system=system,
        route=f"native {system}",
        level="unverified",
        evidence=(),
        limitations=("No AgentFEM release evidence is recorded for this platform.",),
    )


def current_support() -> PlatformSupport:
    """Detect the current OS, including Windows Subsystem for Linux."""

    system = _platform.system()
    return support_for(system, wsl=system == "Linux" and _is_wsl())


def runtime_report() -> RuntimeReport:
    """Return versions and optional integrations useful in issue reports."""

    # Report the code that is actually executing.  Development checkouts can
    # legitimately shadow an older installed distribution, in which case
    # importlib.metadata would otherwise describe the wrong AgentFEM runtime.
    from . import __version__ as runtime_version

    core_packages = (
        "fenics-dolfinx",
        "fenics-ufl",
        "numpy",
        "mpi4py",
        "petsc4py",
        "h5py",
    )
    return RuntimeReport(
        platform=current_support(),
        python=_platform.python_version(),
        machine=_platform.machine(),
        packages={
            "agentfem": runtime_version,
            **{name: _version(name) for name in core_packages},
        },
        mpi=_mpi_runtime(),
        optional=(
            dependencies.status(
                "meshio",
                extra="mesh-formats",
                capability="External CAE mesh conversion",
            ),
            dependencies.status(
                "gmsh",
                extra="gmsh",
                capability="Gmsh model and .msh import",
            ),
            dependencies.status(
                "pyvista",
                extra="visualization",
                capability="In-process visualisation",
            ),
            dependencies.status(
                "torch",
                extra="ml",
                capability="PyTorch dataset and learning adapters",
            ),
            dependencies.status(
                "dolfinx_mpc",
                extra="parallel-mpc",
                capability="Distributed multi-point constraints",
            ),
        ),
    )


def _mpi_runtime() -> dict[str, object]:
    """Describe the mpi4py vendor and avoid PATH launcher mismatches."""

    try:
        from mpi4py import MPI

        vendor_name, vendor_version = MPI.get_vendor()
        vendor = f"{vendor_name} {'.'.join(str(item) for item in vendor_version)}"
    except Exception as exc:
        vendor = f"unavailable ({type(exc).__name__}: {exc})"
    environment_launcher = Path(sys.prefix) / "bin" / "mpiexec"
    path_launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    recommended = (
        str(environment_launcher)
        if environment_launcher.is_file()
        else path_launcher
    )
    mismatch = False
    if environment_launcher.is_file() and path_launcher is not None:
        try:
            mismatch = not Path(path_launcher).samefile(environment_launcher)
        except OSError:
            mismatch = str(Path(path_launcher).resolve()) != str(environment_launcher.resolve())
    return {
        "vendor": vendor,
        "environment_launcher": (
            str(environment_launcher) if environment_launcher.is_file() else None
        ),
        "path_launcher": path_launcher,
        "recommended_launcher": recommended,
        "path_mismatch": mismatch,
    }


def _is_wsl() -> bool:
    try:
        text = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in text or "wsl" in text


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


__all__ = [
    "PlatformSupport",
    "RuntimeReport",
    "current_support",
    "runtime_report",
    "support_for",
]
