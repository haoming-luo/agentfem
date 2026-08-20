"""Evidence-based platform and runtime support reporting."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
import platform as _platform
from pathlib import Path
import shutil
import subprocess
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
    operating_system: dict[str, str]
    packages: dict[str, str | None]
    mpi: dict[str, object]
    numerics: dict[str, object]
    optional: tuple[dependencies.DependencyStatus, ...]
    execution: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "platform": self.platform.summary(),
            "python": self.python,
            "machine": self.machine,
            "operating_system": dict(self.operating_system),
            "packages": dict(self.packages),
            "mpi": dict(self.mpi),
            "numerics": dict(self.numerics),
            "optional": tuple(item.summary() for item in self.optional),
            "execution": dict(self.execution),
        }

    def format(self) -> str:
        lines = [self.platform.format(), f"  Python: {self.python}", f"  machine: {self.machine}"]
        lines.append(
            "  OS: "
            f"{self.operating_system['system']} {self.operating_system['release']}"
        )
        lines.extend(
            f"  {name}: {version or 'not installed'}"
            for name, version in self.packages.items()
        )
        lines.append(f"  MPI vendor: {self.mpi['vendor']}")
        lines.append(f"  MPI ranks: {self.mpi['rank_count']}")
        lines.append(f"  PETSc scalar: {self.numerics['petsc_scalar_type']}")
        lines.append(f"  MPI launcher: {self.mpi['recommended_launcher']}")
        lines.append(f"  Python executable: {self.execution['python_executable']}")
        lines.append(f"  imported AgentFEM: {self.execution['imported_package']}")
        lines.append(
            "  installed distribution: "
            f"{self.execution['installed_distribution'] or 'not installed'}"
        )
        lines.append(f"  runtime mode: {self.execution['mode']}")
        if self.execution["distribution_mismatch"]:
            lines.append(
                "  warning: the imported AgentFEM package differs from the installed "
                "distribution; a source checkout may be shadowing the environment"
            )
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
        "fenics-basix",
        "fenics-ffcx",
        "numpy",
        "mpi4py",
        "petsc4py",
        "h5py",
    )
    return RuntimeReport(
        platform=current_support(),
        python=_platform.python_version(),
        machine=_platform.machine(),
        operating_system={
            "system": _platform.system(),
            "release": _platform.release(),
            "version": _platform.version(),
        },
        packages={
            "agentfem": runtime_version,
            **{name: _version(name) for name in core_packages},
        },
        mpi=_mpi_runtime(),
        numerics=_numeric_runtime(),
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
        execution=_execution_identity(),
    )


def _execution_identity() -> dict[str, object]:
    """Identify the exact code and interpreter used by the current process."""

    imported_package = Path(__file__).resolve().parent
    distribution_package = None
    try:
        distribution = metadata.distribution("agentfem")
        candidate = Path(distribution.locate_file("agentfem")).resolve()
        if candidate.exists():
            distribution_package = candidate
    except metadata.PackageNotFoundError:
        pass
    mismatch = (
        distribution_package is not None
        and distribution_package != imported_package
    )
    source_root = _source_root(imported_package)
    source_identity = _git_identity(source_root)
    distribution_identity = _distribution_identity()
    return {
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "imported_package": str(imported_package),
        "installed_distribution": (
            None if distribution_package is None else str(distribution_package)
        ),
        "distribution_mismatch": mismatch,
        "mode": "source_checkout" if source_root is not None else "installed_distribution",
        "source_root": None if source_root is None else str(source_root),
        "source": source_identity,
        "distribution": distribution_identity,
    }


def _mpi_runtime() -> dict[str, object]:
    """Describe the mpi4py vendor and avoid PATH launcher mismatches."""

    try:
        from mpi4py import MPI

        vendor_name, vendor_version = MPI.get_vendor()
        vendor = f"{vendor_name} {'.'.join(str(item) for item in vendor_version)}"
        rank_count = int(MPI.COMM_WORLD.size)
    except Exception as exc:
        vendor = f"unavailable ({type(exc).__name__}: {exc})"
        rank_count = 1
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
        "rank_count": rank_count,
        "environment_launcher": (
            str(environment_launcher) if environment_launcher.is_file() else None
        ),
        "path_launcher": path_launcher,
        "recommended_launcher": recommended,
        "path_mismatch": mismatch,
    }


def _numeric_runtime() -> dict[str, object]:
    """Return the scalar and floating-point contract used by this process."""

    import numpy as np

    petsc_scalar = "unavailable"
    petsc_real = "unavailable"
    try:
        from petsc4py import PETSc

        petsc_scalar = np.dtype(PETSc.ScalarType).name
        petsc_real = np.dtype(PETSc.RealType).name
    except Exception:
        pass
    return {
        "python_float_mantissa_bits": int(sys.float_info.mant_dig),
        "numpy_default_float": np.dtype(float).name,
        "petsc_scalar_type": petsc_scalar,
        "petsc_real_type": petsc_real,
        "petsc_complex": bool(np.issubdtype(np.dtype(petsc_scalar), np.complexfloating))
        if petsc_scalar != "unavailable"
        else None,
    }


def _source_root(imported_package: Path) -> Path | None:
    """Locate a ``src``-layout checkout without assuming package-root metadata."""

    for candidate in imported_package.parents:
        if not (candidate / "pyproject.toml").is_file():
            continue
        try:
            if (candidate / "src" / "agentfem").resolve() == imported_package:
                return candidate
        except OSError:
            continue
    return None


def _git_identity(root: Path | None) -> dict[str, object] | None:
    if root is None or not (root / ".git").exists():
        return None
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "commit": commit,
        "tracked_dirty": bool(status.strip()),
        "package_tree_sha256": _source_tree_digest(root / "src" / "agentfem"),
    }


def _source_tree_digest(package_root: Path) -> str | None:
    """Hash the importable source tree without including build/cache noise."""

    if not package_root.is_dir():
        return None
    checksum = sha256()
    selected = tuple(
        path
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in selected:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        checksum.update(len(relative).to_bytes(8, byteorder="big"))
        checksum.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                checksum.update(chunk)
    return checksum.hexdigest()


def _distribution_identity() -> dict[str, object] | None:
    """Record installed-distribution evidence without inventing a wheel hash."""

    try:
        distribution = metadata.distribution("agentfem")
    except metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    record = distribution.read_text("RECORD")
    parsed_url = None
    if direct_url:
        try:
            parsed_url = json.loads(direct_url)
        except json.JSONDecodeError:
            parsed_url = {"unparsed": direct_url}
    return {
        "version": distribution.version,
        "installer": distribution.read_text("INSTALLER"),
        "direct_url": parsed_url,
        "record_sha256": (
            None if record is None else sha256(record.encode("utf-8")).hexdigest()
        ),
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
