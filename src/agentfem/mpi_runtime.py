"""MPI launcher discovery that is consistent with the active Python runtime.

MPI launchers are not interchangeable.  In particular, an Open MPI
``mpiexec`` can appear first on ``PATH`` while ``mpi4py`` and PETSc in the
active conda environment are linked against MPICH.  This module makes that
compatibility an explicit runtime contract shared by the CLI, ``doctor`` and
automation instead of relying on users to remember an environment-specific
absolute path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MPILauncher:
    """One discovered MPI process launcher and its implementation family."""

    path: str
    source: str
    family: str
    version_line: str | None
    compatible: bool

    def summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source": self.source,
            "family": self.family,
            "version_line": self.version_line,
            "compatible": self.compatible,
        }


@dataclass(frozen=True)
class MPIRuntimeAudit:
    """Compatibility decision between mpi4py and available launchers."""

    vendor_name: str
    vendor_version: tuple[int, ...]
    family: str
    rank_count: int
    launchers: tuple[MPILauncher, ...]
    selected_launcher: str | None
    path_launcher: str | None
    code: str
    message: str

    @property
    def compatible(self) -> bool:
        return self.selected_launcher is not None

    @property
    def vendor(self) -> str:
        version = ".".join(str(item) for item in self.vendor_version)
        return f"{self.vendor_name} {version}".strip()

    @property
    def path_mismatch(self) -> bool:
        return bool(
            self.path_launcher
            and self.selected_launcher
            and not _same_path(self.path_launcher, self.selected_launcher)
        )

    def summary(self) -> dict[str, object]:
        environment_launcher = next(
            (
                item.path
                for item in self.launchers
                if item.source == "python_environment"
            ),
            None,
        )
        return {
            "vendor": self.vendor,
            "vendor_name": self.vendor_name,
            "vendor_version": self.vendor_version,
            "family": self.family,
            "rank_count": self.rank_count,
            "launchers": tuple(item.summary() for item in self.launchers),
            "environment_launcher": environment_launcher,
            "path_launcher": self.path_launcher,
            "recommended_launcher": self.selected_launcher,
            "selected_launcher": self.selected_launcher,
            "compatible": self.compatible,
            "path_mismatch": self.path_mismatch,
            "code": self.code,
            "message": self.message,
        }


class MPILauncherError(RuntimeError):
    """Raised before execution when no compatible MPI launcher is available."""

    def __init__(self, audit: MPIRuntimeAudit):
        super().__init__(f"{audit.code}: {audit.message}")
        self.audit = audit


def _mpi_family(name: str) -> str:
    selected = str(name).strip().lower()
    if "open mpi" in selected or "openmpi" in selected:
        return "openmpi"
    if "mpich" in selected or "hydra" in selected:
        return "mpich"
    if "intel" in selected and "mpi" in selected:
        return "intelmpi"
    if "microsoft" in selected or "ms-mpi" in selected or "msmpi" in selected:
        return "msmpi"
    return "unknown"


def _launcher_family(output: str) -> str:
    lowered = output.lower()
    if "open mpi" in lowered or "open-mpi" in lowered or "openmpi" in lowered:
        return "openmpi"
    if "openrte" in lowered or "orterun" in lowered or "prte" in lowered:
        return "openmpi"
    if "intel(r) mpi" in lowered or "intel mpi" in lowered:
        return "intelmpi"
    # MPICH launchers normally identify the Hydra process manager rather than
    # spelling MPICH on their first line.
    if "hydra build details" in lowered or "process manager:" in lowered:
        return "mpich"
    if "microsoft mpi" in lowered or "ms-mpi" in lowered:
        return "msmpi"
    return "unknown"


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).samefile(right)
    except OSError:
        return Path(left).resolve() == Path(right).resolve()


def _unique_candidates(
    candidates: Iterable[tuple[str | Path | None, str]],
) -> tuple[tuple[str, str], ...]:
    selected: list[tuple[str, str]] = []
    for candidate, source in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if any(_same_path(resolved, existing) for existing, _ in selected):
            continue
        selected.append((resolved, source))
    return tuple(selected)


@lru_cache(maxsize=16)
def _inspect_launcher(path: str) -> tuple[str, str | None]:
    try:
        completed = subprocess.run(
            (path, "--version"),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        output = "\n".join(item for item in (completed.stdout, completed.stderr) if item)
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    lines = tuple(line.strip() for line in output.splitlines() if line.strip())
    return _launcher_family(output), (lines[0] if lines else None)


def audit_mpi_runtime(
    *,
    python_executable: str | Path | None = None,
    vendor: tuple[str, Sequence[int]] | None = None,
    rank_count: int | None = None,
    path_mpiexec: str | None = None,
    path_mpirun: str | None = None,
) -> MPIRuntimeAudit:
    """Audit launchers against the MPI implementation used by ``mpi4py``.

    The optional arguments are intentionally public so installers and tests can
    inspect another Python environment without mutating ``PATH``.
    """

    if vendor is None or rank_count is None:
        try:
            from mpi4py import MPI

            if vendor is None:
                vendor = MPI.get_vendor()
            if rank_count is None:
                rank_count = int(MPI.COMM_WORLD.size)
        except Exception:
            vendor = vendor or ("unavailable", ())
            rank_count = 1 if rank_count is None else rank_count
    vendor_name, raw_version = vendor
    vendor_version = tuple(int(item) for item in raw_version)
    family = _mpi_family(vendor_name)

    executable = Path(python_executable or sys.executable).expanduser().resolve()
    environment_bin = executable.parent
    if path_mpiexec is None:
        path_mpiexec = shutil.which("mpiexec")
    if path_mpirun is None:
        path_mpirun = shutil.which("mpirun")
    path_launcher = path_mpiexec or path_mpirun
    candidates = _unique_candidates(
        (
            (environment_bin / "mpiexec", "python_environment"),
            (environment_bin / "mpirun", "python_environment"),
            (path_mpiexec, "PATH"),
            (path_mpirun, "PATH"),
        )
    )
    launchers: list[MPILauncher] = []
    for path, source in candidates:
        launcher_family, version_line = _inspect_launcher(path)
        launchers.append(
            MPILauncher(
                path=path,
                source=source,
                family=launcher_family,
                version_line=version_line,
                compatible=family != "unknown" and launcher_family == family,
            )
        )
    compatible = tuple(item for item in launchers if item.compatible)
    selected = compatible[0].path if compatible else None
    if selected is None and not launchers:
        code = "AFM-MPI-LAUNCHER-MISSING"
        message = "No mpiexec or mpirun launcher was found for the active Python environment."
    elif selected is None:
        code = "AFM-MPI-LAUNCHER-MISMATCH"
        available = ", ".join(f"{item.family}:{item.path}" for item in launchers)
        message = (
            f"mpi4py uses {vendor_name} ({family}), but no compatible launcher was found; "
            f"discovered {available}."
        )
    elif path_launcher and not _same_path(path_launcher, selected):
        code = "AFM-MPI-LAUNCHER-RECOVERED"
        message = (
            f"PATH selects {path_launcher}, but AgentFEM selected the compatible "
            f"{family} launcher {selected}."
        )
    else:
        code = "AFM-MPI-LAUNCHER-OK"
        message = f"The selected {family} launcher matches mpi4py."
    return MPIRuntimeAudit(
        vendor_name=str(vendor_name),
        vendor_version=vendor_version,
        family=family,
        rank_count=int(rank_count),
        launchers=tuple(launchers),
        selected_launcher=selected,
        path_launcher=path_launcher,
        code=code,
        message=message,
    )


def compatible_mpi_launcher() -> str:
    """Return a verified launcher or fail before MPI initialization."""

    audit = audit_mpi_runtime()
    if audit.selected_launcher is None:
        raise MPILauncherError(audit)
    return audit.selected_launcher


def mpi_command(ranks: int, command: Sequence[str]) -> tuple[str, ...]:
    """Build an argv-safe MPI command using the verified launcher."""

    if int(ranks) <= 0:
        raise ValueError("MPI rank count must be positive.")
    if not command:
        raise ValueError("An MPI child command is required.")
    return (compatible_mpi_launcher(), "-n", str(int(ranks)), *(str(item) for item in command))


__all__ = [
    "MPILauncher",
    "MPILauncherError",
    "MPIRuntimeAudit",
    "audit_mpi_runtime",
    "compatible_mpi_launcher",
    "mpi_command",
]
